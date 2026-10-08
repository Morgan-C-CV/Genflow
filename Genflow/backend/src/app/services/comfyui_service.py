"""Turn a Genflow schema into a ComfyUI workflow and push it to the queue."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.modules.comfyui_graph import build_api_graph, build_ui_workflow, rank_similar
from app.repositories.comfyui_repository import (
    ComfyUIError,
    ComfyUIRepository,
    ComfyUIValidationError,
)


class ComfyUIService:
    def __init__(self, repository: Optional[ComfyUIRepository] = None, base_url: str = "http://127.0.0.1:8188"):
        self.repository = repository or ComfyUIRepository(base_url=base_url)

    # -- capability -------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """Report ComfyUI reachability and the assets it can actually run."""
        try:
            stats = self.repository.system_stats()
            capabilities = self.repository.capabilities()
        except ComfyUIError as exc:
            return {
                "reachable": False,
                "base_url": self.repository.base_url,
                "error": str(exc),
                "checkpoints": [],
                "loras": [],
                "samplers": [],
            }

        devices = stats.get("devices") or []
        return {
            "reachable": True,
            "base_url": self.repository.base_url,
            "comfyui_version": (stats.get("system") or {}).get("comfyui_version", ""),
            "device": devices[0].get("name") if devices else "",
            "checkpoints": capabilities["checkpoints"],
            "loras": capabilities["loras"],
            "samplers": capabilities["samplers"],
        }

    # -- build ------------------------------------------------------------
    def build_workflow(
        self,
        schema: Any,
        *,
        width: int = 1024,
        height: int = 1024,
        batch_size: int = 1,
        seed: Optional[int] = None,
        filename_prefix: str = "Genflow",
        title: str = "Genflow Workflow",
        checkpoint_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build API-format and UI-format workflows for ``schema``."""
        warnings: List[str] = []
        try:
            capabilities = self.repository.capabilities()
            checkpoints = capabilities["checkpoints"]
            loras = capabilities["loras"]
            samplers = capabilities["samplers"]
            object_info = capabilities["object_info"]
        except ComfyUIError as exc:
            warnings.append(
                f"Could not read ComfyUI capabilities ({exc}); building the graph with "
                "Genflow values only."
            )
            checkpoints, loras, samplers, object_info = [], [], [], {}

        result = build_api_graph(
            schema,
            checkpoints=checkpoints,
            loras=loras,
            samplers=samplers,
            width=width,
            height=height,
            batch_size=batch_size,
            filename_prefix=filename_prefix,
            seed_override=seed,
            checkpoint_override=checkpoint_override,
        )
        warnings.extend(result.warnings)

        ui_workflow: Dict[str, Any] = {}
        if object_info:
            ui_workflow = build_ui_workflow(result.graph, object_info, title=title)

        return {
            "api_graph": result.graph,
            "ui_workflow": ui_workflow,
            "warnings": warnings,
            "checkpoint": result.checkpoint,
            "requested_checkpoint": str(getattr(schema, "model", "") or "").strip(),
            "checkpoint_resolved": result.checkpoint_resolved,
            "applied_loras": result.applied_loras,
            "unresolved_loras": result.unresolved_loras,
            "controls": result.controls,
            "available_checkpoints": checkpoints,
            "available_loras": loras,
            "remediation": self._remediation_from_build(
                requested_checkpoint=result.checkpoint,
                checkpoint_resolved=result.checkpoint_resolved,
                installed_checkpoints=checkpoints,
                unresolved_loras=result.unresolved_loras,
                installed_loras=loras,
            ),
            "title": title,
        }

    # -- push -------------------------------------------------------------
    def push_workflow(self, schema: Any, **build_kwargs: Any) -> Dict[str, Any]:
        """Build the workflow and submit it to ComfyUI's ``/prompt`` endpoint."""
        built = self.build_workflow(schema, **build_kwargs)
        payload: Dict[str, Any] = {
            **built,
            "pushed": False,
            "prompt_id": "",
            "queue_number": None,
            "node_errors": {},
            "error": "",
        }

        # Always attempt the submission, even when no checkpoint or LoRA matched.
        # The caller may deliberately push anyway; ComfyUI then validates the
        # graph and its node_errors explain exactly which value is unusable.
        try:
            response = self.repository.queue_prompt(built["api_graph"])
        except ComfyUIValidationError as exc:
            payload["error"] = str(exc)
            payload["node_errors"] = exc.node_errors
            # ComfyUI is authoritative about what actually failed, so rebuild the
            # remediation list from its own per-node errors.
            payload["remediation"] = self._remediation_from_failure(
                node_errors=exc.node_errors,
                installed_checkpoints=built["available_checkpoints"],
                installed_loras=built["available_loras"],
            )
            return payload
        except ComfyUIError as exc:
            payload["error"] = str(exc)
            return payload

        payload["pushed"] = True
        payload["prompt_id"] = str(response.get("prompt_id", ""))
        payload["queue_number"] = response.get("number")
        payload["node_errors"] = response.get("node_errors") or {}
        payload["remediation"] = []
        return payload

    # -- remediation ------------------------------------------------------
    @staticmethod
    def _checkpoint_item(requested: str, installed: List[str]) -> Dict[str, Any]:
        if installed:
            message = (
                f"Checkpoint {requested!r} is not installed. Choose one of the "
                f"{len(installed)} installed checkpoint(s), or install this model into "
                "ComfyUI/models/checkpoints."
            )
        else:
            message = (
                f"Checkpoint {requested!r} is not installed and ComfyUI has no checkpoints "
                "at all. Download a model into ComfyUI/models/checkpoints, then retry."
            )
        return {
            "kind": "checkpoint",
            "input_name": "ckpt_name",
            "requested": requested,
            "installed": list(installed),
            "suggestions": rank_similar(requested, installed),
            "fixable": bool(installed),
            "message": message,
        }

    @staticmethod
    def _lora_item(requested: str, installed: List[str]) -> Dict[str, Any]:
        suggestions = rank_similar(requested, installed)
        message = f"LoRA {requested!r} is not installed."
        if suggestions:
            message += f" Closest installed: {', '.join(suggestions[:3])}."
        return {
            "kind": "lora",
            "input_name": "lora_name",
            "requested": requested,
            "installed": list(installed),
            "suggestions": suggestions,
            "fixable": False,
            "message": message,
        }

    def _remediation_from_build(
        self,
        *,
        requested_checkpoint: str,
        checkpoint_resolved: bool,
        installed_checkpoints: List[str],
        unresolved_loras: List[str],
        installed_loras: List[str],
    ) -> List[Dict[str, Any]]:
        """Pre-emptive fixes, derived before ComfyUI has seen the graph."""
        items: List[Dict[str, Any]] = []
        if not checkpoint_resolved:
            items.append(self._checkpoint_item(requested_checkpoint, list(installed_checkpoints)))
        for name in unresolved_loras:
            items.append(self._lora_item(str(name), list(installed_loras)))
        return items

    def _remediation_from_failure(
        self,
        *,
        node_errors: Dict[str, Any],
        installed_checkpoints: List[str],
        installed_loras: List[str],
    ) -> List[Dict[str, Any]]:
        """Fixes derived from ComfyUI's own validation errors (authoritative)."""
        items: List[Dict[str, Any]] = []
        for node_id, node_payload in (node_errors or {}).items():
            if not isinstance(node_payload, dict):
                continue
            class_type = str(node_payload.get("class_type") or "")
            for entry in node_payload.get("errors") or []:
                if not isinstance(entry, dict):
                    continue
                extra = entry.get("extra_info") or {}
                input_name = str(extra.get("input_name") or "")
                received = str(extra.get("received_value") or "")
                if input_name == "ckpt_name":
                    items.append(self._checkpoint_item(received, list(installed_checkpoints)))
                elif input_name == "lora_name":
                    items.append(self._lora_item(received, list(installed_loras)))
                else:
                    items.append(
                        {
                            "kind": "input",
                            "node_id": str(node_id),
                            "class_type": class_type,
                            "input_name": input_name,
                            "requested": received,
                            "installed": [],
                            "suggestions": [],
                            "fixable": False,
                            "message": str(
                                entry.get("details")
                                or entry.get("message")
                                or "ComfyUI rejected this input."
                            ),
                        }
                    )
        return items

    # -- results ----------------------------------------------------------
    def prompt_result(self, prompt_id: str) -> Dict[str, Any]:
        """Return generated image descriptors for a finished prompt."""
        try:
            history = self.repository.history(prompt_id)
        except ComfyUIError as exc:
            return {"ready": False, "images": [], "error": str(exc)}

        entry = history.get(prompt_id)
        if not entry:
            return {"ready": False, "images": [], "error": ""}

        status = (entry.get("status") or {}).get("status_str", "")
        images: List[Dict[str, str]] = []
        for node_output in (entry.get("outputs") or {}).values():
            for image in node_output.get("images", []) or []:
                filename = image.get("filename", "")
                if not filename:
                    continue
                subfolder = image.get("subfolder", "") or ""
                type_ = image.get("type", "output") or "output"
                images.append(
                    {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": type_,
                        "url": self.repository.view_url(filename, subfolder, type_),
                    }
                )

        return {
            "ready": bool(images) or status in {"error", "success"},
            "status": status,
            "images": images,
            "error": "",
        }
