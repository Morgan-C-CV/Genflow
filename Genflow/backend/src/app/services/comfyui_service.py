"""Turn a Genflow schema into a ComfyUI workflow and push it to the queue."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.modules.comfyui_graph import build_api_graph, build_ui_workflow
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
            "checkpoint_resolved": result.checkpoint_resolved,
            "applied_loras": result.applied_loras,
            "unresolved_loras": result.unresolved_loras,
            "controls": result.controls,
            "available_checkpoints": checkpoints,
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
            return payload
        except ComfyUIError as exc:
            payload["error"] = str(exc)
            return payload

        payload["pushed"] = True
        payload["prompt_id"] = str(response.get("prompt_id", ""))
        payload["queue_number"] = response.get("number")
        payload["node_errors"] = response.get("node_errors") or {}
        return payload

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
