"""HTTP access to a local ComfyUI instance.

Keeping this in ``repositories`` matches the project convention that external IO
lives behind a repository while ``services`` own orchestration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests


class ComfyUIError(RuntimeError):
    """Raised when ComfyUI is unreachable or rejects a request."""


class ComfyUIValidationError(ComfyUIError):
    """Raised when ComfyUI rejects a prompt payload with node errors."""

    def __init__(self, message: str, *, node_errors: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.node_errors = node_errors or {}


class ComfyUIRepository:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid4())

    # -- low level --------------------------------------------------------
    def _get(self, path: str, **params: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.get(url, params=params or None, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ComfyUIError(f"ComfyUI is unreachable at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise ComfyUIError(f"ComfyUI GET {path} failed with HTTP {response.status_code}: {response.text[:400]}")
        return response.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ComfyUIError(f"ComfyUI is unreachable at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text[:600]
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, dict):
                if body.get("node_errors"):
                    raise ComfyUIValidationError(
                        f"ComfyUI rejected the workflow: {body.get('error') or detail}",
                        node_errors=body.get("node_errors"),
                    )
                if body.get("error"):
                    raise ComfyUIError(f"ComfyUI rejected the request: {body['error']}")
            raise ComfyUIError(f"ComfyUI POST {path} failed with HTTP {response.status_code}: {detail}")
        return response.json()

    # -- capability discovery --------------------------------------------
    def object_info(self) -> Dict[str, Any]:
        return self._get("/object_info")

    def system_stats(self) -> Dict[str, Any]:
        return self._get("/system_stats")

    def _enum_options(self, object_info: Dict[str, Any], node: str, field: str) -> List[str]:
        try:
            spec = object_info[node]["input"]["required"][field]
        except (KeyError, TypeError):
            return []
        if isinstance(spec, list) and spec and isinstance(spec[0], list):
            return [str(item) for item in spec[0]]
        return []

    def available_checkpoints(self, object_info: Optional[Dict[str, Any]] = None) -> List[str]:
        info = object_info if object_info is not None else self.object_info()
        return self._enum_options(info, "CheckpointLoaderSimple", "ckpt_name")

    def available_loras(self, object_info: Optional[Dict[str, Any]] = None) -> List[str]:
        info = object_info if object_info is not None else self.object_info()
        return self._enum_options(info, "LoraLoader", "lora_name")

    def available_samplers(self, object_info: Optional[Dict[str, Any]] = None) -> List[str]:
        info = object_info if object_info is not None else self.object_info()
        return self._enum_options(info, "KSampler", "sampler_name")

    def capabilities(self) -> Dict[str, Any]:
        """Snapshot of what this ComfyUI install can actually run."""
        info = self.object_info()
        return {
            "base_url": self.base_url,
            "checkpoints": self.available_checkpoints(info),
            "loras": self.available_loras(info),
            "samplers": self.available_samplers(info),
            "object_info": info,
        }

    # -- execution --------------------------------------------------------
    def queue_prompt(self, graph: Dict[str, Any], client_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {"prompt": graph, "client_id": client_id or self.client_id}
        return self._post("/prompt", payload)

    def history(self, prompt_id: str) -> Dict[str, Any]:
        return self._get(f"/history/{prompt_id}")

    def queue_state(self) -> Dict[str, Any]:
        return self._get("/queue")

    def interrupt(self) -> None:
        self._post("/interrupt", {})

    def view_url(self, filename: str, subfolder: str = "", type_: str = "output") -> str:
        from urllib.parse import urlencode

        query = urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
        return f"{self.base_url}/view?{query}"
