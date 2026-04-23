from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class LiveBackendConfig:
    backend_kind: str = ""
    endpoint: str = ""
    timeout_seconds: float = 30.0
    enabled: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


def resolve_live_backend_config(env: dict | None = None) -> LiveBackendConfig:
    env = env or os.environ
    enabled_raw = str(env.get("GENFLOW_LIVE_BACKEND_ENABLED", "")).strip().lower()
    backend_kind = str(env.get("GENFLOW_LIVE_BACKEND_KIND", "")).strip().lower()
    endpoint = str(env.get("GENFLOW_LIVE_BACKEND_ENDPOINT", "")).strip()
    timeout_raw = str(env.get("GENFLOW_LIVE_BACKEND_TIMEOUT_SECONDS", "30")).strip()
    enabled = enabled_raw in {"1", "true", "yes", "on"} or bool(backend_kind)

    try:
        timeout_seconds = float(timeout_raw) if timeout_raw else 30.0
    except ValueError:
        timeout_seconds = 30.0

    metadata = {
        "namespace": str(env.get("GENFLOW_LIVE_BACKEND_NAMESPACE", "")).strip(),
        "workflow_profile": str(env.get("GENFLOW_LIVE_BACKEND_PROFILE", "")).strip(),
    }
    metadata = {key: value for key, value in metadata.items() if value}

    return LiveBackendConfig(
        backend_kind=backend_kind,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        enabled=enabled,
        metadata=metadata,
    )
