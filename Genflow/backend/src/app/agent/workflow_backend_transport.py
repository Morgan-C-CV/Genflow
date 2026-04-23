from __future__ import annotations

from app.agent.live_backend_config import LiveBackendConfig
from app.agent.live_backend_errors import (
    LiveBackendNotConfiguredError,
    LiveBackendNotImplementedError,
    LiveBackendUnavailableError,
)


SUPPORTED_BACKEND_KINDS = {"workflow_shell"}


class WorkflowBackendTransport:
    def __init__(self, config: LiveBackendConfig):
        self.config = config

    def dispatch(self, execution_kind: str, request):
        if not self.config.enabled:
            raise LiveBackendNotConfiguredError("Live backend substrate is not configured.")
        if not self.config.backend_kind:
            raise LiveBackendNotConfiguredError("Live backend kind is missing from configuration.")
        if self.config.backend_kind not in SUPPORTED_BACKEND_KINDS:
            raise LiveBackendUnavailableError(
                f"Unsupported live backend kind: {self.config.backend_kind}."
            )
        raise LiveBackendNotImplementedError(
            f"Live backend substrate '{self.config.backend_kind}' is configured but dispatch is not implemented yet."
        )
