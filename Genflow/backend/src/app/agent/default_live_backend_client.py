from __future__ import annotations

from typing import Any

from app.agent.live_backend_client import LiveBackendClient
from app.agent.live_backend_errors import (
    LiveBackendDispatchError,
    LiveBackendResponseError,
    LiveBackendUnavailableError,
    LiveExecutionError,
)
from app.agent.live_execution_models import (
    CommitExecutionRequest,
    ExecutionRequest,
    ExecutionResponse,
    PreviewExecutionRequest,
    normalize_execution_response,
)


class DefaultLiveBackendClient(LiveBackendClient):
    def __init__(self, transport=None):
        self.transport = transport

    def run_initial(self, request: ExecutionRequest) -> ExecutionResponse:
        raw_response = self._dispatch("initial", request)
        return normalize_execution_response(raw_response, expected_kind="initial")

    def run_preview(self, request: PreviewExecutionRequest) -> ExecutionResponse:
        raw_response = self._dispatch("preview", request)
        return normalize_execution_response(raw_response, expected_kind="preview")

    def run_commit(self, request: CommitExecutionRequest) -> ExecutionResponse:
        raw_response = self._dispatch("commit", request)
        return normalize_execution_response(raw_response, expected_kind="commit")

    def _dispatch(self, execution_kind: str, request: ExecutionRequest) -> Any:
        if self.transport is None:
            raise LiveBackendUnavailableError("Live backend transport is not configured.")

        try:
            if callable(self.transport):
                return self.transport(execution_kind, request)
            if hasattr(self.transport, "dispatch"):
                return self.transport.dispatch(execution_kind, request)
            raise LiveBackendUnavailableError("Live backend transport does not expose a callable dispatch interface.")
        except LiveExecutionError:
            raise
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise LiveBackendUnavailableError(
                f"Live backend transport is unavailable for {execution_kind} execution."
            ) from exc
        except Exception as exc:
            raise LiveBackendDispatchError(
                f"Live backend dispatch failed for {execution_kind} execution."
            ) from exc
