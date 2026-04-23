from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.agent.live_execution_models import (
    CommitExecutionRequest,
    ExecutionRequest,
    ExecutionResponse,
    PreviewExecutionRequest,
)


@runtime_checkable
class LiveBackendClient(Protocol):
    def run_initial(self, request: ExecutionRequest) -> ExecutionResponse:
        ...

    def run_preview(self, request: PreviewExecutionRequest) -> ExecutionResponse:
        ...

    def run_commit(self, request: CommitExecutionRequest) -> ExecutionResponse:
        ...
