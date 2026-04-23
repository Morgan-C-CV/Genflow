from __future__ import annotations

from typing import Dict, Optional

from app.agent.execution_adapter import ExecutionAdapter
from app.agent.runtime_models import (
    CommittedPatch,
    NormalizedSchema,
    PreviewProbe,
    PreviewResult,
    ResultPayload,
    ResultSummary,
)


class LiveExecutionAdapter(ExecutionAdapter):
    def __init__(
        self,
        workflow_backend=None,
        preview_backend=None,
        artifact_store=None,
    ):
        self.workflow_backend = workflow_backend
        self.preview_backend = preview_backend
        self.artifact_store = artifact_store

    def produce_initial_result(
        self,
        schema: NormalizedSchema,
        reference_bundle: Optional[Dict[str, object]] = None,
    ) -> tuple[ResultPayload, ResultSummary]:
        raise NotImplementedError("Live execution adapter is not wired yet.")

    def execute_preview_probe(
        self,
        schema: NormalizedSchema,
        probe: PreviewProbe,
    ) -> PreviewResult:
        raise NotImplementedError("Live execution adapter is not wired yet.")

    def execute_committed_patch(
        self,
        schema: NormalizedSchema,
        patch: CommittedPatch,
    ) -> tuple[ResultPayload, ResultSummary]:
        raise NotImplementedError("Live execution adapter is not wired yet.")
