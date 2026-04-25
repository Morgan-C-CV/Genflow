from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable

from app.agent.runtime_models import (
    CommittedPatch,
    NormalizedSchema,
    PreviewProbe,
    PreviewResult,
    ResultPayload,
    ResultSummary,
)
from app.agent.workflow_graph_patch_models import WorkflowGraphPatch


@runtime_checkable
class ExecutionAdapter(Protocol):
    def produce_initial_result(
        self,
        schema: NormalizedSchema,
        reference_bundle: Optional[Dict[str, object]] = None,
    ) -> tuple[ResultPayload, ResultSummary]:
        ...

    def execute_preview_probe(
        self,
        schema: NormalizedSchema,
        probe: PreviewProbe,
    ) -> PreviewResult:
        ...

    def execute_committed_patch(
        self,
        schema: NormalizedSchema,
        patch: CommittedPatch,
        graph_patch: WorkflowGraphPatch | None = None,
        commit_execution_mode: str = "",
        commit_execution_authority: str = "",
        commit_execution_implementation_mode: str = "",
    ) -> tuple[ResultPayload, ResultSummary]:
        ...
