"""Agent package.

Expose low-side-effect compatibility exports at package level while avoiding
heavy imports during basic runtime-model and memory tests.
"""

from .memory import AgentMemoryService, AgentSessionState
from .runtime_models import (
    CommittedPatch,
    NormalizedSchema,
    ParsedFeedbackEvidence,
    PreviewProbe,
    PreviewResult,
    RepairHypothesis,
    ResultPayload,
    ResultSummary,
    VerifierResult,
)
from .workflow_runtime_models import (
    WorkflowExecutionConfig,
    WorkflowGraphPlaceholder,
    WorkflowIdentity,
    WorkflowNodeRef,
    WorkflowScope,
    WorkflowStateSnapshot,
    WorkflowTopologySlice,
)
from .workflow_descriptor_models import (
    SurrogateExecutionDescriptor,
    SurrogateRepairDescriptor,
    SurrogateWorkflowDescriptor,
)
from .workflow_document_models import (
    SurrogateWorkflowDocument,
    SurrogateWorkflowEdge,
    SurrogateWorkflowNode,
    SurrogateWorkflowRegion,
)
from .workflow_snapshot_builder import SurrogateWorkflowSnapshot

__all__ = [
    "AgentMemoryService",
    "AgentSessionState",
    "CommittedPatch",
    "NormalizedSchema",
    "ParsedFeedbackEvidence",
    "PreviewProbe",
    "PreviewResult",
    "RepairHypothesis",
    "ResultPayload",
    "ResultSummary",
    "VerifierResult",
    "WorkflowExecutionConfig",
    "WorkflowGraphPlaceholder",
    "WorkflowIdentity",
    "WorkflowNodeRef",
    "WorkflowScope",
    "WorkflowStateSnapshot",
    "WorkflowTopologySlice",
    "SurrogateExecutionDescriptor",
    "SurrogateRepairDescriptor",
    "SurrogateWorkflowDescriptor",
    "SurrogateWorkflowDocument",
    "SurrogateWorkflowEdge",
    "SurrogateWorkflowNode",
    "SurrogateWorkflowRegion",
    "SurrogateWorkflowSnapshot",
]
