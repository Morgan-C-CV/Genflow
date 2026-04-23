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
]
