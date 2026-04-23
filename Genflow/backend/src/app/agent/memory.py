from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Dict, List, Optional
from uuid import uuid4

from app.agent.runtime_models import (
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

if TYPE_CHECKING:
    from app.agents.creative_agent import (
        CandidateWall,
        CreativeIntentPlan,
        ExpandedQuery,
        ResourceRecommendation,
    )


@dataclass
class AgentSessionState:
    session_id: str
    original_intent: str
    clarified_intent: str
    clarification_closed: bool = False
    clarification_rounds: int = 0
    plan: Optional[CreativeIntentPlan] = None
    resource_recommendation: Optional[ResourceRecommendation] = None
    previous_expansions: List[ExpandedQuery] = field(default_factory=list)
    previous_wall_indices: List[int] = field(default_factory=list)
    latest_expansions: List[ExpandedQuery] = field(default_factory=list)
    latest_wall: Optional[CandidateWall] = None
    current_schema: NormalizedSchema = field(default_factory=NormalizedSchema)
    current_schema_raw: str = ""
    current_result_id: str = ""
    current_result_payload: ResultPayload = field(default_factory=ResultPayload)
    current_result_summary: ResultSummary = field(default_factory=ResultSummary)
    previous_result_summary: ResultSummary = field(default_factory=ResultSummary)
    result_comparison_notes: List[str] = field(default_factory=list)
    accepted_results: List[ResultPayload] = field(default_factory=list)
    preview_results: List[PreviewResult] = field(default_factory=list)
    selected_gallery_index: int | None = None
    selected_reference_bundle: Dict[str, object] = field(default_factory=dict)
    selected_reference_ids: List[int] = field(default_factory=list)
    current_gallery_anchor_summary: str = ""
    feedback_history: List[str] = field(default_factory=list)
    latest_feedback: str = ""
    parsed_feedback: ParsedFeedbackEvidence = field(default_factory=ParsedFeedbackEvidence)
    preserve_constraints: List[str] = field(default_factory=list)
    dissatisfaction_axes: List[str] = field(default_factory=list)
    requested_changes: List[str] = field(default_factory=list)
    repair_hypotheses: List[RepairHypothesis] = field(default_factory=list)
    local_probes: List[PreviewProbe] = field(default_factory=list)
    preview_probe_candidates: List[PreviewProbe] = field(default_factory=list)
    preview_probe_results: List[PreviewResult] = field(default_factory=list)
    selected_probe: PreviewProbe = field(default_factory=PreviewProbe)
    patch_history: List[CommittedPatch] = field(default_factory=list)
    accepted_patch: CommittedPatch = field(default_factory=CommittedPatch)
    current_uncertainty_estimate: float = 0.0
    latest_verifier_result: VerifierResult = field(default_factory=VerifierResult)
    continue_recommended: bool = False
    stop_reason: str = ""
    verifier_confidence: float = 0.0


class AgentMemoryService:
    def __init__(self):
        self._lock = RLock()
        self._sessions: Dict[str, AgentSessionState] = {}

    def create_session(self, user_intent: str) -> AgentSessionState:
        session = AgentSessionState(
            session_id=str(uuid4()),
            original_intent=user_intent.strip(),
            clarified_intent=user_intent.strip(),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> AgentSessionState:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session

    def save_session(self, session: AgentSessionState) -> AgentSessionState:
        with self._lock:
            self._sessions[session.session_id] = session
        return session
