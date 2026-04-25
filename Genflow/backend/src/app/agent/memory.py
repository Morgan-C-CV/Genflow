from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Dict, List, Optional
from uuid import uuid4

from app.agent.runtime_models import (
    CommittedPatch,
    ExecutionSourceEvidenceSummary,
    NormalizedSchema,
    ParsedFeedbackEvidence,
    PreviewProbe,
    PreviewResult,
    RepairHypothesis,
    ResultPayload,
    ResultSummary,
    VerifierResult,
    VerifierSignalSummary,
)
from app.agent.refinement_benchmark_retriever import RefinementBenchmarkSet
from app.agent.benchmark_comparison_summary import BenchmarkComparisonSummary
from app.agent.verifier_repair_recommendation import VerifierRepairRecommendation
from app.agent.workflow_graph_patch_models import WorkflowGraphPatch, WorkflowGraphPatchCandidate
from app.agent.workflow_runtime_models import (
    WorkflowExecutionConfig,
    WorkflowGraphPlaceholder,
    WorkflowIdentity,
    WorkflowScope,
    WorkflowStateSnapshot,
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
    refinement_benchmark_set: RefinementBenchmarkSet = field(default_factory=RefinementBenchmarkSet)
    refinement_benchmark_summary: str = ""
    benchmark_comparison_summary: BenchmarkComparisonSummary = field(default_factory=BenchmarkComparisonSummary)
    workflow_id: str = ""
    workflow_identity: WorkflowIdentity = field(default_factory=WorkflowIdentity)
    workflow_state: WorkflowStateSnapshot = field(default_factory=WorkflowStateSnapshot)
    editable_scopes: List[WorkflowScope] = field(default_factory=list)
    protected_scopes: List[WorkflowScope] = field(default_factory=list)
    last_execution_config: WorkflowExecutionConfig = field(default_factory=WorkflowExecutionConfig)
    workflow_metadata: Dict[str, object] = field(default_factory=dict)
    workflow_graph_placeholder: WorkflowGraphPlaceholder = field(default_factory=WorkflowGraphPlaceholder)
    workflow_topology_hints: Dict[str, object] = field(default_factory=dict)
    workflow_topology_entry_node_ids: List[str] = field(default_factory=list)
    workflow_topology_exit_node_ids: List[str] = field(default_factory=list)
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
    top_schema_patch_candidate: CommittedPatch = field(default_factory=CommittedPatch)
    current_workflow_graph_patch: WorkflowGraphPatch = field(default_factory=WorkflowGraphPatch)
    selected_workflow_graph_patch: WorkflowGraphPatch = field(default_factory=WorkflowGraphPatch)
    workflow_graph_patch_candidates: List[WorkflowGraphPatchCandidate] = field(default_factory=list)
    top_workflow_graph_patch_candidate: WorkflowGraphPatchCandidate = field(default_factory=WorkflowGraphPatchCandidate)
    preferred_commit_source: str = "schema"
    commit_execution_mode: str = "schema_execution_fallback"
    selected_graph_native_patch_candidate: WorkflowGraphPatchCandidate = field(
        default_factory=WorkflowGraphPatchCandidate
    )
    latest_execution_source_evidence: ExecutionSourceEvidenceSummary = field(
        default_factory=ExecutionSourceEvidenceSummary
    )
    current_uncertainty_estimate: float = 0.0
    latest_verifier_result: VerifierResult = field(default_factory=VerifierResult)
    latest_verifier_signal_summary: VerifierSignalSummary = field(default_factory=VerifierSignalSummary)
    latest_verifier_repair_recommendation: VerifierRepairRecommendation = field(
        default_factory=VerifierRepairRecommendation
    )
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
