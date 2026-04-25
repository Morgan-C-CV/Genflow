from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class NormalizedSchema:
    prompt: str = ""
    negative_prompt: str = ""
    cfgscale: str = ""
    steps: str = ""
    sampler: str = ""
    seed: str = ""
    model: str = ""
    clipskip: str = ""
    style: List[str] = field(default_factory=list)
    lora: List[str] = field(default_factory=list)
    full_metadata_string: str = ""
    raw_fields: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResultPayload:
    result_id: str = ""
    result_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultSummary:
    summary_text: str = ""
    changed_axes: List[str] = field(default_factory=list)
    preserved_axes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ParsedFeedbackEvidence:
    dissatisfaction_scope: List[str] = field(default_factory=list)
    preserve_constraints: List[str] = field(default_factory=list)
    requested_changes: List[str] = field(default_factory=list)
    uncertainty_estimate: float = 0.0
    raw_feedback: str = ""
    parser_notes: List[str] = field(default_factory=list)


@dataclass
class RepairHypothesis:
    hypothesis_id: str = ""
    summary: str = ""
    likely_changed_axes: List[str] = field(default_factory=list)
    likely_preserved_axes: List[str] = field(default_factory=list)
    likely_patch_family: str = ""
    rank: int = 0


@dataclass
class PreviewProbe:
    probe_id: str = ""
    summary: str = ""
    target_axes: List[str] = field(default_factory=list)
    preserve_axes: List[str] = field(default_factory=list)
    preview_execution_spec: Dict[str, Any] = field(default_factory=dict)
    source_kind: str = ""


@dataclass
class PreviewResult:
    probe_id: str = ""
    summary: ResultSummary = field(default_factory=ResultSummary)
    payload: ResultPayload = field(default_factory=ResultPayload)
    comparison_notes: List[str] = field(default_factory=list)


@dataclass
class CommittedPatch:
    patch_id: str = ""
    target_fields: List[str] = field(default_factory=list)
    target_axes: List[str] = field(default_factory=list)
    preserve_axes: List[str] = field(default_factory=list)
    changes: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierResult:
    improved: bool = False
    continue_recommended: bool = False
    confidence: float = 0.0
    regression_notes: List[str] = field(default_factory=list)
    summary: str = ""
    signal_summary: "VerifierSignalSummary" = field(default_factory=lambda: VerifierSignalSummary())


@dataclass
class VerifierSignalSummary:
    target_alignment_score: float = 0.0
    preserve_risk_score: float = 0.0
    benchmark_support_score: float = 0.0
    execution_evidence_score: float = 0.0
    total_score: float = 0.0
    notes: List[str] = field(default_factory=list)
    regression_notes: List[str] = field(default_factory=list)


@dataclass
class ExecutionSourceEvidenceSummary:
    commit_execution_mode: str = ""
    commit_execution_authority: str = ""
    request_primary_plan_kind: str = ""
    commit_execution_implementation_mode: str = ""
    request_graph_native_realization: bool = False
    request_backend_execution_mode: str = ""
    backend_graph_primary_capable: bool = False
    backend_graph_native_realization_supported: bool = False
    backend_graph_commit_payload_supplied: bool = False
    backend_graph_commit_payload_consumed: bool = False
    backend_graph_native_execution_realized: bool = False
    backend_accepted_execution_mode: str = ""
    backend_realized_execution_mode: str = ""
    execution_behavior_branch: str = ""
    preferred_commit_source: str = ""
    request_graph_native_artifact_input_received: bool = False
    selected_workflow_graph_patch_id: str = ""
    top_schema_patch_id: str = ""
    top_graph_patch_candidate_id: str = ""
    request_patch_id: str = ""
    response_patch_id: str = ""
    backend_graph_patch_id: str = ""
    backend_echoed_commit_source: str = ""
    backend_echoed_commit_execution_mode: str = ""
    backend_echoed_commit_execution_authority: str = ""
    backend_echoed_primary_plan_kind: str = ""
    backend_echoed_commit_execution_implementation_mode: str = ""
    backend_echoed_graph_primary_capable: bool = False
    backend_echoed_graph_native_realization_supported: bool = False
    backend_echoed_graph_commit_payload_supplied: bool = False
    backend_echoed_graph_commit_payload_consumed: bool = False
    backend_echoed_graph_native_execution_realized: bool = False
    backend_echoed_backend_execution_mode: str = ""
    backend_echoed_accepted_backend_execution_mode: str = ""
    backend_echoed_realized_backend_execution_mode: str = ""
    backend_echoed_execution_behavior_branch: str = ""
    backend_echoed_graph_native_artifact_input_received: bool = False
    comparison_notes: List[str] = field(default_factory=list)
