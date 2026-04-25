from __future__ import annotations

"""Deterministic arbitration between schema and graph-native commit winners."""

from dataclasses import dataclass, field

from app.agent.runtime_models import CommittedPatch
from app.agent.workflow_graph_patch_models import WorkflowGraphPatchCandidate


@dataclass
class WorkflowPatchCommitSelection:
    preferred_commit_source: str = "schema"
    rationale: list[str] = field(default_factory=list)
    selected_graph_native_patch_candidate: WorkflowGraphPatchCandidate = field(
        default_factory=WorkflowGraphPatchCandidate
    )


def select_commit_patch_winner(
    schema_patch_winner: CommittedPatch,
    graph_patch_winner: WorkflowGraphPatchCandidate,
    session,
) -> WorkflowPatchCommitSelection:
    if not schema_patch_winner.patch_id:
        return WorkflowPatchCommitSelection(
            preferred_commit_source="graph" if graph_patch_winner.candidate_id else "schema",
            rationale=["no_schema_patch_winner"],
            selected_graph_native_patch_candidate=graph_patch_winner,
        )

    if not graph_patch_winner.candidate_id:
        return WorkflowPatchCommitSelection(
            preferred_commit_source="schema",
            rationale=["no_graph_patch_winner"],
            selected_graph_native_patch_candidate=graph_patch_winner,
        )

    schema_axes = set(schema_patch_winner.target_axes)
    graph_axes = set(graph_patch_winner.target_axes)
    schema_preserve = set(schema_patch_winner.preserve_axes)
    graph_preserve = set(graph_patch_winner.preserve_axes)
    preserve_constraints = set(getattr(session, "preserve_constraints", []))
    aligned_flag = bool(schema_patch_winner.metadata.get("graph_native_aligned_winner", False))
    aligned_candidate_id = str(schema_patch_winner.metadata.get("aligned_graph_candidate_id", ""))
    benchmark_support = getattr(session.latest_verifier_signal_summary, "benchmark_support_score", 0.0)

    rationale: list[str] = []
    graph_score = float(graph_patch_winner.metadata.get("pbo_score", 0.0))
    schema_score = float(schema_patch_winner.metadata.get("pbo_score", 0.0))

    axes_overlap = sorted(schema_axes & graph_axes)
    if axes_overlap:
        rationale.append(f"target_axes_overlap={','.join(axes_overlap)}")

    preserve_compatible = not bool((schema_axes | graph_axes) & preserve_constraints - (schema_preserve | graph_preserve))
    if preserve_compatible:
        rationale.append("preserve_compatible")
    else:
        rationale.append("preserve_conflict_detected")

    if aligned_flag and aligned_candidate_id == graph_patch_winner.candidate_id:
        rationale.append("graph_native_aligned_winner")

    safe_graph_preferred = (
        aligned_flag
        and aligned_candidate_id == graph_patch_winner.candidate_id
        and preserve_compatible
        and bool(axes_overlap)
    )
    if safe_graph_preferred:
        rationale.append("safe_aligned_graph_preference")
        if benchmark_support >= 1.0:
            rationale.append("benchmark_supported_alignment")
        return WorkflowPatchCommitSelection(
            preferred_commit_source="graph",
            rationale=rationale,
            selected_graph_native_patch_candidate=graph_patch_winner,
        )

    if graph_score > schema_score and preserve_compatible and bool(axes_overlap):
        rationale.append("graph_score_exceeds_schema")
        return WorkflowPatchCommitSelection(
            preferred_commit_source="graph",
            rationale=rationale,
            selected_graph_native_patch_candidate=graph_patch_winner,
        )

    rationale.append("schema_path_retained_for_execution_compatibility")
    return WorkflowPatchCommitSelection(
        preferred_commit_source="schema",
        rationale=rationale,
        selected_graph_native_patch_candidate=graph_patch_winner,
    )
