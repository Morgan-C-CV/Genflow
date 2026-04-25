from __future__ import annotations

"""Local PBO reranking for workflow graph patch candidates."""

from app.agent.benchmark_comparison_summary import BenchmarkComparisonSummary
from app.agent.workflow_graph_patch_models import WorkflowGraphPatchCandidate


def rank_workflow_graph_patch_candidates(
    candidates: list[WorkflowGraphPatchCandidate],
    session_context,
) -> list[WorkflowGraphPatchCandidate]:
    ranked_items = []
    for index, candidate in enumerate(candidates):
        score, rationale = _score_candidate(candidate, session_context)
        ranked_items.append(
            (
                -score,
                candidate.candidate_id,
                index,
                _clone_candidate_with_annotations(candidate, score, rationale),
            )
        )
    ranked_items.sort()
    return [item[-1] for item in ranked_items]


def _score_candidate(
    candidate: WorkflowGraphPatchCandidate,
    session_context,
) -> tuple[float, list[str]]:
    score = 0.0
    rationale: list[str] = []
    dissatisfaction_axes = set(getattr(session_context, "dissatisfaction_axes", []))
    preserve_constraints = set(getattr(session_context, "preserve_constraints", []))
    target_axes = set(candidate.target_axes)
    preserve_axes = set(candidate.preserve_axes)
    benchmark_summary: BenchmarkComparisonSummary | None = getattr(
        session_context, "benchmark_comparison_summary", None
    )
    benchmark_focus_axes = set(benchmark_summary.focus_axes if benchmark_summary else [])
    benchmark_preserve_axes = set(benchmark_summary.preserve_axes if benchmark_summary else [])

    dissatisfaction_hits = sorted(target_axes & dissatisfaction_axes)
    if dissatisfaction_hits:
        gain = 4.0 * len(dissatisfaction_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:dissatisfaction_match={','.join(dissatisfaction_hits)}")
    elif dissatisfaction_axes:
        penalty = 2.5
        score -= penalty
        rationale.append(f"-{penalty:.1f}:misses_dissatisfaction_axes")

    preserve_alignment_hits = sorted(preserve_axes & preserve_constraints)
    if preserve_alignment_hits:
        gain = 2.0 * len(preserve_alignment_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:preserve_alignment={','.join(preserve_alignment_hits)}")

    preserve_collision_hits = sorted(target_axes & preserve_constraints)
    if preserve_collision_hits:
        penalty = 5.5 * len(preserve_collision_hits)
        score -= penalty
        rationale.append(f"-{penalty:.1f}:preserve_collision={','.join(preserve_collision_hits)}")

    benchmark_focus_hits = sorted(target_axes & benchmark_focus_axes)
    if benchmark_focus_hits:
        gain = 3.5 * len(benchmark_focus_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:benchmark_focus={','.join(benchmark_focus_hits)}")
    elif benchmark_focus_axes:
        penalty = 1.5
        score -= penalty
        rationale.append(f"-{penalty:.1f}:misses_benchmark_focus")

    benchmark_preserve_hits = sorted(preserve_axes & benchmark_preserve_axes)
    if benchmark_preserve_hits:
        gain = 1.0 * len(benchmark_preserve_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:benchmark_preserve={','.join(benchmark_preserve_hits)}")

    region_bonus = 1.25 * len(candidate.region_patches)
    if region_bonus:
        score += region_bonus
        rationale.append(f"+{region_bonus:.1f}:region_patch_coverage")

    node_bonus = min(2.0, 0.5 * len(candidate.node_patches))
    if node_bonus:
        score += node_bonus
        rationale.append(f"+{node_bonus:.1f}:node_patch_coverage")

    edge_bonus = min(1.5, 0.25 * len(candidate.edge_patches))
    if edge_bonus:
        score += edge_bonus
        rationale.append(f"+{edge_bonus:.1f}:edge_patch_coverage")

    candidate_kind_bonus = _candidate_kind_bonus(candidate.candidate_kind, preserve_constraints)
    if candidate_kind_bonus:
        score += candidate_kind_bonus
        rationale.append(f"+{candidate_kind_bonus:.1f}:candidate_kind={candidate.candidate_kind}")

    return round(score, 2), rationale


def _candidate_kind_bonus(candidate_kind: str, preserve_constraints: set[str]) -> float:
    if candidate_kind == "conservative" and preserve_constraints:
        return 1.5
    if candidate_kind == "aggressive":
        return 0.5
    if candidate_kind == "balanced":
        return 1.0
    return 0.0


def _clone_candidate_with_annotations(
    candidate: WorkflowGraphPatchCandidate,
    score: float,
    rationale: list[str],
) -> WorkflowGraphPatchCandidate:
    metadata = dict(candidate.metadata)
    metadata["pbo_score"] = score
    metadata["pbo_rationale"] = list(rationale)
    return WorkflowGraphPatchCandidate(
        workflow_id=candidate.workflow_id,
        candidate_id=candidate.candidate_id,
        candidate_kind=candidate.candidate_kind,
        node_patches=list(candidate.node_patches),
        edge_patches=list(candidate.edge_patches),
        region_patches=list(candidate.region_patches),
        target_axes=list(candidate.target_axes),
        preserve_axes=list(candidate.preserve_axes),
        candidate_rationale=candidate.candidate_rationale,
        metadata=metadata,
    )
