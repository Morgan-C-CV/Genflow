from __future__ import annotations

"""Local PBO reranking for committed patch candidates."""

from typing import Iterable, List

from app.agent.benchmark_comparison_summary import BenchmarkComparisonSummary
from app.agent.refinement_benchmark_retriever import RefinementBenchmarkSet
from app.agent.runtime_models import CommittedPatch, ParsedFeedbackEvidence


def rank_patch_candidates(
    patch_candidates: List[CommittedPatch],
    parsed_feedback: ParsedFeedbackEvidence,
    benchmark_comparison_summary: BenchmarkComparisonSummary | None = None,
    refinement_benchmark_set: RefinementBenchmarkSet | None = None,
) -> List[CommittedPatch]:
    ranked_items = []
    for index, patch in enumerate(patch_candidates):
        score, rationale = _score_patch(
            patch=patch,
            parsed_feedback=parsed_feedback,
            benchmark_comparison_summary=benchmark_comparison_summary,
            refinement_benchmark_set=refinement_benchmark_set,
        )
        ranked_items.append(
            (
                -score,
                patch.patch_id,
                index,
                _clone_patch_with_pbo_annotations(patch, score, rationale),
            )
        )
    ranked_items.sort()
    return [item[-1] for item in ranked_items]


def _score_patch(
    patch: CommittedPatch,
    parsed_feedback: ParsedFeedbackEvidence,
    benchmark_comparison_summary: BenchmarkComparisonSummary | None,
    refinement_benchmark_set: RefinementBenchmarkSet | None,
) -> tuple[float, list[str]]:
    score = 0.0
    rationale: list[str] = []
    target_axes = set(patch.target_axes)
    preserve_axes = set(patch.preserve_axes)
    dissatisfaction_axes = set(parsed_feedback.dissatisfaction_scope)
    benchmark_focus_axes = set(benchmark_comparison_summary.focus_axes if benchmark_comparison_summary else [])
    benchmark_preserve_axes = set(benchmark_comparison_summary.preserve_axes if benchmark_comparison_summary else [])
    protected_axes = _collect_protected_axes(parsed_feedback, benchmark_comparison_summary, patch)

    dissatisfaction_hits = sorted(target_axes & dissatisfaction_axes)
    if dissatisfaction_hits:
        gain = 4.5 * len(dissatisfaction_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:dissatisfaction_match={','.join(dissatisfaction_hits)}")

    benchmark_focus_hits = sorted(target_axes & benchmark_focus_axes)
    if benchmark_focus_hits:
        gain = 6.0 * len(benchmark_focus_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:benchmark_focus={','.join(benchmark_focus_hits)}")

    preserve_alignment_hits = sorted(preserve_axes & protected_axes)
    if preserve_alignment_hits:
        gain = 1.5 * len(preserve_alignment_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:preserve_alignment={','.join(preserve_alignment_hits)}")

    preserve_collision_hits = sorted(target_axes & protected_axes)
    if preserve_collision_hits:
        penalty = 5.5 * len(preserve_collision_hits)
        score -= penalty
        rationale.append(f"-{penalty:.1f}:preserve_collision={','.join(preserve_collision_hits)}")

    if benchmark_comparison_summary is not None:
        coverage_gain = min(2.0, benchmark_comparison_summary.confidence_hint * 2.0)
        if coverage_gain:
            score += coverage_gain
            rationale.append(f"+{coverage_gain:.1f}:benchmark_confidence")

    source_bonus = _source_bonus(refinement_benchmark_set)
    if source_bonus:
        score += source_bonus
        rationale.append(f"+{source_bonus:.1f}:benchmark_source")

    return round(score, 2), rationale


def _collect_protected_axes(
    parsed_feedback: ParsedFeedbackEvidence,
    benchmark_comparison_summary: BenchmarkComparisonSummary | None,
    patch: CommittedPatch,
) -> set[str]:
    preserve_axes = set(benchmark_comparison_summary.preserve_axes if benchmark_comparison_summary else [])
    lower_constraints = " ".join(parsed_feedback.preserve_constraints).lower()
    candidate_axes = set(patch.target_axes) | set(patch.preserve_axes) | preserve_axes
    for axis in candidate_axes:
        if axis.lower() in lower_constraints:
            preserve_axes.add(axis)
    return preserve_axes


def _source_bonus(refinement_benchmark_set: RefinementBenchmarkSet | None) -> float:
    if refinement_benchmark_set is None or not refinement_benchmark_set.benchmark_source:
        return 0.0
    if refinement_benchmark_set.benchmark_source == "refinement_search_bundle":
        return 0.75
    return 0.25


def _clone_patch_with_pbo_annotations(
    patch: CommittedPatch,
    score: float,
    rationale: Iterable[str],
) -> CommittedPatch:
    metadata = dict(patch.metadata)
    metadata["pbo_score"] = score
    metadata["pbo_rationale"] = list(rationale)
    return CommittedPatch(
        patch_id=patch.patch_id,
        target_fields=list(patch.target_fields),
        target_axes=list(patch.target_axes),
        preserve_axes=list(patch.preserve_axes),
        changes=dict(patch.changes),
        rationale=patch.rationale,
        metadata=metadata,
    )
