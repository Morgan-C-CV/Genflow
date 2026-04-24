from __future__ import annotations

"""Local PBO reranking for preview probe candidates."""

from typing import Iterable, List

from app.agent.benchmark_comparison_summary import BenchmarkComparisonSummary
from app.agent.refinement_benchmark_retriever import RefinementBenchmarkSet
from app.agent.runtime_models import ParsedFeedbackEvidence, PreviewProbe


def rank_probe_candidates(
    probes: List[PreviewProbe],
    parsed_feedback: ParsedFeedbackEvidence,
    benchmark_comparison_summary: BenchmarkComparisonSummary | None = None,
    refinement_benchmark_set: RefinementBenchmarkSet | None = None,
) -> List[PreviewProbe]:
    ranked_items = []
    for index, probe in enumerate(probes):
        score, rationale = _score_probe(
            probe=probe,
            parsed_feedback=parsed_feedback,
            benchmark_comparison_summary=benchmark_comparison_summary,
            refinement_benchmark_set=refinement_benchmark_set,
        )
        ranked_items.append(
            (
                -score,
                probe.probe_id,
                index,
                _clone_probe_with_pbo_annotations(probe, score=score, rationale=rationale),
            )
        )
    ranked_items.sort()
    return [item[-1] for item in ranked_items]


def _score_probe(
    probe: PreviewProbe,
    parsed_feedback: ParsedFeedbackEvidence,
    benchmark_comparison_summary: BenchmarkComparisonSummary | None,
    refinement_benchmark_set: RefinementBenchmarkSet | None,
) -> tuple[float, list[str]]:
    score = 0.0
    rationale: list[str] = []
    target_axes = set(probe.target_axes)
    preserve_axes = set(probe.preserve_axes)
    dissatisfaction_axes = set(parsed_feedback.dissatisfaction_scope)
    benchmark_focus_axes = set((benchmark_comparison_summary.focus_axes if benchmark_comparison_summary else []))
    benchmark_preserve_axes = set((benchmark_comparison_summary.preserve_axes if benchmark_comparison_summary else []))

    dissatisfaction_hits = sorted(target_axes & dissatisfaction_axes)
    if dissatisfaction_hits:
        gain = 4.0 * len(dissatisfaction_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:dissatisfaction_match={','.join(dissatisfaction_hits)}")

    protected_axes = _collect_preserve_axes(parsed_feedback, benchmark_comparison_summary, probe)

    preserve_alignment_hits = sorted(preserve_axes & protected_axes)
    if preserve_alignment_hits:
        gain = 1.5 * len(preserve_alignment_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:preserve_alignment={','.join(preserve_alignment_hits)}")

    preserve_collision_hits = sorted(target_axes & protected_axes)
    if preserve_collision_hits:
        penalty = 5.0 * len(preserve_collision_hits)
        score -= penalty
        rationale.append(f"-{penalty:.1f}:preserve_collision={','.join(preserve_collision_hits)}")

    benchmark_focus_hits = sorted(target_axes & benchmark_focus_axes)
    if benchmark_focus_hits:
        gain = 4.5 * len(benchmark_focus_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:benchmark_focus={','.join(benchmark_focus_hits)}")

    benchmark_preserve_hits = sorted(preserve_axes & benchmark_preserve_axes)
    if benchmark_preserve_hits:
        gain = 1.0 * len(benchmark_preserve_hits)
        score += gain
        rationale.append(f"+{gain:.1f}:benchmark_preserve={','.join(benchmark_preserve_hits)}")

    coverage_bonus = _coverage_bonus(refinement_benchmark_set, benchmark_comparison_summary)
    if coverage_bonus:
        score += coverage_bonus
        rationale.append(f"+{coverage_bonus:.1f}:benchmark_coverage")

    source_bonus = _source_bonus(refinement_benchmark_set)
    if source_bonus:
        score += source_bonus
        rationale.append(f"+{source_bonus:.1f}:benchmark_source")

    return round(score, 2), rationale


def _collect_preserve_axes(
    parsed_feedback: ParsedFeedbackEvidence,
    benchmark_comparison_summary: BenchmarkComparisonSummary | None,
    probe: PreviewProbe,
) -> set[str]:
    preserve_axes = set(benchmark_comparison_summary.preserve_axes if benchmark_comparison_summary else [])
    lower_constraints = " ".join(parsed_feedback.preserve_constraints).lower()
    candidate_axes = _axis_candidates(benchmark_comparison_summary) | set(probe.target_axes) | set(probe.preserve_axes)
    for axis in candidate_axes:
        if axis.lower() in lower_constraints:
            preserve_axes.add(axis)
    return preserve_axes


def _axis_candidates(benchmark_comparison_summary: BenchmarkComparisonSummary | None) -> set[str]:
    if benchmark_comparison_summary is None:
        return set()
    return set(benchmark_comparison_summary.focus_axes) | set(benchmark_comparison_summary.preserve_axes)


def _coverage_bonus(
    refinement_benchmark_set: RefinementBenchmarkSet | None,
    benchmark_comparison_summary: BenchmarkComparisonSummary | None,
) -> float:
    if benchmark_comparison_summary is None:
        return 0.0
    candidate_count = len(benchmark_comparison_summary.compared_candidate_ids)
    if not candidate_count and refinement_benchmark_set is not None:
        candidate_count = len(refinement_benchmark_set.comparison_candidates)
    return min(1.5, 0.5 * candidate_count)


def _source_bonus(refinement_benchmark_set: RefinementBenchmarkSet | None) -> float:
    if refinement_benchmark_set is None or not refinement_benchmark_set.benchmark_source:
        return 0.0
    if refinement_benchmark_set.benchmark_source == "refinement_search_bundle":
        return 0.75
    return 0.25


def _clone_probe_with_pbo_annotations(
    probe: PreviewProbe,
    score: float,
    rationale: Iterable[str],
) -> PreviewProbe:
    preview_execution_spec = dict(probe.preview_execution_spec)
    preview_execution_spec["pbo_score"] = score
    preview_execution_spec["pbo_rationale"] = list(rationale)
    return PreviewProbe(
        probe_id=probe.probe_id,
        summary=probe.summary,
        target_axes=list(probe.target_axes),
        preserve_axes=list(probe.preserve_axes),
        preview_execution_spec=preview_execution_spec,
        source_kind=probe.source_kind,
    )
