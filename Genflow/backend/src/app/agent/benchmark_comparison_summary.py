from __future__ import annotations

"""Build a compact comparison summary on top of a refinement benchmark set.

This layer is distinct from the raw benchmark set. It produces a small,
deterministic summary intended for later ranking/verifier-style consumers
without introducing ranking policy in the current phase.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from app.agent.refinement_benchmark_retriever import RefinementBenchmarkSet

if TYPE_CHECKING:
    from app.agent.memory import AgentSessionState


@dataclass
class BenchmarkComparisonItem:
    anchor_ids: List[int] = field(default_factory=list)
    candidate_id: str = ""
    candidate_reference_id: int | None = None
    focus_axes: List[str] = field(default_factory=list)
    preserve_axes: List[str] = field(default_factory=list)
    comparison_notes: List[str] = field(default_factory=list)
    coverage_hint: float = 0.0


@dataclass
class BenchmarkComparisonSummary:
    compared_anchor_ids: List[int] = field(default_factory=list)
    compared_candidate_ids: List[str] = field(default_factory=list)
    focus_axes: List[str] = field(default_factory=list)
    preserve_axes: List[str] = field(default_factory=list)
    summary_bullets: List[str] = field(default_factory=list)
    comparison_items: List[BenchmarkComparisonItem] = field(default_factory=list)
    confidence_hint: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_benchmark_comparison_summary(
    benchmark_set: RefinementBenchmarkSet,
    session: AgentSessionState,
) -> BenchmarkComparisonSummary:
    focus_axes = list(session.dissatisfaction_axes)
    preserve_axes = list(session.preserve_constraints)
    candidate_ids = [candidate.candidate_id for candidate in benchmark_set.comparison_candidates]
    coverage_hint = _compute_coverage_hint(
        focus_axes=focus_axes,
        preserve_axes=preserve_axes,
        candidate_count=len(benchmark_set.comparison_candidates),
    )
    items: list[BenchmarkComparisonItem] = []
    for candidate in benchmark_set.comparison_candidates:
        items.append(
            BenchmarkComparisonItem(
                anchor_ids=list(benchmark_set.anchor_ids),
                candidate_id=candidate.candidate_id,
                candidate_reference_id=candidate.reference_id,
                focus_axes=list(focus_axes),
                preserve_axes=list(preserve_axes),
                comparison_notes=[
                    candidate.selection_rationale,
                    f"anchor_overlap={candidate.metadata.get('anchor_overlap', False)}",
                ],
                coverage_hint=coverage_hint,
            )
        )
    summary_bullets = [
        f"benchmark_source={benchmark_set.benchmark_source}",
        f"anchor_count={len(benchmark_set.anchor_ids)}",
        f"candidate_count={len(benchmark_set.comparison_candidates)}",
    ]
    if focus_axes:
        summary_bullets.append(f"focus_axes={','.join(focus_axes)}")
    if preserve_axes:
        summary_bullets.append(f"preserve_axes={','.join(preserve_axes)}")

    return BenchmarkComparisonSummary(
        compared_anchor_ids=list(benchmark_set.anchor_ids),
        compared_candidate_ids=candidate_ids,
        focus_axes=focus_axes,
        preserve_axes=preserve_axes,
        summary_bullets=summary_bullets,
        comparison_items=items,
        confidence_hint=coverage_hint,
        metadata={
            "benchmark_kind": benchmark_set.benchmark_kind,
            "benchmark_source": benchmark_set.benchmark_source,
            "current_result_id": session.current_result_id,
            "selected_probe_id": session.selected_probe.probe_id,
        },
    )


def _compute_coverage_hint(
    focus_axes: List[str],
    preserve_axes: List[str],
    candidate_count: int,
) -> float:
    target_count = max(1, len(focus_axes) + len(preserve_axes))
    return round(min(1.0, candidate_count / target_count), 2)
