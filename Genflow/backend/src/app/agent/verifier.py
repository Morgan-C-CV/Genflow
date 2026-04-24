from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.agent.benchmark_comparison_summary import BenchmarkComparisonSummary
from app.agent.runtime_models import CommittedPatch, PreviewProbe, ResultSummary, VerifierResult


@dataclass
class VerifierSignalBreakdown:
    target_alignment_score: float = 0.0
    preserve_risk_score: float = 0.0
    benchmark_support_score: float = 0.0
    execution_evidence_score: float = 0.0
    notes: List[str] = field(default_factory=list)
    regression_notes: List[str] = field(default_factory=list)


class Verifier:
    def verify(
        self,
        previous_result_summary: ResultSummary,
        updated_result_summary: ResultSummary,
        selected_probe: PreviewProbe,
        committed_patch: CommittedPatch,
        preserve_constraints: List[str],
        benchmark_comparison_summary: BenchmarkComparisonSummary | None = None,
    ) -> VerifierResult:
        breakdown = VerifierSignalBreakdown()
        breakdown.notes.extend(
            self._score_target_alignment(
                updated_result_summary=updated_result_summary,
                selected_probe=selected_probe,
                committed_patch=committed_patch,
                breakdown=breakdown,
            )
        )
        breakdown.regression_notes.extend(
            self._score_preserve_risk(
                selected_probe=selected_probe,
                committed_patch=committed_patch,
                preserve_constraints=preserve_constraints,
                breakdown=breakdown,
            )
        )
        breakdown.notes.extend(
            self._score_benchmark_support(
                benchmark_comparison_summary=benchmark_comparison_summary,
                selected_probe=selected_probe,
                committed_patch=committed_patch,
                breakdown=breakdown,
            )
        )
        breakdown.notes.extend(
            self._score_execution_evidence(
                previous_result_summary=previous_result_summary,
                updated_result_summary=updated_result_summary,
                committed_patch=committed_patch,
                breakdown=breakdown,
            )
        )

        total_score = (
            breakdown.target_alignment_score
            + breakdown.benchmark_support_score
            + breakdown.execution_evidence_score
            - breakdown.preserve_risk_score
        )
        improved = (
            total_score >= 1.5
            and breakdown.target_alignment_score > 0.0
            and breakdown.execution_evidence_score > 0.0
            and breakdown.preserve_risk_score < 2.0
        )
        continue_recommended = not improved
        confidence = self._compute_confidence(total_score=total_score, improved=improved)
        regression_notes = list(dict.fromkeys(breakdown.regression_notes + breakdown.notes))
        summary = self._build_summary(
            committed_patch=committed_patch,
            updated_result_summary=updated_result_summary,
            improved=improved,
            breakdown=breakdown,
            total_score=total_score,
        )

        return VerifierResult(
            improved=improved,
            continue_recommended=continue_recommended,
            confidence=confidence,
            regression_notes=regression_notes,
            summary=summary,
        )

    def _score_target_alignment(
        self,
        updated_result_summary: ResultSummary,
        selected_probe: PreviewProbe,
        committed_patch: CommittedPatch,
        breakdown: VerifierSignalBreakdown,
    ) -> list[str]:
        notes: list[str] = []
        target_axes = set(selected_probe.target_axes) | set(committed_patch.target_axes)
        changed_axes = set(updated_result_summary.changed_axes)
        alignment_hits = sorted(target_axes & changed_axes)
        if alignment_hits:
            gain = 1.6 * len(alignment_hits)
            breakdown.target_alignment_score += gain
            notes.append(f"target_alignment={','.join(alignment_hits)}")
        else:
            notes.append("target_alignment=none")

        preserved_hits = sorted(set(selected_probe.preserve_axes) & set(updated_result_summary.preserved_axes))
        if preserved_hits:
            gain = 0.4 * len(preserved_hits)
            breakdown.target_alignment_score += gain
            notes.append(f"preserve_alignment={','.join(preserved_hits)}")
        return notes

    def _score_preserve_risk(
        self,
        selected_probe: PreviewProbe,
        committed_patch: CommittedPatch,
        preserve_constraints: List[str],
        breakdown: VerifierSignalBreakdown,
    ) -> list[str]:
        regression_notes: list[str] = []
        protected_axes = self._collect_preserve_axes(selected_probe, preserve_constraints)

        overlapping_axes = sorted(set(selected_probe.preserve_axes) & set(selected_probe.target_axes))
        if overlapping_axes:
            penalty = 1.5 * len(overlapping_axes)
            breakdown.preserve_risk_score += penalty
            regression_notes.append(f"preserve overlap risk={','.join(overlapping_axes)}")

        patch_collisions = sorted(set(committed_patch.target_axes) & protected_axes)
        if patch_collisions:
            penalty = 1.75 * len(patch_collisions)
            breakdown.preserve_risk_score += penalty
            regression_notes.append(f"patch collides with preserve axes={','.join(patch_collisions)}")
        return regression_notes

    def _score_benchmark_support(
        self,
        benchmark_comparison_summary: BenchmarkComparisonSummary | None,
        selected_probe: PreviewProbe,
        committed_patch: CommittedPatch,
        breakdown: VerifierSignalBreakdown,
    ) -> list[str]:
        notes: list[str] = []
        if benchmark_comparison_summary is None or not benchmark_comparison_summary.compared_candidate_ids:
            notes.append("benchmark_support=none")
            return notes

        benchmark_focus = set(benchmark_comparison_summary.focus_axes)
        benchmark_preserve = set(benchmark_comparison_summary.preserve_axes)
        target_hits = sorted((set(selected_probe.target_axes) | set(committed_patch.target_axes)) & benchmark_focus)
        preserve_hits = sorted(set(committed_patch.preserve_axes) & benchmark_preserve)

        if target_hits:
            gain = 0.9 * len(target_hits)
            breakdown.benchmark_support_score += gain
            notes.append(f"benchmark_focus_support={','.join(target_hits)}")
        if preserve_hits:
            gain = 0.35 * len(preserve_hits)
            breakdown.benchmark_support_score += gain
            notes.append(f"benchmark_preserve_support={','.join(preserve_hits)}")

        candidate_count = len(benchmark_comparison_summary.compared_candidate_ids)
        if candidate_count:
            gain = min(0.75, 0.2 * candidate_count)
            breakdown.benchmark_support_score += gain
            benchmark_source = str(benchmark_comparison_summary.metadata.get("benchmark_source", "benchmark"))
            notes.append(f"benchmark_context={benchmark_source}:{candidate_count}_candidates")
        return notes

    def _score_execution_evidence(
        self,
        previous_result_summary: ResultSummary,
        updated_result_summary: ResultSummary,
        committed_patch: CommittedPatch,
        breakdown: VerifierSignalBreakdown,
    ) -> list[str]:
        notes: list[str] = []
        if previous_result_summary.summary_text != updated_result_summary.summary_text:
            breakdown.execution_evidence_score += 1.2
            notes.append("execution_evidence=summary_changed")
        else:
            breakdown.regression_notes.append("updated result summary did not change relative to previous result")

        changed_axes = set(updated_result_summary.changed_axes)
        if changed_axes & set(committed_patch.target_axes):
            breakdown.execution_evidence_score += 0.8
            notes.append("execution_evidence=target_axes_changed")
        else:
            breakdown.regression_notes.append("execution evidence did not confirm committed target axes")
        return notes

    @staticmethod
    def _collect_preserve_axes(
        selected_probe: PreviewProbe,
        preserve_constraints: List[str],
    ) -> set[str]:
        protected_axes = set(selected_probe.preserve_axes)
        preserve_text = " ".join(preserve_constraints).lower()
        for axis in set(selected_probe.target_axes) | set(selected_probe.preserve_axes):
            if axis.lower() in preserve_text:
                protected_axes.add(axis)
        return protected_axes

    @staticmethod
    def _compute_confidence(total_score: float, improved: bool) -> float:
        if improved:
            return round(min(0.95, 0.55 + max(0.0, total_score) * 0.08), 2)
        return round(max(0.2, 0.5 + min(0.0, total_score) * 0.08), 2)

    @staticmethod
    def _build_summary(
        committed_patch: CommittedPatch,
        updated_result_summary: ResultSummary,
        improved: bool,
        breakdown: VerifierSignalBreakdown,
        total_score: float,
    ) -> str:
        signal_parts = [
            f"target_alignment={breakdown.target_alignment_score:.2f}",
            f"preserve_risk={breakdown.preserve_risk_score:.2f}",
            f"benchmark_support={breakdown.benchmark_support_score:.2f}",
            f"execution_evidence={breakdown.execution_evidence_score:.2f}",
            f"total={total_score:.2f}",
        ]
        return (
            f"Verifier judged improvement={improved} for patch={committed_patch.patch_id}; "
            f"changed_axes={','.join(updated_result_summary.changed_axes) or 'none'}; "
            f"signals[{', '.join(signal_parts)}]."
        )
