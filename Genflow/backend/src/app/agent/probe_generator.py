from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from app.agent.refinement_benchmark_retriever import RefinementBenchmarkSet
from app.agent.runtime_models import (
    NormalizedSchema,
    ParsedFeedbackEvidence,
    PreviewProbe,
    RepairHypothesis,
)

# Hyper Candidate Strategy (thesis 4.3): draw three candidates at increasing
# distance from the current result — a close candidate, a local exploratory
# candidate, and a farther candidate.
HCS_REGIMES: Tuple[str, str, str] = ("close", "exploratory", "far")

# How far each patch family moves away from the current result. Prompt-level
# tweaks stay close; swapping the resource shifts the whole look.
_FAMILY_DISTANCE = {
    "small_prompt_adjustment": 1,
    "lighting_prompt_adjustment": 1,
    "prompt_color_adjustment": 2,
    "prompt_composition_adjustment": 2,
    "background_prompt_adjustment": 3,
    "subject_prompt_adjustment": 3,
    "resource_shift": 4,
}

# Used when the hypothesis builder yields fewer than three hypotheses.
_FALLBACK_FAMILIES = {
    "close": ("small_prompt_adjustment", "Test a constrained local prompt variation."),
    "exploratory": ("prompt_composition_adjustment", "Explore a nearby compositional alternative."),
    "far": ("resource_shift", "Try a distinctly different resource direction."),
}


def _regime_for_family(patch_family: str) -> str:
    """Map a patch family to the HCS regime its distance implies."""
    distance = _FAMILY_DISTANCE.get(patch_family, 2)
    if distance <= 1:
        return "close"
    if distance == 2:
        return "exploratory"
    return "far"


class PreviewProbeGenerator:
    def generate(
        self,
        current_schema: NormalizedSchema,
        parsed_feedback: ParsedFeedbackEvidence,
        repair_hypotheses: List[RepairHypothesis],
        selected_gallery_index: int | None = None,
        selected_reference_ids: List[int] | None = None,
        refinement_benchmark_set: RefinementBenchmarkSet | None = None,
    ) -> List[PreviewProbe]:
        selected_reference_ids = selected_reference_ids or []
        benchmark_context = self._build_benchmark_context(refinement_benchmark_set)

        ranked = sorted(
            repair_hypotheses,
            key=lambda hypothesis: _FAMILY_DISTANCE.get(hypothesis.likely_patch_family, 2),
        )
        assigned = self._assign_regimes(ranked, parsed_feedback)

        probes: List[PreviewProbe] = []
        for position, (regime, hypothesis) in enumerate(zip(HCS_REGIMES, assigned), start=1):
            probes.append(
                self._build_probe(
                    position=position,
                    regime=regime,
                    hypothesis=hypothesis,
                    current_schema=current_schema,
                    parsed_feedback=parsed_feedback,
                    selected_gallery_index=selected_gallery_index,
                    selected_reference_ids=selected_reference_ids,
                    benchmark_context=benchmark_context,
                )
            )
        return probes

    @staticmethod
    def _assign_regimes(
        ranked: Sequence[RepairHypothesis],
        parsed_feedback: ParsedFeedbackEvidence,
    ) -> List[RepairHypothesis]:
        """Place each hypothesis in the regime its distance implies, fill the gaps.

        A resource swap is inherently a far move and a lighting tweak is inherently
        close, so the hypothesis's patch family decides its slot rather than its
        position in the list. Regimes with no hypothesis get a synthetic one, so HCS
        always yields exactly three candidates at increasing distance.
        """
        axes = list(parsed_feedback.dissatisfaction_scope) or ["style"]
        preserve = list(parsed_feedback.preserve_constraints)

        slots: dict[str, RepairHypothesis] = {}
        for hypothesis in ranked:  # already ascending by distance
            slots.setdefault(_regime_for_family(hypothesis.likely_patch_family), hypothesis)

        return [
            slots.get(regime) or PreviewProbeGenerator._synthetic(regime, axes, preserve)
            for regime in HCS_REGIMES
        ]

    @staticmethod
    def _synthetic(regime: str, axes: List[str], preserve: List[str]) -> RepairHypothesis:
        family, summary = _FALLBACK_FAMILIES[regime]
        return RepairHypothesis(
            hypothesis_id=f"h_{regime}",
            summary=summary,
            likely_changed_axes=list(axes[:2]),
            likely_preserved_axes=list(preserve),
            likely_patch_family=family,
            rank=HCS_REGIMES.index(regime),
        )

    @staticmethod
    def _build_probe(
        position: int,
        regime: str,
        hypothesis: RepairHypothesis,
        current_schema: NormalizedSchema,
        parsed_feedback: ParsedFeedbackEvidence,
        selected_gallery_index: Optional[int],
        selected_reference_ids: List[int],
        benchmark_context: dict,
    ) -> PreviewProbe:
        execution_spec = {
            "patch_family": hypothesis.likely_patch_family,
            "hcs_regime": regime,
            "reference_anchor": selected_gallery_index,
            "reference_ids": list(selected_reference_ids[:3]),
            "schema_hint": {
                "model": current_schema.model,
                "sampler": current_schema.sampler,
                "style": list(current_schema.style[:2]),
            },
            "requested_changes": list(parsed_feedback.requested_changes[:2]),
        }
        if benchmark_context:
            execution_spec["benchmark_context"] = benchmark_context

        return PreviewProbe(
            probe_id=f"p_{position:03d}",
            summary=PreviewProbeGenerator._build_probe_summary(
                hypothesis.summary, benchmark_context
            ),
            target_axes=list(hypothesis.likely_changed_axes),
            preserve_axes=list(hypothesis.likely_preserved_axes),
            preview_execution_spec=execution_spec,
            source_kind=PreviewProbeGenerator._source_kind_for_patch_family(
                hypothesis.likely_patch_family
            ),
            hcs_regime=regime,
        )

    @staticmethod
    def _source_kind_for_patch_family(patch_family: str) -> str:
        if patch_family == "resource_shift":
            return "resource_shift"
        if "prompt" in patch_family:
            return "schema_variation"
        return "gallery"

    @staticmethod
    def _build_benchmark_context(
        refinement_benchmark_set: RefinementBenchmarkSet | None,
    ) -> dict:
        if refinement_benchmark_set is None or not refinement_benchmark_set.benchmark_id:
            return {}
        return {
            "benchmark_source": refinement_benchmark_set.benchmark_source,
            "anchor_ids": list(refinement_benchmark_set.anchor_ids[:3]),
            "rationale_summary": refinement_benchmark_set.selection_rationale[:2],
        }

    @staticmethod
    def _build_probe_summary(summary: str, benchmark_context: dict) -> str:
        if not benchmark_context:
            return summary
        benchmark_source = benchmark_context.get("benchmark_source", "")
        if not benchmark_source:
            return summary
        return f"{summary} [{benchmark_source}]"
