from __future__ import annotations

from typing import Iterable, List

from app.agent.runtime_models import NormalizedSchema, ParsedFeedbackEvidence, RepairHypothesis, ResultSummary


PATCH_FAMILY_BY_AXIS = {
    "style": "resource_shift",
    "composition": "prompt_composition_adjustment",
    "color_palette": "prompt_color_adjustment",
    "lighting_vibe": "lighting_prompt_adjustment",
    "background_setting": "background_prompt_adjustment",
    "subject": "subject_prompt_adjustment",
}


class RepairHypothesisBuilder:
    def build(
        self,
        current_schema: NormalizedSchema,
        current_result_summary: ResultSummary,
        feedback_evidence: ParsedFeedbackEvidence,
        history: Iterable[str] | None = None,
    ) -> List[RepairHypothesis]:
        history = list(history or [])
        preserve_axes = self._infer_preserved_axes(feedback_evidence, current_result_summary)
        axes = feedback_evidence.dissatisfaction_scope or ["style", "color_palette"]

        hypotheses: List[RepairHypothesis] = []
        for rank, axis in enumerate(axes[:4], start=1):
            patch_family = PATCH_FAMILY_BY_AXIS.get(axis, "prompt_adjustment")
            schema_hint = self._schema_hint(axis, current_schema)
            hypotheses.append(
                RepairHypothesis(
                    hypothesis_id=f"h_{rank:03d}",
                    summary=f"{axis} is the most likely mismatch; {schema_hint}",
                    likely_changed_axes=[axis],
                    likely_preserved_axes=preserve_axes,
                    likely_patch_family=patch_family,
                    rank=rank,
                )
            )

        if len(hypotheses) < 2:
            hypotheses.append(
                RepairHypothesis(
                    hypothesis_id=f"h_{len(hypotheses) + 1:03d}",
                    summary="style-resource alignment is weak relative to the requested direction.",
                    likely_changed_axes=["style"],
                    likely_preserved_axes=preserve_axes,
                    likely_patch_family="resource_shift",
                    rank=len(hypotheses) + 1,
                )
            )

        if feedback_evidence.uncertainty_estimate >= 0.65 and len(hypotheses) < 4:
            hypotheses.append(
                RepairHypothesis(
                    hypothesis_id=f"h_{len(hypotheses) + 1:03d}",
                    summary="feedback is still ambiguous; a small prompt-space adjustment should be tested before larger changes.",
                    likely_changed_axes=axes[:1],
                    likely_preserved_axes=preserve_axes,
                    likely_patch_family="small_prompt_adjustment",
                    rank=len(hypotheses) + 1,
                )
            )

        if history and len(hypotheses) < 4:
            hypotheses.append(
                RepairHypothesis(
                    hypothesis_id=f"h_{len(hypotheses) + 1:03d}",
                    summary="prior feedback history suggests preserving accepted aspects while narrowing the next repair scope.",
                    likely_changed_axes=axes[:1],
                    likely_preserved_axes=preserve_axes,
                    likely_patch_family="constrained_local_adjustment",
                    rank=len(hypotheses) + 1,
                )
            )

        return hypotheses[:4]

    @staticmethod
    def _infer_preserved_axes(
        feedback_evidence: ParsedFeedbackEvidence,
        current_result_summary: ResultSummary,
    ) -> List[str]:
        preserved = list(current_result_summary.preserved_axes)
        for clause in feedback_evidence.preserve_constraints:
            lower_clause = clause.lower()
            if "composition" in lower_clause or "构图" in clause:
                preserved.append("composition")
            if "style" in lower_clause or "风格" in clause:
                preserved.append("style")
            if "background" in lower_clause or "背景" in clause:
                preserved.append("background_setting")
            if "subject" in lower_clause or "人物" in clause or "主体" in clause:
                preserved.append("subject")
        seen = set()
        ordered = []
        for item in preserved:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    @staticmethod
    def _schema_hint(axis: str, current_schema: NormalizedSchema) -> str:
        if axis == "style" and current_schema.model:
            return f"current model={current_schema.model} may be biasing style."
        if axis == "color_palette" and current_schema.style:
            return f"current style tags={', '.join(current_schema.style[:2]) or 'none'} may be steering color."
        if axis == "lighting_vibe" and current_schema.prompt:
            return "current prompt likely under-specifies lighting direction."
        if axis == "composition" and current_schema.prompt:
            return "current prompt likely needs stronger composition constraints."
        return "current schema likely needs a local adjustment."
