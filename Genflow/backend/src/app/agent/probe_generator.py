from __future__ import annotations

from typing import List

from app.agent.runtime_models import (
    NormalizedSchema,
    ParsedFeedbackEvidence,
    PreviewProbe,
    RepairHypothesis,
)


class PreviewProbeGenerator:
    def generate(
        self,
        current_schema: NormalizedSchema,
        parsed_feedback: ParsedFeedbackEvidence,
        repair_hypotheses: List[RepairHypothesis],
        selected_gallery_index: int | None = None,
        selected_reference_ids: List[int] | None = None,
    ) -> List[PreviewProbe]:
        probes: List[PreviewProbe] = []
        selected_reference_ids = selected_reference_ids or []

        for index, hypothesis in enumerate(repair_hypotheses[:4], start=1):
            source_kind = self._source_kind_for_patch_family(hypothesis.likely_patch_family)
            execution_spec = {
                "patch_family": hypothesis.likely_patch_family,
                "reference_anchor": selected_gallery_index,
                "reference_ids": list(selected_reference_ids[:3]),
                "schema_hint": {
                    "model": current_schema.model,
                    "sampler": current_schema.sampler,
                    "style": list(current_schema.style[:2]),
                },
                "requested_changes": list(parsed_feedback.requested_changes[:2]),
            }
            probes.append(
                PreviewProbe(
                    probe_id=f"p_{index:03d}",
                    summary=hypothesis.summary,
                    target_axes=list(hypothesis.likely_changed_axes),
                    preserve_axes=list(hypothesis.likely_preserved_axes),
                    preview_execution_spec=execution_spec,
                    source_kind=source_kind,
                )
            )

        if len(probes) < 2:
            probes.append(
                PreviewProbe(
                    probe_id=f"p_{len(probes) + 1:03d}",
                    summary="Test a constrained local prompt variation before any larger edit.",
                    target_axes=list(parsed_feedback.dissatisfaction_scope[:1] or ["style"]),
                    preserve_axes=["composition"] if "composition" in " ".join(parsed_feedback.preserve_constraints).lower() else [],
                    preview_execution_spec={
                        "patch_family": "small_prompt_adjustment",
                        "reference_anchor": selected_gallery_index,
                        "reference_ids": list(selected_reference_ids[:2]),
                    },
                    source_kind="schema_variation",
                )
            )

        return probes[:4]

    @staticmethod
    def _source_kind_for_patch_family(patch_family: str) -> str:
        if patch_family == "resource_shift":
            return "resource_shift"
        if "prompt" in patch_family:
            return "schema_variation"
        return "gallery"
