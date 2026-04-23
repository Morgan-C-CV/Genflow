from __future__ import annotations

from typing import Dict, List

from app.agent.runtime_models import (
    CommittedPatch,
    NormalizedSchema,
    ParsedFeedbackEvidence,
    PreviewProbe,
    RepairHypothesis,
)


AXIS_TO_SCHEMA_FIELDS: Dict[str, List[str]] = {
    "style": ["style", "model"],
    "composition": ["prompt"],
    "color_palette": ["style", "prompt"],
    "lighting_vibe": ["prompt"],
    "background_setting": ["prompt"],
    "subject": ["prompt"],
}


class PatchPlanner:
    def plan(
        self,
        selected_probe: PreviewProbe,
        current_schema: NormalizedSchema,
        parsed_feedback: ParsedFeedbackEvidence,
        repair_hypotheses: List[RepairHypothesis],
    ) -> CommittedPatch:
        target_axes = list(selected_probe.target_axes)
        preserve_axes = list(selected_probe.preserve_axes)
        target_fields = self._target_fields_for_axes(target_axes)
        changes: Dict[str, object] = {}

        if "prompt" in target_fields:
            prompt_addition = " | ".join(parsed_feedback.requested_changes[:2] or [selected_probe.summary])
            changes["prompt"] = self._append_unique(current_schema.prompt, prompt_addition)

        if "style" in target_fields:
            style_additions = self._extract_style_terms(parsed_feedback, selected_probe, repair_hypotheses)
            changes["style"] = self._merge_unique(current_schema.style, style_additions)

        if "model" in target_fields:
            base_model = current_schema.model or "default-model"
            changes["model"] = f"{base_model}-patched"

        rationale = (
            f"Commit selected probe {selected_probe.probe_id} using "
            f"{selected_probe.preview_execution_spec.get('patch_family', 'local_adjustment')} "
            f"to address axes={','.join(target_axes) or 'none'} while preserving "
            f"{','.join(preserve_axes) or 'none'}."
        )
        return CommittedPatch(
            patch_id=f"cp_{selected_probe.probe_id}",
            target_fields=target_fields,
            target_axes=target_axes,
            preserve_axes=preserve_axes,
            changes=changes,
            rationale=rationale,
        )

    @staticmethod
    def _target_fields_for_axes(target_axes: List[str]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for axis in target_axes:
            for field in AXIS_TO_SCHEMA_FIELDS.get(axis, ["prompt"]):
                if field not in seen:
                    seen.add(field)
                    ordered.append(field)
        return ordered or ["prompt"]

    @staticmethod
    def _append_unique(base: str, addition: str) -> str:
        base = base.strip()
        addition = addition.strip()
        if not addition:
            return base
        if addition in base:
            return base
        return f"{base} | {addition}" if base else addition

    @staticmethod
    def _merge_unique(existing: List[str], additions: List[str]) -> List[str]:
        ordered = list(existing)
        seen = set(existing)
        for item in additions:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    @staticmethod
    def _extract_style_terms(
        parsed_feedback: ParsedFeedbackEvidence,
        selected_probe: PreviewProbe,
        repair_hypotheses: List[RepairHypothesis],
    ) -> List[str]:
        terms: List[str] = []
        text_parts = list(parsed_feedback.requested_changes[:2]) + [selected_probe.summary]
        for text in text_parts:
            lowered = text.lower()
            for marker in ["vivid", "bright", "muted", "cinematic", "illustrated", "dramatic", "soft"]:
                if marker in lowered and marker not in terms:
                    terms.append(marker)
        if not terms:
            for hypothesis in repair_hypotheses[:1]:
                for axis in hypothesis.likely_changed_axes:
                    if axis not in terms:
                        terms.append(axis)
        return terms or ["refined"]
