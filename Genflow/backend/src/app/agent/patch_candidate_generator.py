from __future__ import annotations

"""Generate local committed patch candidates from a selected probe."""

from typing import List

from app.agent.patch_planner import PatchPlanner
from app.agent.runtime_models import (
    CommittedPatch,
    NormalizedSchema,
    ParsedFeedbackEvidence,
    PreviewProbe,
    RepairHypothesis,
)


class PatchCandidateGenerator:
    def __init__(self, planner: PatchPlanner | None = None):
        self.planner = planner or PatchPlanner()

    def generate(
        self,
        selected_probe: PreviewProbe,
        current_schema: NormalizedSchema,
        parsed_feedback: ParsedFeedbackEvidence,
        repair_hypotheses: List[RepairHypothesis],
    ) -> List[CommittedPatch]:
        base_patch = self.planner.plan(
            selected_probe=selected_probe,
            current_schema=current_schema,
            parsed_feedback=parsed_feedback,
            repair_hypotheses=repair_hypotheses,
        )
        candidates = [self._with_metadata(base_patch, "balanced")]

        if "prompt" not in base_patch.target_fields:
            prompt_patch = self._clone_patch(
                base_patch,
                patch_id=f"{base_patch.patch_id}_prompt",
                target_fields=self._merge_unique(base_patch.target_fields, ["prompt"]),
                target_axes=self._merge_unique(base_patch.target_axes, ["composition"]),
                changes={
                    **base_patch.changes,
                    "prompt": self.planner._append_unique(
                        current_schema.prompt,
                        " | ".join(parsed_feedback.requested_changes[:1] or [selected_probe.summary]),
                    ),
                },
                rationale=f"{base_patch.rationale} Candidate variant: prompt reinforcement.",
                metadata={"candidate_kind": "prompt_reinforcement"},
            )
            candidates.append(prompt_patch)

        if "style" in base_patch.target_fields or "model" in base_patch.target_fields:
            style_patch = self._clone_patch(
                base_patch,
                patch_id=f"{base_patch.patch_id}_style",
                target_fields=self._merge_unique(base_patch.target_fields, ["style"]),
                target_axes=self._merge_unique(base_patch.target_axes, ["style"]),
                changes={
                    **base_patch.changes,
                    "style": self.planner._merge_unique(
                        current_schema.style,
                        self.planner._extract_style_terms(parsed_feedback, selected_probe, repair_hypotheses) + ["dramatic"],
                    ),
                },
                rationale=f"{base_patch.rationale} Candidate variant: stronger style shift.",
                metadata={"candidate_kind": "style_shift"},
            )
            candidates.append(style_patch)

        safe_target_axes = [axis for axis in base_patch.target_axes if axis not in base_patch.preserve_axes]
        if safe_target_axes != base_patch.target_axes:
            preserve_safe_patch = self._clone_patch(
                base_patch,
                patch_id=f"{base_patch.patch_id}_safe",
                target_axes=safe_target_axes,
                target_fields=self.planner._target_fields_for_axes(safe_target_axes),
                rationale=f"{base_patch.rationale} Candidate variant: preserve-safe reduction.",
                metadata={"candidate_kind": "preserve_safe"},
            )
            candidates.append(preserve_safe_patch)

        return candidates[:4]

    @staticmethod
    def _clone_patch(
        patch: CommittedPatch,
        *,
        patch_id: str,
        target_fields: List[str] | None = None,
        target_axes: List[str] | None = None,
        changes: dict | None = None,
        rationale: str | None = None,
        metadata: dict | None = None,
    ) -> CommittedPatch:
        return CommittedPatch(
            patch_id=patch_id,
            target_fields=list(target_fields if target_fields is not None else patch.target_fields),
            target_axes=list(target_axes if target_axes is not None else patch.target_axes),
            preserve_axes=list(patch.preserve_axes),
            changes=dict(changes if changes is not None else patch.changes),
            rationale=rationale if rationale is not None else patch.rationale,
            metadata=dict(metadata or patch.metadata),
        )

    @staticmethod
    def _with_metadata(patch: CommittedPatch, candidate_kind: str) -> CommittedPatch:
        return CommittedPatch(
            patch_id=patch.patch_id,
            target_fields=list(patch.target_fields),
            target_axes=list(patch.target_axes),
            preserve_axes=list(patch.preserve_axes),
            changes=dict(patch.changes),
            rationale=patch.rationale,
            metadata={"candidate_kind": candidate_kind},
        )

    @staticmethod
    def _merge_unique(existing: List[str], additions: List[str]) -> List[str]:
        ordered = list(existing)
        seen = set(existing)
        for item in additions:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered
