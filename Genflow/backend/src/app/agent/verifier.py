from __future__ import annotations

from typing import List

from app.agent.runtime_models import CommittedPatch, PreviewProbe, ResultSummary, VerifierResult


class Verifier:
    def verify(
        self,
        previous_result_summary: ResultSummary,
        updated_result_summary: ResultSummary,
        selected_probe: PreviewProbe,
        committed_patch: CommittedPatch,
        preserve_constraints: List[str],
    ) -> VerifierResult:
        regression_notes: List[str] = []
        preserve_text = " ".join(preserve_constraints).lower()

        for axis in selected_probe.preserve_axes:
            if axis in selected_probe.target_axes:
                regression_notes.append(f"preserve axis '{axis}' overlaps with selected change scope")
            if axis in preserve_text and axis in committed_patch.target_axes:
                regression_notes.append(f"preserve axis '{axis}' was included in committed patch target axes")

        improved = bool(set(updated_result_summary.changed_axes) & set(committed_patch.target_axes)) and not regression_notes
        confidence = 0.82 if improved else 0.38
        continue_recommended = not improved
        summary = (
            f"Verifier judged improvement={improved} for patch={committed_patch.patch_id}; "
            f"changed_axes={','.join(updated_result_summary.changed_axes) or 'none'}."
        )

        if previous_result_summary.summary_text == updated_result_summary.summary_text:
            regression_notes.append("updated result summary did not change relative to previous result")
            improved = False
            continue_recommended = True
            confidence = min(confidence, 0.35)
            summary = (
                f"Verifier judged improvement={improved} for patch={committed_patch.patch_id}; "
                "updated result summary did not materially change."
            )

        return VerifierResult(
            improved=improved,
            continue_recommended=continue_recommended,
            confidence=confidence,
            regression_notes=regression_notes,
            summary=summary,
        )
