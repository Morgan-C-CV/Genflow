import unittest

from app.agent.patch_planner import PatchPlanner
from app.agent.runtime_models import (
    NormalizedSchema,
    ParsedFeedbackEvidence,
    PreviewProbe,
    RepairHypothesis,
)


class PatchPlannerTest(unittest.TestCase):
    def test_plan_returns_typed_patch_over_normalized_schema(self):
        planner = PatchPlanner()
        schema = NormalizedSchema(
            prompt="a cinematic portrait",
            model="sdxl-base",
            style=["cinematic"],
        )
        probe = PreviewProbe(
            probe_id="p_001",
            summary="style mismatch",
            target_axes=["style"],
            preserve_axes=["composition"],
            preview_execution_spec={"patch_family": "resource_shift"},
            source_kind="resource_shift",
        )
        feedback = ParsedFeedbackEvidence(
            requested_changes=["make the style more vivid"],
            preserve_constraints=["Keep the composition"],
        )
        hypotheses = [
            RepairHypothesis(
                hypothesis_id="h_001",
                summary="style mismatch",
                likely_changed_axes=["style"],
                likely_preserved_axes=["composition"],
                likely_patch_family="resource_shift",
                rank=1,
            )
        ]

        patch = planner.plan(probe, schema, feedback, hypotheses)

        self.assertEqual(patch.patch_id, "cp_p_001")
        self.assertIn("style", patch.target_fields)
        self.assertIn("model", patch.target_fields)
        self.assertIn("style", patch.target_axes)
        self.assertEqual(patch.preserve_axes, ["composition"])
        self.assertIn("style", patch.changes)
        self.assertIn("model", patch.changes)
        self.assertIsInstance(patch.changes["style"], list)


if __name__ == "__main__":
    unittest.main()
