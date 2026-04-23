import unittest

from app.agent.repair_hypothesis import RepairHypothesisBuilder
from app.agent.runtime_models import NormalizedSchema, ParsedFeedbackEvidence, ResultSummary


class RepairHypothesisTest(unittest.TestCase):
    def test_build_returns_ranked_hypotheses(self):
        builder = RepairHypothesisBuilder()
        schema = NormalizedSchema(model="sdxl-base", style=["cinematic", "muted"])
        summary = ResultSummary(summary_text="Mock result", preserved_axes=["composition"])
        feedback = ParsedFeedbackEvidence(
            dissatisfaction_scope=["style", "color_palette"],
            preserve_constraints=["Keep the composition"],
            requested_changes=["make the style more vivid"],
            uncertainty_estimate=0.3,
            raw_feedback="Keep the composition, but make the style more vivid.",
        )

        hypotheses = builder.build(schema, summary, feedback, history=["first feedback"])

        self.assertGreaterEqual(len(hypotheses), 2)
        self.assertLessEqual(len(hypotheses), 4)
        self.assertEqual(hypotheses[0].hypothesis_id, "h_001")
        self.assertIn("style", hypotheses[0].likely_changed_axes)
        self.assertIn("composition", hypotheses[0].likely_preserved_axes)
        self.assertTrue(hypotheses[0].likely_patch_family)

    def test_build_adds_fallback_when_feedback_is_ambiguous(self):
        builder = RepairHypothesisBuilder()
        schema = NormalizedSchema(model="sdxl-base")
        summary = ResultSummary(summary_text="Mock result")
        feedback = ParsedFeedbackEvidence(
            dissatisfaction_scope=[],
            preserve_constraints=[],
            requested_changes=[],
            uncertainty_estimate=0.9,
            raw_feedback="make it better",
        )

        hypotheses = builder.build(schema, summary, feedback, history=[])

        self.assertGreaterEqual(len(hypotheses), 2)
        self.assertTrue(any(item.likely_patch_family == "small_prompt_adjustment" for item in hypotheses))


if __name__ == "__main__":
    unittest.main()
