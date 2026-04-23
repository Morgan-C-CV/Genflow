import unittest

from app.agent.feedback_parser import FeedbackParser


class FeedbackParserTest(unittest.TestCase):
    def test_parse_extracts_scope_preserve_change_and_uncertainty(self):
        parser = FeedbackParser()
        evidence = parser.parse(
            feedback_text="Keep the composition, but make the style more vivid and the color less dull."
        )

        self.assertIn("composition", evidence.preserve_constraints[0].lower())
        self.assertIn("style", evidence.dissatisfaction_scope)
        self.assertIn("color_palette", evidence.dissatisfaction_scope)
        self.assertTrue(any("more vivid" in item.lower() for item in evidence.requested_changes))
        self.assertLess(evidence.uncertainty_estimate, 0.7)

    def test_parse_handles_ambiguous_feedback(self):
        parser = FeedbackParser()
        evidence = parser.parse(feedback_text="This is not right, make it better.")

        self.assertEqual(evidence.dissatisfaction_scope, [])
        self.assertGreaterEqual(evidence.uncertainty_estimate, 0.7)
        self.assertIn("no_explicit_axis_detected", evidence.parser_notes)


if __name__ == "__main__":
    unittest.main()
