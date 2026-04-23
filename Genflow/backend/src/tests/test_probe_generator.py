import unittest

from app.agent.probe_generator import PreviewProbeGenerator
from app.agent.runtime_models import (
    NormalizedSchema,
    ParsedFeedbackEvidence,
    RepairHypothesis,
)


class ProbeGeneratorTest(unittest.TestCase):
    def test_generate_returns_two_to_four_typed_probes(self):
        generator = PreviewProbeGenerator()
        schema = NormalizedSchema(model="sdxl-base", sampler="DPM++ 2M", style=["cinematic"])
        feedback = ParsedFeedbackEvidence(
            dissatisfaction_scope=["style", "color_palette"],
            preserve_constraints=["Keep the composition"],
            requested_changes=["make the style more vivid"],
            uncertainty_estimate=0.3,
        )
        hypotheses = [
            RepairHypothesis(
                hypothesis_id="h_001",
                summary="style mismatch",
                likely_changed_axes=["style"],
                likely_preserved_axes=["composition"],
                likely_patch_family="resource_shift",
                rank=1,
            ),
            RepairHypothesis(
                hypothesis_id="h_002",
                summary="color mismatch",
                likely_changed_axes=["color_palette"],
                likely_preserved_axes=["composition"],
                likely_patch_family="prompt_color_adjustment",
                rank=2,
            ),
        ]

        probes = generator.generate(schema, feedback, hypotheses, selected_gallery_index=7, selected_reference_ids=[101, 102])

        self.assertGreaterEqual(len(probes), 2)
        self.assertLessEqual(len(probes), 4)
        self.assertEqual(probes[0].probe_id, "p_001")
        self.assertEqual(probes[0].target_axes, ["style"])
        self.assertEqual(probes[0].preserve_axes, ["composition"])
        self.assertIn("patch_family", probes[0].preview_execution_spec)
        self.assertTrue(probes[0].source_kind)


if __name__ == "__main__":
    unittest.main()
