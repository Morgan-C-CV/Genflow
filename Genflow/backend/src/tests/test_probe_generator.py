import unittest

from app.agent.probe_generator import PreviewProbeGenerator
from app.agent.refinement_benchmark_retriever import RefinementBenchmarkCandidate, RefinementBenchmarkSet
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
        self.assertNotIn("benchmark_context", probes[0].preview_execution_spec)

    def test_generate_with_benchmark_adds_lightweight_benchmark_context(self):
        generator = PreviewProbeGenerator()
        schema = NormalizedSchema(model="sdxl-base", sampler="DPM++ 2M", style=["cinematic"])
        feedback = ParsedFeedbackEvidence(
            dissatisfaction_scope=["style"],
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
            )
        ]
        benchmark_set = RefinementBenchmarkSet(
            benchmark_id="refinement-benchmark-1",
            benchmark_kind="refinement_local_comparison",
            benchmark_source="refinement_search_bundle",
            anchor_ids=[101, 102],
            comparison_candidates=[
                RefinementBenchmarkCandidate(
                    candidate_id="benchmark-candidate-101",
                    reference_id=101,
                )
            ],
            selection_rationale=["focus_axes=style", "preserve=Keep the composition"],
        )

        probes = generator.generate(
            schema,
            feedback,
            hypotheses,
            selected_gallery_index=7,
            selected_reference_ids=[101, 102],
            refinement_benchmark_set=benchmark_set,
        )

        self.assertEqual(len(probes), 2)
        self.assertIn("[refinement_search_bundle]", probes[0].summary)
        self.assertEqual(
            probes[0].preview_execution_spec["benchmark_context"]["benchmark_source"],
            "refinement_search_bundle",
        )
        self.assertEqual(
            probes[0].preview_execution_spec["benchmark_context"]["anchor_ids"],
            [101, 102],
        )


if __name__ == "__main__":
    unittest.main()
