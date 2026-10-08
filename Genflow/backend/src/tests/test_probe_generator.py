import unittest

from app.agent.probe_generator import HCS_REGIMES, PreviewProbeGenerator
from app.agent.refinement_benchmark_retriever import RefinementBenchmarkCandidate, RefinementBenchmarkSet
from app.agent.runtime_models import (
    NormalizedSchema,
    ParsedFeedbackEvidence,
    RepairHypothesis,
)


class ProbeGeneratorTest(unittest.TestCase):
    """Hyper Candidate Strategy: exactly three probes at increasing distance."""

    def test_generate_returns_three_hcs_probes_ordered_by_distance(self):
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

        self.assertEqual(len(probes), 3)
        self.assertEqual([probe.hcs_regime for probe in probes], list(HCS_REGIMES))
        self.assertEqual([probe.probe_id for probe in probes], ["p_001", "p_002", "p_003"])

        close, exploratory, far = probes
        # A colour tweak is a close move; swapping the resource is a far one.
        self.assertEqual(exploratory.preview_execution_spec["patch_family"], "prompt_color_adjustment")
        self.assertEqual(exploratory.target_axes, ["color_palette"])
        self.assertEqual(far.preview_execution_spec["patch_family"], "resource_shift")
        self.assertEqual(far.target_axes, ["style"])
        # No close-distance hypothesis exists, so one is synthesised.
        self.assertEqual(close.preview_execution_spec["patch_family"], "small_prompt_adjustment")

        for probe in probes:
            self.assertIn("patch_family", probe.preview_execution_spec)
            self.assertEqual(probe.preview_execution_spec["hcs_regime"], probe.hcs_regime)
            self.assertTrue(probe.source_kind)
            self.assertTrue(probe.preserve_axes)
            self.assertNotIn("benchmark_context", probe.preview_execution_spec)

    def test_generate_keeps_a_lone_hypothesis_in_its_own_regime(self):
        generator = PreviewProbeGenerator()
        schema = NormalizedSchema(model="sdxl-base", sampler="DPM++ 2M")
        feedback = ParsedFeedbackEvidence(
            dissatisfaction_scope=["lighting_vibe"],
            preserve_constraints=["Keep the composition"],
            uncertainty_estimate=0.2,
        )
        hypotheses = [
            RepairHypothesis(
                hypothesis_id="h_001",
                summary="lighting mismatch",
                likely_changed_axes=["lighting_vibe"],
                likely_preserved_axes=["composition"],
                likely_patch_family="lighting_prompt_adjustment",
                rank=1,
            )
        ]

        probes = generator.generate(schema, feedback, hypotheses)

        self.assertEqual(len(probes), 3)
        # A lighting tweak is close, so it must not be promoted to the far slot.
        self.assertEqual(probes[0].preview_execution_spec["patch_family"], "lighting_prompt_adjustment")
        self.assertEqual(probes[0].hcs_regime, "close")
        self.assertEqual(probes[2].hcs_regime, "far")

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

        self.assertEqual(len(probes), 3)
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
