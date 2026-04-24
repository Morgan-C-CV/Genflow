import unittest

from app.agent.benchmark_comparison_summary import BenchmarkComparisonSummary
from app.agent.pbo_probe_ranker import rank_probe_candidates
from app.agent.refinement_benchmark_retriever import RefinementBenchmarkSet
from app.agent.runtime_models import ParsedFeedbackEvidence, PreviewProbe


class PboProbeRankerTest(unittest.TestCase):
    def _make_probes(self):
        return [
            PreviewProbe(
                probe_id="p_color",
                summary="color direction",
                target_axes=["color_palette"],
                preserve_axes=["composition"],
                preview_execution_spec={"patch_family": "prompt_color_adjustment"},
                source_kind="schema_variation",
            ),
            PreviewProbe(
                probe_id="p_style",
                summary="style direction",
                target_axes=["style"],
                preserve_axes=["composition"],
                preview_execution_spec={"patch_family": "resource_shift"},
                source_kind="resource_shift",
            ),
            PreviewProbe(
                probe_id="p_composition",
                summary="composition direction",
                target_axes=["composition"],
                preserve_axes=[],
                preview_execution_spec={"patch_family": "small_prompt_adjustment"},
                source_kind="gallery",
            ),
        ]

    def test_ranking_changes_with_dissatisfaction_axes(self):
        probes = self._make_probes()
        feedback = ParsedFeedbackEvidence(
            dissatisfaction_scope=["style"],
            preserve_constraints=["Keep the composition"],
        )

        ranked = rank_probe_candidates(probes, feedback)

        self.assertEqual(ranked[0].probe_id, "p_style")
        self.assertLess(ranked[-1].preview_execution_spec["pbo_score"], ranked[0].preview_execution_spec["pbo_score"])

    def test_ranking_changes_with_preserve_constraints(self):
        probes = self._make_probes()
        feedback = ParsedFeedbackEvidence(
            dissatisfaction_scope=["style"],
            preserve_constraints=["composition"],
        )

        ranked = rank_probe_candidates(probes, feedback)

        self.assertEqual(ranked[-1].probe_id, "p_composition")
        self.assertTrue(any("preserve_collision=composition" in item for item in ranked[-1].preview_execution_spec["pbo_rationale"]))

    def test_ranking_changes_with_benchmark_comparison_summary(self):
        probes = self._make_probes()
        feedback = ParsedFeedbackEvidence(dissatisfaction_scope=["color_palette"])
        benchmark_summary = BenchmarkComparisonSummary(
            compared_anchor_ids=[101, 102],
            compared_candidate_ids=["benchmark-candidate-101", "benchmark-candidate-102"],
            focus_axes=["style"],
            preserve_axes=["composition"],
            confidence_hint=0.8,
            metadata={"benchmark_source": "refinement_search_bundle"},
        )
        benchmark_set = RefinementBenchmarkSet(
            benchmark_id="refinement-benchmark-1",
            benchmark_kind="refinement_local_comparison",
            benchmark_source="refinement_search_bundle",
        )

        ranked = rank_probe_candidates(
            probes,
            feedback,
            benchmark_comparison_summary=benchmark_summary,
            refinement_benchmark_set=benchmark_set,
        )

        self.assertEqual(ranked[0].probe_id, "p_style")
        self.assertIn("pbo_score", ranked[0].preview_execution_spec)
        self.assertTrue(any("benchmark_focus=style" in item for item in ranked[0].preview_execution_spec["pbo_rationale"]))

    def test_ranking_is_stable_without_benchmark(self):
        probes = self._make_probes()
        feedback = ParsedFeedbackEvidence(dissatisfaction_scope=["style"])

        ranked = rank_probe_candidates(probes, feedback, benchmark_comparison_summary=None, refinement_benchmark_set=None)

        self.assertEqual([probe.probe_id for probe in ranked], ["p_style", "p_color", "p_composition"])


if __name__ == "__main__":
    unittest.main()
