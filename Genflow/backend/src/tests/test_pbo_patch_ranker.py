import unittest

from app.agent.benchmark_comparison_summary import BenchmarkComparisonSummary
from app.agent.pbo_patch_ranker import rank_patch_candidates
from app.agent.refinement_benchmark_retriever import RefinementBenchmarkSet
from app.agent.runtime_models import CommittedPatch, ParsedFeedbackEvidence


class PboPatchRankerTest(unittest.TestCase):
    def _make_patch_candidates(self):
        return [
            CommittedPatch(
                patch_id="cp_composition",
                target_fields=["prompt"],
                target_axes=["composition"],
                preserve_axes=["composition"],
                changes={"prompt": "preserve composition"},
                rationale="composition patch",
            ),
            CommittedPatch(
                patch_id="cp_style",
                target_fields=["style", "model"],
                target_axes=["style"],
                preserve_axes=["composition"],
                changes={"style": ["cinematic", "vivid"], "model": "sdxl-base-patched"},
                rationale="style patch",
            ),
            CommittedPatch(
                patch_id="cp_color",
                target_fields=["style", "prompt"],
                target_axes=["color_palette"],
                preserve_axes=["composition"],
                changes={"style": ["cinematic", "warm"], "prompt": "color shift"},
                rationale="color patch",
            ),
        ]

    def test_patch_ranking_changes_with_dissatisfaction_axes(self):
        feedback = ParsedFeedbackEvidence(dissatisfaction_scope=["style"], preserve_constraints=["composition"])

        ranked = rank_patch_candidates(self._make_patch_candidates(), feedback)

        self.assertEqual(ranked[0].patch_id, "cp_style")

    def test_patch_ranking_changes_with_preserve_constraints(self):
        feedback = ParsedFeedbackEvidence(dissatisfaction_scope=["style"], preserve_constraints=["composition"])

        ranked = rank_patch_candidates(self._make_patch_candidates(), feedback)

        self.assertEqual(ranked[-1].patch_id, "cp_composition")
        self.assertTrue(any("preserve_collision=composition" in item for item in ranked[-1].metadata["pbo_rationale"]))

    def test_patch_ranking_changes_with_benchmark_summary(self):
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

        ranked = rank_patch_candidates(
            self._make_patch_candidates(),
            feedback,
            benchmark_comparison_summary=benchmark_summary,
            refinement_benchmark_set=benchmark_set,
        )

        self.assertEqual(ranked[0].patch_id, "cp_style")
        self.assertTrue(any("benchmark_focus=style" in item for item in ranked[0].metadata["pbo_rationale"]))

    def test_patch_ranking_is_stable_without_benchmark(self):
        feedback = ParsedFeedbackEvidence(dissatisfaction_scope=["style"])

        ranked = rank_patch_candidates(self._make_patch_candidates(), feedback)

        self.assertEqual([patch.patch_id for patch in ranked], ["cp_style", "cp_color", "cp_composition"])


if __name__ == "__main__":
    unittest.main()
