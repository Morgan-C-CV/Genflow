import unittest

from app.agent.memory import AgentMemoryService
from app.agent.pbo_benchmark_ranker import rank_benchmark_candidates
from app.agent.refinement_benchmark_retriever import RefinementBenchmarkCandidate
from app.agent.runtime_models import ParsedFeedbackEvidence


class PboBenchmarkRankerTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.current_schema.model = "sdxl-base"
        session.current_result_summary.summary_text = "Current output is compositionally sound but stylistically flat."
        session.parsed_feedback = ParsedFeedbackEvidence(
            dissatisfaction_scope=["style"],
            preserve_constraints=["Keep the composition"],
            requested_changes=["make it more cinematic"],
            uncertainty_estimate=0.25,
            raw_feedback="Keep the composition, but improve style.",
        )
        session.dissatisfaction_axes = ["style"]
        session.preserve_constraints = ["Keep the composition"]
        session.current_uncertainty_estimate = 0.25
        session.selected_reference_ids = [101]
        return session

    def _make_candidates(self):
        return [
            RefinementBenchmarkCandidate(
                candidate_id="benchmark-candidate-101",
                reference_id=101,
                source_index=7,
                source_role="best",
                selection_rationale="best anchor",
                metadata={
                    "anchor_overlap": True,
                    "novelty_score": 0.0,
                    "coverage_score": 1.0,
                    "benchmark_source": "refinement_search_bundle",
                },
            ),
            RefinementBenchmarkCandidate(
                candidate_id="benchmark-candidate-102",
                reference_id=102,
                source_index=8,
                source_role="complementary_knn",
                selection_rationale="supporting variant",
                metadata={
                    "anchor_overlap": False,
                    "novelty_score": 1.0,
                    "coverage_score": 0.75,
                    "benchmark_source": "refinement_search_bundle",
                },
            ),
            RefinementBenchmarkCandidate(
                candidate_id="benchmark-candidate-103",
                reference_id=103,
                source_index=9,
                source_role="exploratory",
                selection_rationale="exploratory variant",
                metadata={
                    "anchor_overlap": False,
                    "novelty_score": 1.0,
                    "coverage_score": 0.5,
                    "benchmark_source": "refinement_search_bundle",
                },
            ),
        ]

    def test_ranking_prefers_exploratory_candidate_for_change_seeking_context(self):
        session = self._make_session()

        ranked = rank_benchmark_candidates(self._make_candidates(), session)

        self.assertEqual([candidate.candidate_id for candidate in ranked[:3]], [
            "benchmark-candidate-103",
            "benchmark-candidate-102",
            "benchmark-candidate-101",
        ])
        self.assertIn("pbo_score", ranked[0].metadata)
        self.assertIn("pbo_rationale", ranked[0].metadata)

    def test_ranking_prefers_anchor_preserving_candidate_when_preserve_pressure_increases(self):
        session = self._make_session()
        session.preserve_constraints = ["Keep the composition", "best"]
        session.current_uncertainty_estimate = 0.05

        ranked = rank_benchmark_candidates(self._make_candidates(), session)

        self.assertEqual(ranked[0].candidate_id, "benchmark-candidate-101")

    def test_ranking_is_stable_when_context_is_missing(self):
        session = AgentMemoryService().create_session("make a portrait")

        ranked = rank_benchmark_candidates(self._make_candidates(), session)

        self.assertEqual([candidate.candidate_id for candidate in ranked[:3]], [
            "benchmark-candidate-102",
            "benchmark-candidate-103",
            "benchmark-candidate-101",
        ])


if __name__ == "__main__":
    unittest.main()
