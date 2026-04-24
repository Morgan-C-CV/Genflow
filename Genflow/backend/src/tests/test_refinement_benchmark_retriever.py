import unittest
from dataclasses import asdict

from app.agent.memory import AgentMemoryService
from app.agent.refinement_benchmark_retriever import (
    RefinementBenchmarkSet,
    build_benchmark_candidate_pool,
    retrieve_refinement_benchmark_set,
)
from app.agent.runtime_models import ParsedFeedbackEvidence
from tests.test_runtime_service import FakeSearchService


class RefinementBenchmarkRetrieverTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.selected_gallery_index = 7
        session.selected_reference_bundle = {
            "query_index": 7,
            "counts": {"best": 1},
            "references": [
                {"id": 101, "index": 7, "role": "best"},
                {"id": 102, "index": 8, "role": "complementary_knn"},
            ],
        }
        session.selected_reference_ids = [101, 102]
        session.current_gallery_anchor_summary = "Selected gallery anchor bundle with 2 references."
        session.current_schema.prompt = "a cinematic portrait"
        session.current_schema.model = "sdxl-base"
        session.current_result_id = "result-1"
        session.current_result_summary.summary_text = "Current result is compositionally strong but stylistically flat."
        session.latest_feedback = "Keep the composition, but improve style."
        session.parsed_feedback = ParsedFeedbackEvidence(
            dissatisfaction_scope=["style"],
            preserve_constraints=["Keep the composition"],
            requested_changes=["increase cinematic mood"],
            uncertainty_estimate=0.25,
            raw_feedback=session.latest_feedback,
        )
        session.dissatisfaction_axes = ["style"]
        session.preserve_constraints = ["Keep the composition"]
        session.current_uncertainty_estimate = 0.25
        return session

    def test_benchmark_set_defaults_are_safe(self):
        benchmark_set = RefinementBenchmarkSet()

        self.assertEqual(benchmark_set.benchmark_id, "")
        self.assertEqual(benchmark_set.anchor_ids, [])
        self.assertEqual(benchmark_set.comparison_candidates, [])
        self.assertEqual(benchmark_set.selection_rationale, [])
        self.assertEqual(benchmark_set.metadata, {})

    def test_retriever_builds_small_deterministic_local_benchmark_set(self):
        session = self._make_session()
        search_service = FakeSearchService()

        first = retrieve_refinement_benchmark_set(session, search_service=search_service, limit=2)
        second = retrieve_refinement_benchmark_set(session, search_service=search_service, limit=2)

        self.assertEqual(first, second)
        self.assertEqual(first.benchmark_kind, "refinement_local_comparison")
        self.assertEqual(first.benchmark_source, "refinement_search_bundle")
        self.assertEqual(first.anchor_ids, [101, 102])
        self.assertEqual(first.anchor_summary, session.current_gallery_anchor_summary)
        self.assertEqual(len(first.comparison_candidates), 2)
        self.assertIn("focus_axes=style", first.selection_rationale)
        self.assertIn("preserve=Keep the composition", first.selection_rationale)
        self.assertEqual(first.comparison_candidates[0].reference_id, 103)
        self.assertIn("pbo_score", first.comparison_candidates[0].metadata)

    def test_candidate_pool_build_and_reranked_selection_are_distinct_steps(self):
        session = self._make_session()
        search_service = FakeSearchService()

        benchmark_source, candidate_pool = build_benchmark_candidate_pool(session, search_service=search_service)
        benchmark_set = retrieve_refinement_benchmark_set(session, search_service=search_service, limit=2)

        self.assertEqual(benchmark_source, "refinement_search_bundle")
        self.assertEqual(len(candidate_pool), 3)
        self.assertEqual(candidate_pool[0].candidate_id, "benchmark-candidate-101")
        self.assertNotEqual(
            [candidate.candidate_id for candidate in benchmark_set.comparison_candidates],
            [candidate.candidate_id for candidate in candidate_pool[:2]],
        )
        self.assertEqual(
            [candidate.candidate_id for candidate in benchmark_set.comparison_candidates],
            ["benchmark-candidate-103", "benchmark-candidate-102"],
        )

    def test_retriever_reranking_changes_with_refinement_context(self):
        session = self._make_session()
        search_service = FakeSearchService()

        exploratory_first = retrieve_refinement_benchmark_set(session, search_service=search_service, limit=3)
        self.assertEqual(exploratory_first.comparison_candidates[0].candidate_id, "benchmark-candidate-103")

        session.preserve_constraints = ["Keep the composition", "best"]
        session.current_uncertainty_estimate = 0.05

        preserve_first = retrieve_refinement_benchmark_set(session, search_service=search_service, limit=3)

        self.assertEqual(preserve_first.comparison_candidates[0].candidate_id, "benchmark-candidate-101")

    def test_benchmark_set_is_independent_from_cold_start_bundle_and_serializable(self):
        session = self._make_session()
        search_service = FakeSearchService()

        benchmark_set = retrieve_refinement_benchmark_set(session, search_service=search_service, limit=3)
        session.refinement_benchmark_set = benchmark_set
        session.refinement_benchmark_summary = "; ".join(benchmark_set.selection_rationale)
        serialized = asdict(benchmark_set)

        self.assertIsNot(benchmark_set, session.selected_reference_bundle)
        self.assertNotIn("references", serialized)
        self.assertIn("comparison_candidates", serialized)
        self.assertEqual(serialized["metadata"]["feedback_present"], True)
        self.assertEqual(session.refinement_benchmark_set.benchmark_id, benchmark_set.benchmark_id)
        self.assertTrue(session.refinement_benchmark_summary)


if __name__ == "__main__":
    unittest.main()
