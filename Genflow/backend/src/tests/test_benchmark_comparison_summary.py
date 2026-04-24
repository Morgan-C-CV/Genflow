import unittest
from dataclasses import asdict

from app.agent.benchmark_comparison_summary import (
    BenchmarkComparisonSummary,
    build_benchmark_comparison_summary,
)
from app.agent.memory import AgentMemoryService
from app.agent.refinement_benchmark_retriever import (
    RefinementBenchmarkCandidate,
    RefinementBenchmarkSet,
)


class BenchmarkComparisonSummaryTest(unittest.TestCase):
    def _make_session_and_benchmark_set(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.current_result_id = "result-1"
        session.dissatisfaction_axes = ["style"]
        session.preserve_constraints = ["Keep the composition"]
        benchmark_set = RefinementBenchmarkSet(
            benchmark_id=f"refinement-benchmark-{session.session_id}",
            benchmark_kind="refinement_local_comparison",
            benchmark_source="refinement_search_bundle",
            anchor_ids=[101, 102],
            anchor_summary="Selected gallery anchor bundle with 2 references.",
            comparison_candidates=[
                RefinementBenchmarkCandidate(
                    candidate_id="benchmark-candidate-101",
                    reference_id=101,
                    source_index=7,
                    source_role="best",
                    selection_rationale="selected role=best ref=101 for focus=style while preserving=Keep the composition",
                    metadata={"anchor_overlap": True},
                ),
                RefinementBenchmarkCandidate(
                    candidate_id="benchmark-candidate-102",
                    reference_id=102,
                    source_index=8,
                    source_role="complementary_knn",
                    selection_rationale="selected role=complementary_knn ref=102 for focus=style while preserving=Keep the composition",
                    metadata={"anchor_overlap": True},
                ),
            ],
            selection_rationale=["focus_axes=style", "preserve=Keep the composition"],
        )
        return session, benchmark_set

    def test_summary_defaults_are_safe(self):
        summary = BenchmarkComparisonSummary()

        self.assertEqual(summary.compared_anchor_ids, [])
        self.assertEqual(summary.compared_candidate_ids, [])
        self.assertEqual(summary.summary_bullets, [])
        self.assertEqual(summary.comparison_items, [])
        self.assertEqual(summary.metadata, {})

    def test_builder_generates_stable_summary(self):
        session, benchmark_set = self._make_session_and_benchmark_set()

        first = build_benchmark_comparison_summary(benchmark_set, session)
        second = build_benchmark_comparison_summary(benchmark_set, session)

        self.assertEqual(first, second)
        self.assertEqual(first.compared_anchor_ids, [101, 102])
        self.assertEqual(first.compared_candidate_ids, ["benchmark-candidate-101", "benchmark-candidate-102"])
        self.assertEqual(first.focus_axes, ["style"])
        self.assertEqual(first.preserve_axes, ["Keep the composition"])
        self.assertIn("benchmark_source=refinement_search_bundle", first.summary_bullets)
        self.assertGreater(first.confidence_hint, 0)

    def test_summary_is_serializable_and_distinct_from_cold_start_bundle(self):
        session, benchmark_set = self._make_session_and_benchmark_set()
        summary = build_benchmark_comparison_summary(benchmark_set, session)
        serialized = asdict(summary)

        self.assertNotIn("references", serialized)
        self.assertIn("comparison_items", serialized)
        self.assertEqual(serialized["metadata"]["benchmark_kind"], "refinement_local_comparison")


if __name__ == "__main__":
    unittest.main()
