import unittest

from app.agent.benchmark_comparison_summary import BenchmarkComparisonSummary
from app.agent.runtime_models import CommittedPatch, PreviewProbe, ResultSummary
from app.agent.verifier import Verifier


class VerifierTest(unittest.TestCase):
    def test_verifier_returns_structured_decision(self):
        verifier = Verifier()
        previous = ResultSummary(summary_text="before", changed_axes=["initial_generation"])
        updated = ResultSummary(
            summary_text="after",
            changed_axes=["style"],
            preserved_axes=["composition"],
        )
        probe = PreviewProbe(
            probe_id="p_001",
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        patch = CommittedPatch(
            patch_id="cp_p_001",
            target_fields=["style", "model"],
            target_axes=["style"],
            preserve_axes=["composition"],
            rationale="adjust style",
        )

        result = verifier.verify(previous, updated, probe, patch, ["Keep the composition"])

        self.assertIsInstance(result.improved, bool)
        self.assertIsInstance(result.continue_recommended, bool)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.regression_notes, list)
        self.assertTrue(result.summary)

    def test_verifier_with_benchmark_summary_keeps_decision_shape_and_adds_context_note(self):
        verifier = Verifier()
        previous = ResultSummary(summary_text="before", changed_axes=["initial_generation"])
        updated = ResultSummary(
            summary_text="after",
            changed_axes=["style"],
            preserved_axes=["composition"],
        )
        probe = PreviewProbe(
            probe_id="p_001",
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        patch = CommittedPatch(
            patch_id="cp_p_001",
            target_fields=["style", "model"],
            target_axes=["style"],
            preserve_axes=["composition"],
            rationale="adjust style",
        )
        benchmark_summary = BenchmarkComparisonSummary(
            compared_anchor_ids=[101, 102],
            compared_candidate_ids=["benchmark-candidate-101", "benchmark-candidate-102"],
            metadata={"benchmark_source": "refinement_search_bundle"},
        )

        result = verifier.verify(
            previous,
            updated,
            probe,
            patch,
            ["Keep the composition"],
            benchmark_comparison_summary=benchmark_summary,
        )

        self.assertTrue(result.summary)
        self.assertIn("benchmark_context=refinement_search_bundle:2_candidates", result.summary)
        self.assertIn("benchmark_context=refinement_search_bundle:2_candidates", result.regression_notes)


if __name__ == "__main__":
    unittest.main()
