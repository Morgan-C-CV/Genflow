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
        self.assertIn("signals[", result.summary)
        self.assertGreater(result.signal_summary.target_alignment_score, 0.0)
        self.assertGreater(result.signal_summary.execution_evidence_score, 0.0)

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
        self.assertIn("benchmark_support=", result.summary)
        self.assertIn("benchmark_context=refinement_search_bundle:2_candidates", result.regression_notes)

    def test_verifier_becomes_more_conservative_when_preserve_risk_is_high(self):
        verifier = Verifier()
        previous = ResultSummary(summary_text="before", changed_axes=["initial_generation"])
        updated = ResultSummary(
            summary_text="after",
            changed_axes=["style"],
            preserved_axes=[],
        )
        probe = PreviewProbe(
            probe_id="p_001",
            target_axes=["style"],
            preserve_axes=["style"],
        )
        patch = CommittedPatch(
            patch_id="cp_p_001",
            target_fields=["style"],
            target_axes=["style"],
            preserve_axes=["style"],
            rationale="aggressive style rewrite",
        )

        result = verifier.verify(previous, updated, probe, patch, ["Keep the style"])

        self.assertFalse(result.improved)
        self.assertTrue(result.continue_recommended)
        self.assertIn("preserve overlap risk=style", result.regression_notes)

    def test_verifier_benchmark_support_can_raise_confidence_without_changing_contract(self):
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
            target_fields=["style"],
            target_axes=["style"],
            preserve_axes=["composition"],
            rationale="style patch",
        )
        with_benchmark = verifier.verify(
            previous,
            updated,
            probe,
            patch,
            ["Keep the composition"],
            benchmark_comparison_summary=BenchmarkComparisonSummary(
                compared_anchor_ids=[101, 102],
                compared_candidate_ids=["benchmark-candidate-101", "benchmark-candidate-102"],
                focus_axes=["style"],
                preserve_axes=["composition"],
                metadata={"benchmark_source": "refinement_search_bundle"},
            ),
        )
        without_benchmark = verifier.verify(
            previous,
            updated,
            probe,
            patch,
            ["Keep the composition"],
            benchmark_comparison_summary=None,
        )

        self.assertGreaterEqual(with_benchmark.confidence, without_benchmark.confidence)
        self.assertIn("benchmark_context=refinement_search_bundle:2_candidates", with_benchmark.regression_notes)
        self.assertGreater(with_benchmark.signal_summary.benchmark_support_score, 0.0)

    def test_verifier_does_not_report_improvement_when_execution_change_evidence_is_weak(self):
        verifier = Verifier()
        previous = ResultSummary(summary_text="before", changed_axes=["initial_generation"])
        updated = ResultSummary(
            summary_text="before",
            changed_axes=["color_palette"],
            preserved_axes=["composition"],
        )
        probe = PreviewProbe(
            probe_id="p_001",
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        patch = CommittedPatch(
            patch_id="cp_p_001",
            target_fields=["style"],
            target_axes=["style"],
            preserve_axes=["composition"],
            rationale="style patch",
        )

        result = verifier.verify(previous, updated, probe, patch, ["Keep the composition"])

        self.assertFalse(result.improved)
        self.assertTrue(result.continue_recommended)
        self.assertIn("updated result summary did not change relative to previous result", result.regression_notes)
        self.assertIn("execution evidence did not confirm committed target axes", result.regression_notes)
        self.assertEqual(result.signal_summary.execution_evidence_score, 0.0)


if __name__ == "__main__":
    unittest.main()
