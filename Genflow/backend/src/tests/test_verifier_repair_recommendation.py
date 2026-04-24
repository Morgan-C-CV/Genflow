import unittest

from app.agent.runtime_models import VerifierResult, VerifierSignalSummary
from app.agent.verifier_repair_recommendation import (
    VerifierRepairRecommendation,
    build_verifier_repair_recommendation,
)


class VerifierRepairRecommendationTest(unittest.TestCase):
    def test_defaults_are_safe(self):
        recommendation = VerifierRepairRecommendation()

        self.assertEqual(recommendation.recommended_action, "")
        self.assertEqual(recommendation.rationale, [])
        self.assertEqual(recommendation.priority, "")
        self.assertEqual(recommendation.supporting_signals, [])

    def test_high_preserve_risk_recommends_reduce_preserve_risk(self):
        recommendation = build_verifier_repair_recommendation(
            verifier_signal_summary=VerifierSignalSummary(
                preserve_risk_score=2.5,
                execution_evidence_score=1.4,
                benchmark_support_score=0.8,
            ),
            verifier_result=VerifierResult(improved=False, continue_recommended=True),
        )

        self.assertEqual(recommendation.recommended_action, "reduce_preserve_risk")
        self.assertEqual(recommendation.priority, "high")
        self.assertIn("high_preserve_risk", recommendation.supporting_signals)

    def test_weak_execution_evidence_and_high_uncertainty_recommends_refresh_benchmarks(self):
        session = type("SessionStub", (), {"current_uncertainty_estimate": 0.6})()
        recommendation = build_verifier_repair_recommendation(
            verifier_signal_summary=VerifierSignalSummary(
                preserve_risk_score=0.2,
                execution_evidence_score=0.0,
                benchmark_support_score=0.2,
            ),
            verifier_result=VerifierResult(improved=False, continue_recommended=True),
            session=session,
        )

        self.assertEqual(recommendation.recommended_action, "refresh_benchmarks")
        self.assertIn("low_benchmark_support", recommendation.supporting_signals)

    def test_strong_supported_acceptance_recommends_stop(self):
        recommendation = build_verifier_repair_recommendation(
            verifier_signal_summary=VerifierSignalSummary(
                preserve_risk_score=0.1,
                execution_evidence_score=2.0,
                benchmark_support_score=1.2,
            ),
            verifier_result=VerifierResult(improved=True, continue_recommended=False),
        )

        self.assertEqual(recommendation.recommended_action, "stop")
        self.assertIn("strong_benchmark_support", recommendation.supporting_signals)


if __name__ == "__main__":
    unittest.main()
