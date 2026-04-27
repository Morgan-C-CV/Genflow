import unittest

from app.agent.memory import AgentMemoryService
from app.agent.orchestration_policy import decide_next_action
from app.agent.runtime_models import (
    ExecutionSourceEvidenceSummary,
    PreviewProbe,
    ResultSummary,
    VerifierResult,
    VerifierSignalSummary,
)
from app.agent.verifier_repair_recommendation import VerifierRepairRecommendation


class OrchestrationPolicyTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.current_result_summary = ResultSummary(summary_text="current result exists")
        return session

    def _prepare_verified_session(self):
        session = self._make_session()
        session.repair_hypotheses = [object()]
        session.refinement_benchmark_set.comparison_candidates = [object()]
        session.preview_probe_candidates = [PreviewProbe(probe_id="p_001")]
        session.selected_probe = PreviewProbe(probe_id="p_001", target_axes=["style"], preserve_axes=["composition"])
        session.preview_probe_results = [type("PreviewResultStub", (), {"probe_id": "p_001"})()]
        session.accepted_patch.patch_id = "cp_001"
        session.accepted_patch.target_axes = ["style"]
        session.previous_result_summary = ResultSummary(summary_text="pre-patch result")
        return session

    def test_policy_requests_hypotheses_when_missing(self):
        session = self._make_session()

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "build_hypotheses")

    def test_policy_requests_benchmarks_when_uncertainty_is_high_and_benchmarks_missing(self):
        session = self._make_session()
        session.repair_hypotheses = [object()]
        session.current_uncertainty_estimate = 0.6

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "retrieve_benchmarks")

    def test_policy_requests_probe_generation_before_commit(self):
        session = self._make_session()
        session.repair_hypotheses = [object()]
        session.current_uncertainty_estimate = 0.2
        session.refinement_benchmark_set.comparison_candidates = [object()]

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "generate_probes")

    def test_policy_requests_preview_for_selected_probe_without_preview_result(self):
        session = self._make_session()
        session.repair_hypotheses = [object()]
        session.refinement_benchmark_set.comparison_candidates = [object()]
        session.preview_probe_candidates = [PreviewProbe(probe_id="p_001")]
        session.selected_probe = PreviewProbe(probe_id="p_001", target_axes=["style"])

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "preview_selected_probe")

    def test_policy_requests_execute_after_patch_commit_before_execution(self):
        session = self._make_session()
        session.repair_hypotheses = [object()]
        session.refinement_benchmark_set.comparison_candidates = [object()]
        session.preview_probe_candidates = [PreviewProbe(probe_id="p_001")]
        session.selected_probe = PreviewProbe(probe_id="p_001")
        session.preview_probe_results = [type("PreviewResultStub", (), {"probe_id": "p_001"})()]
        session.accepted_patch.patch_id = "cp_001"

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "execute_patch")

    def test_policy_requests_verify_after_patch_execution(self):
        session = self._make_session()
        session.repair_hypotheses = [object()]
        session.refinement_benchmark_set.comparison_candidates = [object()]
        session.preview_probe_candidates = [PreviewProbe(probe_id="p_001")]
        session.selected_probe = PreviewProbe(probe_id="p_001")
        session.preview_probe_results = [type("PreviewResultStub", (), {"probe_id": "p_001"})()]
        session.accepted_patch.patch_id = "cp_001"
        session.previous_result_summary = ResultSummary(summary_text="pre-patch result")

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "verify_latest_result")

    def test_policy_stops_when_verifier_recommends_stop(self):
        session = self._prepare_verified_session()
        session.latest_verifier_result = VerifierResult(
            improved=True,
            continue_recommended=False,
            confidence=0.9,
            summary="verifier accepts current direction",
            signal_summary=VerifierSignalSummary(
                target_alignment_score=2.0,
                preserve_risk_score=0.2,
                benchmark_support_score=1.2,
                execution_evidence_score=2.0,
                total_score=5.0,
            ),
        )
        session.latest_verifier_signal_summary = session.latest_verifier_result.signal_summary
        session.latest_verifier_repair_recommendation = VerifierRepairRecommendation(
            recommended_action="stop",
            rationale=["verifier accepts the current direction"],
            priority="low",
            supporting_signals=["strong_benchmark_support"],
        )
        session.continue_recommended = False
        session.stop_reason = "verifier_accepts_current_direction"

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "stop")
        self.assertFalse(decision.continue_loop)
        self.assertIn("verifier_repair_recommendation=stop", decision.rationale)

    def test_policy_uses_high_preserve_risk_to_avoid_premature_stop(self):
        session = self._prepare_verified_session()
        session.latest_verifier_result = VerifierResult(
            improved=False,
            continue_recommended=False,
            confidence=0.55,
            summary="verifier is cautious",
            signal_summary=VerifierSignalSummary(
                target_alignment_score=1.2,
                preserve_risk_score=2.4,
                benchmark_support_score=1.1,
                execution_evidence_score=1.4,
                total_score=1.3,
            ),
        )
        session.latest_verifier_signal_summary = session.latest_verifier_result.signal_summary
        session.latest_verifier_repair_recommendation = VerifierRepairRecommendation(
            recommended_action="reduce_preserve_risk",
            rationale=["preserve risk is materially elevated"],
            priority="high",
            supporting_signals=["high_preserve_risk"],
        )
        session.continue_recommended = False
        session.stop_reason = "verifier_accepts_current_direction"

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "generate_probes")
        self.assertIn("verifier_repair_recommendation=reduce_preserve_risk", decision.rationale)

    def test_policy_uses_weak_execution_evidence_to_retrieve_benchmarks(self):
        session = self._prepare_verified_session()
        session.current_uncertainty_estimate = 0.6
        session.latest_verifier_result = VerifierResult(
            improved=False,
            continue_recommended=True,
            confidence=0.4,
            summary="verifier wants more evidence",
            signal_summary=VerifierSignalSummary(
                target_alignment_score=1.0,
                preserve_risk_score=0.4,
                benchmark_support_score=0.2,
                execution_evidence_score=0.0,
                total_score=0.8,
            ),
        )
        session.latest_verifier_signal_summary = session.latest_verifier_result.signal_summary
        session.latest_verifier_repair_recommendation = VerifierRepairRecommendation(
            recommended_action="refresh_benchmarks",
            rationale=["benchmark support is weak under high uncertainty"],
            priority="high",
            supporting_signals=["weak_execution_evidence", "low_benchmark_support"],
        )
        session.continue_recommended = True

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "retrieve_benchmarks")
        self.assertIn("verifier_repair_recommendation=refresh_benchmarks", decision.rationale)

    def test_policy_falls_back_to_signal_based_logic_when_recommendation_is_missing(self):
        session = self._prepare_verified_session()
        session.current_uncertainty_estimate = 0.6
        session.latest_verifier_result = VerifierResult(
            improved=False,
            continue_recommended=True,
            confidence=0.4,
            summary="verifier wants more evidence",
            signal_summary=VerifierSignalSummary(
                target_alignment_score=1.0,
                preserve_risk_score=0.4,
                benchmark_support_score=0.2,
                execution_evidence_score=0.0,
                total_score=0.8,
            ),
        )
        session.latest_verifier_signal_summary = session.latest_verifier_result.signal_summary
        session.continue_recommended = True

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "retrieve_benchmarks")
        self.assertIn("weak_execution_evidence", decision.rationale)
        self.assertIn("low_benchmark_support", decision.rationale)

    def test_policy_uses_retry_graph_native_execution_hint_before_verifier(self):
        session = self._prepare_verified_session()
        session.latest_execution_source_evidence = ExecutionSourceEvidenceSummary(
            backend_graph_native_remediation_hint="retry_graph_native_execution",
            backend_graph_native_realization_reason="graph_native_realization_achieved",
            backend_graph_native_execution_realized=True,
        )

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "verify_latest_result")
        self.assertIn("backend_graph_native_remediation_hint=retry_graph_native_execution", decision.rationale)

    def test_policy_uses_enrich_graph_payload_hint_to_generate_probes(self):
        session = self._prepare_verified_session()
        session.latest_execution_source_evidence = ExecutionSourceEvidenceSummary(
            backend_graph_native_remediation_hint="enrich_graph_payload",
            backend_graph_native_realization_reason="insufficient_graph_payload_completeness",
        )

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "generate_probes")
        self.assertIn("backend_graph_native_remediation_hint=enrich_graph_payload", decision.rationale)

    def test_policy_uses_restore_preserve_alignment_hint_to_rebuild_hypotheses(self):
        session = self._prepare_verified_session()
        session.latest_execution_source_evidence = ExecutionSourceEvidenceSummary(
            backend_graph_native_remediation_hint="restore_preserve_alignment",
            backend_graph_native_realization_reason="preserve_safety_downgrade",
        )

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "build_hypotheses")
        self.assertIn("backend_graph_native_remediation_hint=restore_preserve_alignment", decision.rationale)

    def test_policy_uses_fallback_schema_execution_hint_to_verify_current_result(self):
        session = self._prepare_verified_session()
        session.latest_execution_source_evidence = ExecutionSourceEvidenceSummary(
            backend_graph_native_remediation_hint="fallback_schema_execution",
            backend_graph_native_realization_reason="unsupported_backend_capability",
        )

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "verify_latest_result")
        self.assertIn("backend_graph_native_remediation_hint=fallback_schema_execution", decision.rationale)


if __name__ == "__main__":
    unittest.main()
