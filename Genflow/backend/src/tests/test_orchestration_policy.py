import unittest

from app.agent.memory import AgentMemoryService
from app.agent.orchestration_policy import decide_next_action
from app.agent.runtime_models import PreviewProbe, ResultSummary, VerifierResult


class OrchestrationPolicyTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.current_result_summary = ResultSummary(summary_text="current result exists")
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
        session = self._make_session()
        session.latest_verifier_result = VerifierResult(
            improved=True,
            continue_recommended=False,
            confidence=0.9,
            summary="verifier accepts current direction",
        )
        session.continue_recommended = False
        session.stop_reason = "verifier_accepts_current_direction"

        decision = decide_next_action(session)

        self.assertEqual(decision.next_action, "stop")
        self.assertFalse(decision.continue_loop)


if __name__ == "__main__":
    unittest.main()
