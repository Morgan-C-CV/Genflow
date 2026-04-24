import unittest

from app.agent.memory import AgentMemoryService
from app.agent.result_executor import ResultExecutor
from app.agent.runtime_service import AgentRuntimeService
from tests.test_runtime_service import (
    FakeFeedbackParser,
    FakeHypothesisBuilder,
    FakeOrchestrationService,
    FakePatchPlanner,
    FakeProbeGenerator,
    FakeSearchService,
    FakeVerifier,
)


class RuntimeServicePolicyRunnerTest(unittest.TestCase):
    def _make_service(self, *, id_factory):
        memory = AgentMemoryService()
        return AgentRuntimeService(
            memory_service=memory,
            orchestration_service=FakeOrchestrationService(memory),
            search_service=FakeSearchService(),
            execution_adapter=ResultExecutor(id_factory=id_factory),
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            probe_generator=FakeProbeGenerator(),
            patch_planner=FakePatchPlanner(),
            verifier=FakeVerifier(),
        )

    def _prepare_base_repair_session(self, service: AgentRuntimeService):
        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        return session

    def test_run_next_policy_step_builds_hypotheses_when_missing(self):
        service = self._make_service(id_factory=lambda: "policy-runner-initial")
        session = self._prepare_base_repair_session(service)

        result = service.run_next_policy_step(session.session_id)

        self.assertEqual(result.decision.next_action, "build_hypotheses")
        self.assertTrue(result.updated_session.repair_hypotheses)
        self.assertTrue(result.updated_session.refinement_benchmark_set.comparison_candidates)

    def test_run_next_policy_step_generates_probes(self):
        service = self._make_service(id_factory=lambda: "policy-runner-probes")
        session = self._prepare_base_repair_session(service)
        session = service.build_repair_hypotheses(session.session_id)

        result = service.run_next_policy_step(session.session_id)

        self.assertEqual(result.decision.next_action, "generate_probes")
        self.assertTrue(result.updated_session.preview_probe_candidates)
        self.assertTrue(result.updated_session.selected_probe.probe_id)

    def test_run_next_policy_step_previews_selected_probe(self):
        service = self._make_service(id_factory=lambda: "policy-runner-preview")
        session = self._prepare_base_repair_session(service)
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)

        result = service.run_next_policy_step(session.session_id)

        self.assertEqual(result.decision.next_action, "preview_selected_probe")
        self.assertTrue(result.updated_session.preview_probe_results)
        self.assertEqual(
            result.updated_session.preview_probe_results[-1].probe_id,
            result.updated_session.selected_probe.probe_id,
        )

    def test_run_next_policy_step_executes_patch_when_committed_not_executed(self):
        ids = iter(["policy-runner-commit-initial", "policy-runner-commit-result", "policy-runner-commit-result-2"])
        service = self._make_service(id_factory=lambda: next(ids))
        session = self._prepare_base_repair_session(service)
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.preview_selected_probe(session.session_id)
        session = service.commit_patch(session.session_id)

        result = service.run_next_policy_step(session.session_id)

        self.assertEqual(result.decision.next_action, "execute_patch")
        self.assertTrue(result.updated_session.previous_result_summary.summary_text)
        self.assertEqual(result.updated_session.current_result_payload.result_type, "mock_committed_result")

    def test_run_next_policy_step_stops_without_mutation(self):
        ids = iter(["policy-runner-stop-initial", "policy-runner-stop-result", "policy-runner-stop-result-2"])
        service = self._make_service(id_factory=lambda: next(ids))
        session = self._prepare_base_repair_session(service)
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.preview_selected_probe(session.session_id)
        session = service.commit_patch(session.session_id)
        session = service.execute_patch(session.session_id)
        session = service.verify_latest_result(session.session_id)
        before_result_id = session.current_result_id

        result = service.run_next_policy_step(session.session_id)

        self.assertEqual(result.decision.next_action, "stop")
        self.assertEqual(result.updated_session.current_result_id, before_result_id)


if __name__ == "__main__":
    unittest.main()
