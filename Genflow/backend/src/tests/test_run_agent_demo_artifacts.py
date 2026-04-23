import unittest

from app.agent.memory import AgentMemoryService
from app.agent.result_executor import ResultExecutor
from app.agent.runtime_service import AgentRuntimeService
from run_agent_demo import build_session_artifact_payload
from tests.test_runtime_service import (
    FakeFeedbackParser,
    FakeHypothesisBuilder,
    FakeOrchestrationService,
    FakeSearchService,
)


class RunAgentDemoArtifactTest(unittest.TestCase):
    def test_session_artifact_payload_contains_phase_c_state(self):
        memory = AgentMemoryService()
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=FakeOrchestrationService(memory),
            search_service=FakeSearchService(),
            result_executor=ResultExecutor(id_factory=lambda: "artifact-result-1"),
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)

        payload = build_session_artifact_payload(session)

        self.assertIn("parsed_feedback", payload)
        self.assertIn("preserve_constraints", payload)
        self.assertIn("dissatisfaction_axes", payload)
        self.assertIn("requested_changes", payload)
        self.assertIn("current_uncertainty_estimate", payload)
        self.assertIn("repair_hypotheses", payload)
        self.assertEqual(payload["parsed_feedback"]["raw_feedback"], "Keep the composition, but improve style.")
        self.assertEqual(payload["preserve_constraints"], ["Keep the composition"])
        self.assertEqual(payload["dissatisfaction_axes"], ["style"])
        self.assertEqual(payload["requested_changes"], ["make the style brighter"])
        self.assertEqual(payload["current_uncertainty_estimate"], 0.25)
        self.assertEqual(len(payload["repair_hypotheses"]), 2)


if __name__ == "__main__":
    unittest.main()
