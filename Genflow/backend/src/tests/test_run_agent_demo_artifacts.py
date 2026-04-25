import unittest

from app.agent.memory import AgentMemoryService
from app.agent.result_executor import ResultExecutor
from app.agent.runtime_service import AgentRuntimeService
from run_agent_demo import build_session_artifact_payload
from tests.test_runtime_service import (
    FakeFeedbackParser,
    FakeHypothesisBuilder,
    FakeOrchestrationService,
    FakePatchPlanner,
    FakeSearchService,
    FakeVerifier,
    FakeProbeGenerator,
)


class RunAgentDemoArtifactTest(unittest.TestCase):
    def test_session_artifact_payload_contains_phase_c_state(self):
        memory = AgentMemoryService()
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=FakeOrchestrationService(memory),
            search_service=FakeSearchService(),
            execution_adapter=ResultExecutor(id_factory=lambda: "artifact-result-1"),
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
        self.assertIn("refinement_benchmark_set", payload)
        self.assertIn("refinement_benchmark_summary", payload)
        self.assertIn("benchmark_comparison_summary", payload)
        self.assertIn("workflow_graph_patch_candidates", payload)
        self.assertEqual(payload["parsed_feedback"]["raw_feedback"], "Keep the composition, but improve style.")
        self.assertEqual(payload["preserve_constraints"], ["Keep the composition"])
        self.assertEqual(payload["dissatisfaction_axes"], ["style"])
        self.assertEqual(payload["requested_changes"], ["make the style brighter"])
        self.assertEqual(payload["current_uncertainty_estimate"], 0.25)
        self.assertEqual(len(payload["repair_hypotheses"]), 2)
        self.assertEqual(payload["refinement_benchmark_set"]["benchmark_kind"], "refinement_local_comparison")
        self.assertTrue(payload["refinement_benchmark_summary"])
        self.assertEqual(
            payload["benchmark_comparison_summary"]["metadata"]["benchmark_kind"],
            "refinement_local_comparison",
        )

    def test_session_artifact_payload_contains_verifier_signal_summary(self):
        memory = AgentMemoryService()
        ids = iter(["artifact-initial-1", "artifact-commit-1", "artifact-commit-2"])
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=FakeOrchestrationService(memory),
            search_service=FakeSearchService(),
            execution_adapter=ResultExecutor(id_factory=lambda: next(ids)),
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            probe_generator=FakeProbeGenerator(),
            patch_planner=FakePatchPlanner(),
            verifier=FakeVerifier(),
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.preview_selected_probe(session.session_id)
        session = service.commit_patch(session.session_id)
        session = service.execute_patch(session.session_id)
        session = service.verify_latest_result(session.session_id)

        payload = build_session_artifact_payload(session)

        self.assertIn("latest_verifier_signal_summary", payload)
        self.assertEqual(
            payload["latest_verifier_signal_summary"]["total_score"],
            payload["latest_verifier_result"]["signal_summary"]["total_score"],
        )
        self.assertEqual(
            payload["workflow_metadata"]["verifier_signal_summary"]["total_score"],
            payload["latest_verifier_signal_summary"]["total_score"],
        )
        self.assertIn("latest_verifier_repair_recommendation", payload)
        self.assertEqual(
            payload["latest_verifier_repair_recommendation"]["recommended_action"],
            payload["workflow_metadata"]["verifier_repair_recommendation"]["recommended_action"],
        )
        self.assertIn("current_workflow_graph_patch", payload)
        self.assertIn("selected_workflow_graph_patch", payload)
        self.assertIn("top_schema_patch_candidate", payload)
        self.assertIn("top_workflow_graph_patch_candidate", payload)
        self.assertIn("preferred_commit_source", payload)
        self.assertIn("selected_graph_native_patch_candidate", payload)
        self.assertEqual(
            payload["current_workflow_graph_patch"]["patch_id"],
            payload["accepted_patch"]["patch_id"],
        )
        self.assertEqual(
            payload["workflow_metadata"]["workflow_graph_patch"]["patch_id"],
            payload["current_workflow_graph_patch"]["patch_id"],
        )
        self.assertGreater(len(payload["workflow_graph_patch_candidates"]), 1)
        self.assertEqual(
            payload["workflow_metadata"]["workflow_graph_patch_candidates"]["candidate_count"],
            len(payload["workflow_graph_patch_candidates"]),
        )
        self.assertEqual(
            payload["workflow_metadata"]["workflow_graph_patch_candidates"]["top_candidate_id"],
            payload["workflow_graph_patch_candidates"][0]["candidate_id"],
        )
        self.assertIn("pbo_score", payload["workflow_graph_patch_candidates"][0]["metadata"])
        self.assertEqual(
            payload["top_schema_patch_candidate"]["patch_id"],
            payload["accepted_patch"]["patch_id"],
        )
        self.assertEqual(
            payload["top_workflow_graph_patch_candidate"]["candidate_id"],
            payload["workflow_graph_patch_candidates"][0]["candidate_id"],
        )
        self.assertEqual(
            payload["workflow_metadata"]["patch_winner_comparison"]["top_schema_patch_id"],
            payload["top_schema_patch_candidate"]["patch_id"],
        )
        self.assertEqual(
            payload["workflow_metadata"]["patch_winner_comparison"]["top_graph_patch_candidate_id"],
            payload["top_workflow_graph_patch_candidate"]["candidate_id"],
        )
        self.assertEqual(payload["preferred_commit_source"], "graph")
        self.assertEqual(
            payload["selected_graph_native_patch_candidate"]["candidate_id"],
            payload["top_workflow_graph_patch_candidate"]["candidate_id"],
        )
        self.assertEqual(
            payload["selected_workflow_graph_patch"]["metadata"]["candidate_id"],
            payload["selected_graph_native_patch_candidate"]["candidate_id"],
        )
        self.assertEqual(
            payload["workflow_metadata"]["patch_winner_comparison"]["preferred_commit_source"],
            payload["preferred_commit_source"],
        )
        self.assertEqual(payload["commit_execution_mode"], "graph_native_execution_handoff")
        self.assertEqual(payload["commit_execution_authority"], "graph_authoritative")
        self.assertEqual(payload["commit_execution_implementation_mode"], "graph_primary_execution")
        self.assertTrue(payload["request_graph_native_realization"])
        self.assertEqual(payload["request_backend_execution_mode"], "graph_primary_backend_execution")
        self.assertTrue(payload["backend_graph_primary_capable"])
        self.assertTrue(payload["backend_graph_native_realization_supported"])
        self.assertTrue(payload["backend_graph_commit_payload_supplied"])
        self.assertTrue(payload["backend_graph_commit_payload_consumed"])
        self.assertTrue(payload["backend_graph_native_execution_realized"])
        self.assertEqual(
            payload["backend_graph_native_realization_reason"],
            "graph_native_realization_achieved",
        )
        self.assertEqual(payload["request_primary_plan_kind"], "graph_primary")
        self.assertEqual(
            payload["workflow_metadata"]["patch_winner_comparison"]["selected_workflow_graph_patch_id"],
            payload["selected_workflow_graph_patch"]["patch_id"],
        )
        self.assertIn("latest_execution_source_evidence", payload)
        self.assertEqual(
            payload["latest_execution_source_evidence"]["commit_execution_mode"],
            payload["commit_execution_mode"],
        )
        self.assertTrue(
            payload["latest_execution_source_evidence"]["request_graph_native_artifact_input_received"]
        )
        self.assertTrue(
            payload["latest_execution_source_evidence"]["backend_echoed_graph_native_artifact_input_received"]
        )
        self.assertEqual(
            payload["latest_execution_source_evidence"]["commit_execution_authority"],
            "graph_authoritative",
        )
        self.assertEqual(
            payload["latest_execution_source_evidence"]["commit_execution_implementation_mode"],
            "graph_primary_execution",
        )
        self.assertTrue(
            payload["latest_execution_source_evidence"]["request_graph_native_realization"],
        )
        self.assertEqual(
            payload["latest_execution_source_evidence"]["request_backend_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertTrue(
            payload["latest_execution_source_evidence"]["backend_graph_primary_capable"],
        )
        self.assertTrue(
            payload["latest_execution_source_evidence"]["backend_graph_native_realization_supported"],
        )
        self.assertTrue(
            payload["latest_execution_source_evidence"]["backend_graph_commit_payload_supplied"],
        )
        self.assertTrue(
            payload["latest_execution_source_evidence"]["backend_graph_commit_payload_consumed"],
        )
        self.assertTrue(
            payload["latest_execution_source_evidence"]["backend_graph_native_execution_realized"],
        )
        self.assertEqual(
            payload["latest_execution_source_evidence"]["backend_graph_native_realization_reason"],
            "graph_native_realization_achieved",
        )
        self.assertEqual(
            payload["latest_execution_source_evidence"]["backend_accepted_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            payload["latest_execution_source_evidence"]["backend_realized_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            payload["latest_execution_source_evidence"]["request_primary_plan_kind"],
            "graph_primary",
        )
        self.assertEqual(
            payload["latest_execution_source_evidence"]["execution_behavior_branch"],
            "graph_primary_execution_branch",
        )
        self.assertEqual(
            payload["latest_execution_source_evidence"]["preferred_commit_source"],
            payload["preferred_commit_source"],
        )
        self.assertEqual(
            payload["workflow_metadata"]["execution_source_evidence"]["selected_workflow_graph_patch_id"],
            payload["selected_workflow_graph_patch"]["patch_id"],
        )
        self.assertEqual(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_commit_execution_mode"],
            "graph_native_execution_handoff",
        )
        self.assertEqual(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_commit_execution_authority"],
            "graph_authoritative",
        )
        self.assertEqual(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_commit_execution_implementation_mode"],
            "graph_primary_execution",
        )
        self.assertTrue(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_graph_native_realization_supported"],
        )
        self.assertTrue(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_graph_primary_capable"],
        )
        self.assertTrue(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_graph_commit_payload_supplied"],
        )
        self.assertTrue(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_graph_commit_payload_consumed"],
        )
        self.assertTrue(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_graph_native_execution_realized"],
        )
        self.assertEqual(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_graph_native_realization_reason"],
            "graph_native_realization_achieved",
        )
        self.assertEqual(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_backend_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_accepted_backend_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_realized_backend_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_primary_plan_kind"],
            "graph_primary",
        )
        self.assertEqual(
            payload["workflow_metadata"]["execution_source_evidence"]["backend_echoed_execution_behavior_branch"],
            "graph_primary_execution_branch",
        )


if __name__ == "__main__":
    unittest.main()
