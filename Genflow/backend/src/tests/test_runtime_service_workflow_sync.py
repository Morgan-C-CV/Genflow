import unittest

from app.agent.memory import AgentMemoryService
from app.agent.result_executor import ResultExecutor
from app.agent.runtime_service import AgentRuntimeService
from app.agent.workflow_descriptor_builder import build_surrogate_workflow_descriptor
from app.agent.workflow_document_builder import build_surrogate_workflow_document_from_descriptor
from run_agent_demo import build_session_artifact_payload
from tests.test_runtime_service import (
    FakeFeedbackParser,
    FakeHypothesisBuilder,
    FakeOrchestrationService,
    FakePatchPlanner,
    FakeProbeGenerator,
    FakeSearchService,
)


class RuntimeServiceWorkflowSyncTest(unittest.TestCase):
    def _build_service(self, result_ids):
        memory = AgentMemoryService()
        executor = ResultExecutor(id_factory=lambda: next(result_ids))
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=FakeOrchestrationService(memory),
            search_service=FakeSearchService(),
            execution_adapter=executor,
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            probe_generator=FakeProbeGenerator(),
            patch_planner=FakePatchPlanner(),
        )
        return memory, service

    def test_workflow_identity_and_surrogate_payload_are_filled_by_initial_schema_and_result(self):
        _, service = self._build_service(iter(["result-1"]))
        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)

        self.assertTrue(session.workflow_id)
        self.assertEqual(session.workflow_id, session.workflow_identity.workflow_id)
        self.assertEqual(session.workflow_state.identity.workflow_id, session.workflow_id)
        self.assertEqual(session.workflow_identity.workflow_kind, "normalized_schema_surrogate")
        self.assertEqual(session.workflow_state.workflow_metadata["surrogate_kind"], "normalized_schema")
        self.assertEqual(session.workflow_state.surrogate_payload["schema"]["model"], "sdxl-base")
        self.assertEqual(
            session.workflow_state.surrogate_payload["workflow_document_entry_node_ids"],
            ["reference.bundle", "intent.prompt"],
        )
        self.assertEqual(session.last_execution_config.execution_kind, "initial")
        self.assertFalse(session.last_execution_config.preview)

    def test_preview_probe_updates_preview_execution_config(self):
        _, service = self._build_service(iter(["initial-1", "preview-1"]))
        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.preview_probe(session.session_id, "p_001")

        self.assertEqual(session.last_execution_config.execution_kind, "preview")
        self.assertTrue(session.last_execution_config.preview)
        self.assertEqual(session.workflow_state.last_execution_config.execution_kind, "preview")
        self.assertTrue(session.workflow_state.last_execution_config.preview)

    def test_refinement_events_sync_workflow_metadata_and_surrogate_payload(self):
        _, service = self._build_service(iter(["initial-1"]))
        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)

        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        self.assertEqual(session.last_execution_config.execution_kind, "feedback")
        self.assertTrue(session.workflow_metadata["has_feedback"])
        self.assertEqual(session.workflow_metadata["feedback_count"], 1)
        self.assertEqual(session.workflow_metadata["dissatisfaction_axes"], ["style"])
        self.assertEqual(session.workflow_metadata["preserve_constraints"], ["Keep the composition"])
        self.assertEqual(
            session.workflow_state.surrogate_payload["latest_feedback"],
            "Keep the composition, but improve style.",
        )
        self.assertEqual(session.workflow_state.surrogate_payload["current_uncertainty_estimate"], 0.25)

        session = service.build_repair_hypotheses(session.session_id)
        self.assertEqual(session.last_execution_config.execution_kind, "repair_hypotheses")
        self.assertEqual(session.workflow_metadata["repair_hypothesis_count"], 2)
        self.assertEqual(session.workflow_state.surrogate_payload["repair_hypothesis_count"], 2)
        self.assertEqual(
            session.workflow_metadata["benchmark_comparison"]["benchmark_source"],
            session.benchmark_comparison_summary.metadata["benchmark_source"],
        )
        self.assertEqual(
            session.workflow_state.surrogate_payload["benchmark_comparison"]["compared_anchor_ids"],
            session.benchmark_comparison_summary.compared_anchor_ids,
        )

        session = service.generate_local_probes(session.session_id)
        self.assertEqual(session.last_execution_config.execution_kind, "probe_generation")
        self.assertEqual(session.workflow_metadata["probe_count"], 2)
        self.assertEqual(session.workflow_state.surrogate_payload["probe_count"], 2)
        self.assertEqual(
            session.workflow_state.editable_scopes[0].node_ids,
            session.editable_scopes[0].node_ids,
        )
        self.assertEqual(
            session.workflow_state.protected_scopes[0].node_ids,
            session.protected_scopes[0].node_ids,
        )

        session = service.select_probe(session.session_id, "p_001")
        self.assertEqual(session.last_execution_config.execution_kind, "probe_select")
        self.assertEqual(session.workflow_metadata["selected_probe_id"], "p_001")
        self.assertEqual(session.workflow_state.surrogate_payload["selected_probe_id"], "p_001")
        self.assertEqual(session.workflow_metadata["document_region_label"], "repair_region")
        self.assertEqual(
            session.workflow_state.last_execution_config.execution_kind,
            session.last_execution_config.execution_kind,
        )

    def test_execute_patch_updates_workflow_state_and_artifact_consistently(self):
        _, service = self._build_service(iter(["initial-1", "preview-1", "commit-1"]))
        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.preview_probe(session.session_id, "p_001")
        session = service.select_probe(session.session_id, "p_001")
        session = service.commit_patch(session.session_id)
        session = service.execute_patch(session.session_id)

        self.assertEqual(session.last_execution_config.execution_kind, "commit")
        self.assertFalse(session.last_execution_config.preview)
        self.assertEqual(session.workflow_state.last_execution_config.execution_kind, "commit")
        self.assertEqual(session.workflow_state.surrogate_payload["accepted_patch_id"], "cp_p_001")
        self.assertEqual(session.editable_scopes[0].node_ids, ["style", "model"])
        self.assertEqual(session.protected_scopes[0].node_ids, ["Keep the composition"])
        self.assertEqual(
            session.protected_scopes[0].node_ids,
            session.workflow_state.protected_scopes[0].node_ids,
        )

        payload = build_session_artifact_payload(session)
        self.assertEqual(payload["workflow_id"], payload["workflow_identity"]["workflow_id"])
        self.assertEqual(
            payload["workflow_state"]["identity"]["workflow_id"],
            payload["workflow_id"],
        )
        self.assertEqual(
            payload["last_execution_config"]["execution_kind"],
            payload["workflow_state"]["last_execution_config"]["execution_kind"],
        )
        self.assertEqual(
            payload["workflow_metadata"]["backend_kind"],
            payload["workflow_state"]["workflow_metadata"]["backend_kind"],
        )

    def test_runtime_sync_metadata_and_payload_align_with_descriptor_and_document(self):
        _, service = self._build_service(iter(["initial-1"]))
        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.select_probe(session.session_id, "p_001")

        descriptor = build_surrogate_workflow_descriptor(
            session,
            execution_kind=session.last_execution_config.execution_kind,
            preview=session.last_execution_config.preview,
        )
        document = build_surrogate_workflow_document_from_descriptor(descriptor)

        self.assertEqual(session.workflow_metadata["backend_kind"], descriptor.execution.backend_kind)
        self.assertEqual(session.workflow_metadata["workflow_profile"], descriptor.execution.workflow_profile)
        self.assertEqual(session.workflow_metadata["selected_probe_id"], descriptor.repair.selected_probe_id)
        self.assertEqual(
            session.workflow_state.surrogate_payload["schema"]["model"],
            descriptor.schema_model,
        )
        self.assertEqual(
            session.workflow_state.surrogate_payload["selected_reference_ids"],
            descriptor.selected_reference_ids,
        )
        self.assertEqual(
            session.workflow_state.surrogate_payload["accepted_patch_id"],
            descriptor.repair.accepted_patch_id,
        )
        self.assertEqual(
            session.workflow_state.surrogate_payload["workflow_document_region_label"],
            document.metadata["region_label"],
        )
        self.assertEqual(
            session.workflow_state.surrogate_payload["workflow_document_entry_node_ids"],
            document.entry_node_ids,
        )
        self.assertEqual(
            session.workflow_topology_hints["document_region_label"],
            document.metadata["region_label"],
        )
        self.assertEqual(
            session.workflow_topology_entry_node_ids,
            document.entry_node_ids,
        )
        self.assertEqual(
            session.workflow_topology_exit_node_ids,
            document.exit_node_ids,
        )
        self.assertEqual(
            session.workflow_state.workflow_metadata["benchmark_comparison"]["compared_candidate_ids"],
            session.benchmark_comparison_summary.compared_candidate_ids,
        )


if __name__ == "__main__":
    unittest.main()
