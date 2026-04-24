import unittest

from app.agent.memory import AgentMemoryService
from app.agent.workflow_descriptor_builder import build_surrogate_workflow_descriptor
from app.agent.workflow_document_builder import build_surrogate_workflow_document_from_descriptor
from app.agent.workflow_graph_builder import build_surrogate_workflow_graph_from_document
from app.agent.workflow_snapshot_builder import build_surrogate_workflow_snapshot


class WorkflowArchitectureSanityTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.workflow_id = f"workflow-{session.session_id}"
        session.workflow_identity.workflow_kind = "normalized_schema_surrogate"
        session.workflow_identity.workflow_version = "phase-g-skeleton"
        session.workflow_metadata = {
            "backend_kind": "mock",
            "workflow_profile": "default",
        }
        session.current_schema.prompt = "a cinematic portrait"
        session.current_schema.negative_prompt = "blurry"
        session.current_schema.model = "sdxl-base"
        session.current_schema.sampler = "DPM++ 2M"
        session.current_schema.style = ["cinematic"]
        session.current_schema.lora = ["portrait-helper"]
        session.current_schema_raw = "schema"
        session.selected_gallery_index = 7
        session.selected_reference_ids = [101, 202]
        session.current_result_id = "result-1"
        session.latest_feedback = "Keep composition, improve style."
        session.feedback_history = [session.latest_feedback]
        session.dissatisfaction_axes = ["style"]
        session.preserve_constraints = ["composition"]
        session.current_uncertainty_estimate = 0.25
        return session

    def test_builder_layers_chain_together_without_runtime_service(self):
        session = self._make_session()

        descriptor = build_surrogate_workflow_descriptor(
            session,
            execution_kind="initial",
            preview=False,
        )
        document = build_surrogate_workflow_document_from_descriptor(descriptor)
        graph, hints = build_surrogate_workflow_graph_from_document(document)
        snapshot = build_surrogate_workflow_snapshot(
            session,
            execution_kind="initial",
            preview=False,
            backend_kind="mock",
            workflow_profile="default",
        )

        self.assertEqual(document.workflow_id, descriptor.workflow_id)
        self.assertEqual(graph.graph_id, document.workflow_id)
        self.assertEqual(snapshot.workflow_identity.workflow_id, descriptor.workflow_id)
        self.assertEqual(snapshot.workflow_metadata["backend_kind"], descriptor.execution.backend_kind)
        self.assertEqual(
            snapshot.surrogate_payload["workflow_document_id"],
            document.document_id,
        )
        self.assertEqual(
            snapshot.workflow_graph_placeholder.entry_node_ids,
            graph.entry_node_ids,
        )
        self.assertEqual(
            snapshot.workflow_topology_hints["region_label"],
            hints["region_label"],
        )


if __name__ == "__main__":
    unittest.main()
