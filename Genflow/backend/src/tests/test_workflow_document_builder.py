import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, PreviewProbe
from app.agent.workflow_document_builder import build_surrogate_workflow_document
from app.agent.workflow_document_models import SurrogateWorkflowDocument
from app.agent.workflow_graph_builder import build_surrogate_workflow_graph, build_surrogate_workflow_graph_from_document


class WorkflowDocumentBuilderTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.workflow_id = f"workflow-{session.session_id}"
        session.workflow_identity.workflow_kind = "surrogate_workflow"
        session.workflow_metadata = {
            "backend_kind": "mock",
            "workflow_profile": "default",
        }
        session.current_schema.prompt = "a cinematic portrait"
        session.current_schema.model = "sdxl-base"
        session.current_schema_raw = "schema"
        return session

    def test_document_defaults_are_safe(self):
        document = SurrogateWorkflowDocument()

        self.assertEqual(document.document_id, "")
        self.assertEqual(document.nodes, [])
        self.assertEqual(document.edges, [])
        self.assertEqual(document.regions, [])
        self.assertEqual(document.metadata, {})

    def test_builder_produces_stable_initial_document(self):
        session = self._make_session()
        session.selected_gallery_index = 7

        document = build_surrogate_workflow_document(session, execution_kind="initial", preview=False)

        self.assertEqual(document.workflow_id, session.workflow_id)
        self.assertEqual(document.entry_node_ids, ["reference.bundle", "intent.prompt"])
        self.assertEqual(document.exit_node_ids, ["result.output"])
        self.assertEqual(document.regions[0].region_kind, "initial_region")
        self.assertTrue(any(node.role == "reference" for node in document.nodes))
        self.assertTrue(any(edge.edge_kind == "references_inform_prompt" for edge in document.edges))

    def test_builder_includes_repair_nodes_and_region_for_probe_and_patch(self):
        session = self._make_session()
        session.selected_gallery_index = 7
        session.selected_probe = PreviewProbe(
            probe_id="p_001",
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        session.accepted_patch = CommittedPatch(
            patch_id="cp_p_001",
            target_fields=["style", "model"],
        )

        document = build_surrogate_workflow_document(session, execution_kind="commit", preview=False)

        self.assertEqual(document.metadata["graph_regions"], ["repair_region"])
        self.assertEqual(document.regions[0].region_kind, "repair_region")
        self.assertEqual(document.exit_node_ids, ["patch.cp_p_001", "result.output"])
        self.assertTrue(any(node.role == "repair_probe" for node in document.nodes))
        self.assertTrue(any(node.role == "repair_patch" for node in document.nodes))

    def test_graph_builder_from_document_preserves_existing_graph_shape(self):
        session = self._make_session()
        session.selected_gallery_index = 7
        document = build_surrogate_workflow_document(session, execution_kind="initial", preview=False)

        graph_from_document, hints_from_document = build_surrogate_workflow_graph_from_document(document)
        graph_from_session, hints_from_session = build_surrogate_workflow_graph(
            session,
            execution_kind="initial",
            preview=False,
        )

        self.assertEqual(graph_from_document.entry_node_ids, graph_from_session.entry_node_ids)
        self.assertEqual(graph_from_document.exit_node_ids, graph_from_session.exit_node_ids)
        self.assertEqual(
            [(node.node_id, node.role) for node in graph_from_document.node_refs],
            [(node.node_id, node.role) for node in graph_from_session.node_refs],
        )
        self.assertEqual(hints_from_document, hints_from_session)


if __name__ == "__main__":
    unittest.main()
