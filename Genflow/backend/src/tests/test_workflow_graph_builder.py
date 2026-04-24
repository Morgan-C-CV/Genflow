import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, PreviewProbe
from app.agent.workflow_graph_builder import build_surrogate_workflow_graph


class WorkflowGraphBuilderTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.workflow_id = f"workflow-{session.session_id}"
        session.workflow_metadata = {
            "backend_kind": "mock",
            "workflow_profile": "default",
        }
        session.current_schema.prompt = "a cinematic portrait"
        session.current_schema.model = "sdxl-base"
        session.current_schema_raw = "schema"
        return session

    def test_builder_produces_stable_initial_graph_placeholder(self):
        session = self._make_session()
        session.selected_gallery_index = 7

        placeholder, hints = build_surrogate_workflow_graph(session, execution_kind="initial", preview=False)

        self.assertEqual(placeholder.graph_kind, "surrogate_topology")
        self.assertEqual(placeholder.entry_node_ids, ["reference.bundle", "intent.prompt"])
        self.assertEqual(placeholder.exit_node_ids, ["result.output"])
        self.assertEqual(placeholder.topology_slices[0].slice_kind, "initial_region")
        self.assertTrue(any(node.role == "reference" for node in placeholder.node_refs))
        self.assertEqual(hints["region_label"], "initial_region")

    def test_builder_produces_repair_region_graph_for_probe_and_patch(self):
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

        placeholder, hints = build_surrogate_workflow_graph(session, execution_kind="commit", preview=False)

        self.assertEqual(placeholder.metadata["graph_regions"], ["repair_region"])
        self.assertEqual(placeholder.topology_slices[0].slice_kind, "repair_region")
        self.assertEqual(placeholder.exit_node_ids, ["patch.cp_p_001", "result.output"])
        self.assertTrue(any(node.role == "repair_probe" for node in placeholder.node_refs))
        self.assertTrue(any(node.role == "repair_patch" for node in placeholder.node_refs))
        self.assertEqual(hints["accepted_patch_id"], "cp_p_001")

    def test_builder_is_stable_for_same_input(self):
        session = self._make_session()
        session.selected_gallery_index = 7

        first_placeholder, first_hints = build_surrogate_workflow_graph(session, execution_kind="initial", preview=False)
        second_placeholder, second_hints = build_surrogate_workflow_graph(session, execution_kind="initial", preview=False)

        self.assertEqual(first_placeholder.entry_node_ids, second_placeholder.entry_node_ids)
        self.assertEqual(first_placeholder.exit_node_ids, second_placeholder.exit_node_ids)
        self.assertEqual(
            [(node.node_id, node.role, node.upstream_ids, node.downstream_ids) for node in first_placeholder.node_refs],
            [(node.node_id, node.role, node.upstream_ids, node.downstream_ids) for node in second_placeholder.node_refs],
        )
        self.assertEqual(first_hints, second_hints)


if __name__ == "__main__":
    unittest.main()
