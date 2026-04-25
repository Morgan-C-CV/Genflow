import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, PreviewProbe
from app.agent.workflow_graph_source_builder import (
    build_workflow_graph_source,
    build_workflow_graph_source_from_execution_source,
)
from app.agent.workflow_execution_source_models import WorkflowCommitSource


class WorkflowGraphSourceBuilderTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.workflow_id = f"workflow-{session.session_id}"
        session.workflow_identity.workflow_kind = "workflow_native_surrogate"
        session.workflow_identity.workflow_version = "phase-k-workflow-payload"
        session.workflow_metadata = {"backend_kind": "live_backend", "workflow_profile": "default"}
        session.current_schema.prompt = "a cinematic portrait"
        session.current_schema.model = "sdxl-base"
        session.current_schema_raw = "schema"
        session.selected_gallery_index = 7
        session.selected_reference_ids = [101, 202]
        return session

    def test_builder_produces_stable_initial_graph_source(self):
        session = self._make_session()

        graph_source = build_workflow_graph_source(session, execution_kind="initial", preview=False)

        self.assertEqual(graph_source.workflow_id, session.workflow_id)
        self.assertEqual(graph_source.workflow_kind, "workflow_native_surrogate")
        self.assertEqual(graph_source.entry_node_ids, ["reference.bundle", "intent.prompt"])
        self.assertEqual(graph_source.exit_node_ids, ["result.output"])
        self.assertTrue(graph_source.nodes)
        self.assertTrue(graph_source.edges)
        self.assertEqual(graph_source.regions[0].region_type, "initial_region")

    def test_builder_projects_repair_shape_into_graph_source(self):
        session = self._make_session()
        session.selected_probe = PreviewProbe(
            probe_id="p_001",
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        session.accepted_patch = CommittedPatch(
            patch_id="cp_001",
            target_fields=["style", "model"],
        )

        graph_source = build_workflow_graph_source(session, execution_kind="commit", preview=False)

        self.assertEqual(graph_source.exit_node_ids, ["patch.cp_001", "result.output"])
        self.assertTrue(any(node.role == "repair_probe" for node in graph_source.nodes))
        self.assertTrue(any(node.role == "repair_patch" for node in graph_source.nodes))
        self.assertEqual(graph_source.metadata["accepted_patch_id"], "cp_001")

    def test_builder_can_build_graph_source_from_execution_source(self):
        session = self._make_session()
        patch = CommittedPatch(
            patch_id="cp_001",
            target_fields=["style"],
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        source = WorkflowCommitSource(
            workflow_id="workflow-source-1",
            workflow_kind="workflow_native_surrogate",
            workflow_version="phase-k-workflow-payload",
            execution_kind="commit",
            schema=session.current_schema,
            backend_kind="live_backend",
            workflow_profile="default",
            selected_gallery_index=7,
            selected_reference_ids=[101, 202],
            accepted_patch=patch,
        )

        graph_source = build_workflow_graph_source_from_execution_source(source)

        self.assertEqual(graph_source.workflow_id, "workflow-source-1")
        self.assertEqual(graph_source.workflow_version, "phase-k-workflow-payload")
        self.assertEqual(graph_source.regions[0].region_type, "repair_region")


if __name__ == "__main__":
    unittest.main()
