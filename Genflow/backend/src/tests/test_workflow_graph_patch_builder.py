import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, PreviewProbe
from app.agent.workflow_graph_patch_builder import (
    build_workflow_graph_patch,
    build_workflow_graph_patch_from_committed_patch,
)
from app.agent.workflow_graph_source_builder import build_workflow_graph_source


class WorkflowGraphPatchBuilderTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.workflow_id = f"workflow-{session.session_id}"
        session.workflow_identity.workflow_kind = "workflow_native_surrogate"
        session.workflow_identity.workflow_version = "phase-k-workflow-payload"
        session.workflow_metadata = {"backend_kind": "live_backend", "workflow_profile": "default"}
        session.current_schema.prompt = "a cinematic portrait"
        session.current_schema.model = "sdxl-base-patched"
        session.current_schema.style = ["cinematic", "vivid"]
        session.current_schema_raw = "schema"
        session.selected_gallery_index = 7
        session.selected_reference_ids = [101, 202]
        session.selected_probe = PreviewProbe(
            probe_id="p_002",
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        session.accepted_patch = CommittedPatch(
            patch_id="cp_001",
            target_fields=["style", "model"],
            target_axes=["style"],
            preserve_axes=["composition"],
            changes={"style": ["cinematic", "vivid"], "model": "sdxl-base-patched"},
            rationale="apply style-focused committed patch",
        )
        return session

    def test_builder_projects_committed_patch_into_graph_patch(self):
        session = self._make_session()

        graph_patch = build_workflow_graph_patch(session)

        self.assertEqual(graph_patch.workflow_id, session.workflow_id)
        self.assertEqual(graph_patch.patch_id, "cp_001")
        self.assertEqual(graph_patch.patch_kind, "graph_intent_projection")
        self.assertTrue(graph_patch.node_patches)
        self.assertTrue(graph_patch.edge_patches)
        self.assertTrue(graph_patch.region_patches)

    def test_builder_supports_direct_committed_patch_projection(self):
        session = self._make_session()
        graph_source = build_workflow_graph_source(session, execution_kind="commit_plan", preview=False)

        graph_patch = build_workflow_graph_patch_from_committed_patch(
            committed_patch=session.accepted_patch,
            graph_source=graph_source,
            session=session,
        )

        self.assertEqual(graph_patch.metadata["selected_probe_id"], "p_002")
        self.assertEqual(graph_patch.metadata["target_fields"], ["style", "model"])
        self.assertTrue(any(patch.node_id == "render.model" for patch in graph_patch.node_patches))


if __name__ == "__main__":
    unittest.main()
