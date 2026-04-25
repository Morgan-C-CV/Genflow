import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import PreviewProbe
from app.agent.workflow_graph_patch_candidate_builder import (
    build_workflow_graph_patch_candidates,
    build_workflow_graph_patch_candidates_from_probe,
)
from app.agent.workflow_graph_source_builder import build_workflow_graph_source


class WorkflowGraphPatchCandidateBuilderTest(unittest.TestCase):
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
        session.latest_feedback = "Keep composition, improve style."
        session.dissatisfaction_axes = ["style", "lighting"]
        session.preserve_constraints = ["composition"]
        session.selected_probe = PreviewProbe(
            probe_id="p_002",
            summary="push cinematic style while preserving composition",
            target_axes=["style"],
            preserve_axes=["composition"],
            preview_execution_spec={"patch_family": "resource_shift"},
            source_kind="schema_variation",
        )
        session.repair_hypotheses = [object(), object()]
        return session

    def test_graph_patch_candidates_are_stable_and_multiple(self):
        session = self._make_session()

        candidates = build_workflow_graph_patch_candidates(session)

        self.assertGreater(len(candidates), 1)
        self.assertEqual(candidates[0].workflow_id, session.workflow_id)
        self.assertTrue(all(candidate.node_patches for candidate in candidates))

    def test_graph_patch_candidates_can_be_built_from_probe_and_graph_source(self):
        session = self._make_session()
        graph_source = build_workflow_graph_source(session, execution_kind="commit_plan", preview=False)

        candidates = build_workflow_graph_patch_candidates_from_probe(
            selected_probe=session.selected_probe,
            graph_source=graph_source,
            repair_context={
                "preserve_constraints": session.preserve_constraints,
                "dissatisfaction_axes": session.dissatisfaction_axes,
                "repair_hypothesis_count": len(session.repair_hypotheses),
            },
        )

        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0].metadata["probe_id"], "p_002")
        self.assertTrue(any(candidate.candidate_kind == "conservative" for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
