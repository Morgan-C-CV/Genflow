import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, PreviewProbe
from app.agent.workflow_execution_builder import (
    build_workflow_commit_request,
    build_workflow_execution_payload,
    build_workflow_preview_request,
)


class WorkflowExecutionBuilderTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.workflow_id = f"workflow-{session.session_id}"
        session.workflow_identity.workflow_kind = "workflow_native_surrogate"
        session.workflow_identity.workflow_version = "phase-k-workflow-payload"
        session.workflow_metadata = {"backend_kind": "live_backend", "workflow_profile": "default"}
        session.current_schema.prompt = "a vivid portrait"
        session.current_schema.model = "sdxl-base"
        session.current_schema.sampler = "DPM++ 2M"
        session.current_schema_raw = "schema"
        session.selected_gallery_index = 7
        session.selected_reference_ids = [101, 202]
        session.selected_reference_bundle = {"query_index": 7, "references": [{"id": 101}, {"id": 202}]}
        session.selected_probe = PreviewProbe(
            probe_id="p_001",
            summary="push style",
            target_axes=["style"],
            preserve_axes=["composition"],
            preview_execution_spec={"patch_family": "resource_shift"},
            source_kind="schema_variation",
        )
        session.accepted_patch = CommittedPatch(
            patch_id="cp_001",
            target_fields=["style", "model"],
            target_axes=["style"],
            preserve_axes=["composition"],
            changes={"style": ["cinematic"], "model": "sdxl-base-patched"},
            rationale="commit style shift",
        )
        return session

    def test_build_workflow_execution_payload_produces_workflow_native_shape(self):
        session = self._make_session()

        payload = build_workflow_execution_payload(session, execution_kind="initial", preview=False)

        self.assertEqual(payload.workflow_id, session.workflow_id)
        self.assertEqual(payload.workflow_kind, "workflow_native_surrogate")
        self.assertTrue(payload.nodes)
        self.assertTrue(payload.edges)
        self.assertEqual(payload.execution_config["execution_kind"], "initial")
        self.assertEqual(payload.backend_metadata["source_document_id"], f"{session.workflow_id}:initial")

    def test_build_workflow_preview_request_uses_selected_probe(self):
        session = self._make_session()

        request = build_workflow_preview_request(session)

        self.assertEqual(request.workflow_payload.execution_kind, "preview")
        self.assertTrue(request.workflow_payload.preview)
        self.assertEqual(request.preview_patch_spec["probe_id"], "p_001")
        self.assertEqual(request.preview_patch_spec["target_axes"], ["style"])
        self.assertEqual(request.reference_info["reference_ids"], [101, 202])

    def test_build_workflow_commit_request_uses_committed_patch(self):
        session = self._make_session()

        request = build_workflow_commit_request(session)

        self.assertEqual(request.workflow_payload.execution_kind, "commit")
        self.assertFalse(request.workflow_payload.preview)
        self.assertEqual(request.committed_patch_spec["patch_id"], "cp_001")
        self.assertEqual(request.committed_patch_spec["changes"]["model"], "sdxl-base-patched")


if __name__ == "__main__":
    unittest.main()
