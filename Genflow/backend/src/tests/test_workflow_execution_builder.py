import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, PreviewProbe
from app.agent.workflow_graph_patch_builder import (
    build_workflow_graph_patch,
    materialize_workflow_graph_patch_from_candidate,
)
from app.agent.workflow_graph_patch_candidate_builder import build_workflow_graph_patch_candidates
from app.agent.workflow_execution_builder import (
    build_workflow_commit_request,
    build_workflow_commit_request_from_source,
    build_workflow_execution_payload,
    build_workflow_execution_payload_from_graph_source,
    build_workflow_execution_payload_from_source,
    build_workflow_preview_request,
    build_workflow_preview_request_from_source,
)
from app.agent.workflow_graph_source_builder import build_workflow_graph_source
from app.agent.workflow_execution_source_models import (
    WorkflowCommitSource,
    WorkflowExecutionSource,
    WorkflowPreviewSource,
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
        session.current_workflow_graph_patch = build_workflow_graph_patch(session)
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

    def test_build_workflow_execution_payload_from_source_produces_same_contract(self):
        source = WorkflowExecutionSource(
            workflow_id="workflow-source-1",
            workflow_kind="workflow_native_surrogate",
            workflow_version="phase-k-workflow-payload",
            execution_kind="initial",
            schema=self._make_session().current_schema,
            selected_gallery_index=7,
            selected_reference_ids=[101, 202],
            selected_reference_bundle={"query_index": 7, "references": [{"id": 101}, {"id": 202}]},
            backend_kind="live_backend",
            workflow_profile="default",
        )

        payload = build_workflow_execution_payload_from_source(source)

        self.assertEqual(payload.workflow_id, "workflow-source-1")
        self.assertEqual(payload.workflow_kind, "workflow_native_surrogate")
        self.assertEqual(payload.workflow_version, "phase-k-workflow-payload")
        self.assertTrue(payload.nodes)
        self.assertTrue(payload.edges)
        self.assertEqual(payload.execution_config["backend_kind"], "live_backend")

    def test_build_workflow_execution_payload_from_graph_source_preserves_contract(self):
        session = self._make_session()
        graph_source = build_workflow_graph_source(session, execution_kind="initial", preview=False)

        payload = build_workflow_execution_payload_from_graph_source(
            graph_source,
            execution_kind="initial",
            preview=False,
        )

        self.assertEqual(payload.workflow_id, session.workflow_id)
        self.assertEqual(payload.execution_kind, "initial")
        self.assertTrue(payload.nodes)
        self.assertTrue(payload.edges)
        self.assertEqual(payload.backend_metadata["source_graph_id"], session.workflow_id)

    def test_build_workflow_preview_request_uses_selected_probe(self):
        session = self._make_session()

        request = build_workflow_preview_request(session)

        self.assertEqual(request.workflow_payload.execution_kind, "preview")
        self.assertTrue(request.workflow_payload.preview)
        self.assertEqual(request.preview_patch_spec["probe_id"], "p_001")
        self.assertEqual(request.preview_patch_spec["target_axes"], ["style"])
        self.assertEqual(request.graph_patch_spec["patch_kind"], "graph_preview_projection")
        self.assertTrue(request.graph_patch_spec["node_patches"])
        self.assertEqual(request.reference_info["reference_ids"], [101, 202])

    def test_build_workflow_preview_request_from_source_uses_typed_source(self):
        session = self._make_session()
        source = WorkflowPreviewSource(
            workflow_id="workflow-preview-source",
            workflow_kind="workflow_native_surrogate",
            workflow_version="phase-k-workflow-payload",
            execution_kind="preview",
            preview=True,
            schema=session.current_schema,
            selected_reference_ids=[101, 202],
            backend_kind="live_backend",
            workflow_profile="default",
            dissatisfaction_axes=["style"],
            preserve_constraints=["composition"],
            selected_probe=session.selected_probe,
        )

        request = build_workflow_preview_request_from_source(source)

        self.assertEqual(request.workflow_payload.execution_kind, "preview")
        self.assertEqual(request.preview_patch_spec["probe_id"], "p_001")
        self.assertEqual(request.graph_patch_spec["patch_kind"], "graph_preview_projection")
        self.assertEqual(request.reference_info["reference_ids"], [101, 202])

    def test_build_workflow_commit_request_uses_committed_patch(self):
        session = self._make_session()
        session.top_schema_patch_candidate = session.accepted_patch

        request = build_workflow_commit_request(session)

        self.assertEqual(request.workflow_payload.execution_kind, "commit")
        self.assertFalse(request.workflow_payload.preview)
        self.assertEqual(request.committed_patch_spec["patch_id"], "cp_001")
        self.assertEqual(request.committed_patch_spec["changes"]["model"], "sdxl-base-patched")
        self.assertEqual(request.graph_patch_spec["patch_id"], "cp_001")
        self.assertTrue(request.graph_patch_spec["node_patches"])
        self.assertEqual(request.commit_source_payload["commit_execution_mode"], "schema_execution_fallback")
        self.assertEqual(request.commit_source_payload["preferred_commit_source"], "schema")
        self.assertEqual(request.commit_source_payload["top_schema_patch_id"], "cp_001")

    def test_build_workflow_commit_request_prefers_selected_workflow_graph_patch_when_graph_preferred(self):
        session = self._make_session()
        graph_candidates = build_workflow_graph_patch_candidates(session)
        session.top_schema_patch_candidate = session.accepted_patch
        session.top_workflow_graph_patch_candidate = graph_candidates[0]
        session.selected_graph_native_patch_candidate = graph_candidates[0]
        session.selected_workflow_graph_patch = materialize_workflow_graph_patch_from_candidate(
            graph_candidates[0],
            session=session,
        )
        session.preferred_commit_source = "graph"

        request = build_workflow_commit_request(session)

        self.assertEqual(request.graph_patch_spec["patch_id"], session.selected_workflow_graph_patch.patch_id)
        self.assertEqual(
            request.graph_patch_spec["metadata"]["candidate_id"],
            session.selected_graph_native_patch_candidate.candidate_id,
        )
        self.assertEqual(request.commit_source_payload["preferred_commit_source"], "graph")
        self.assertEqual(
            request.commit_source_payload["selected_workflow_graph_patch_id"],
            session.selected_workflow_graph_patch.patch_id,
        )
        self.assertEqual(
            request.commit_source_payload["commit_execution_mode"],
            "schema_execution_fallback",
        )
        self.assertEqual(
            request.commit_source_payload["top_graph_patch_candidate_id"],
            session.top_workflow_graph_patch_candidate.candidate_id,
        )

    def test_build_workflow_commit_request_from_source_uses_typed_source(self):
        session = self._make_session()
        source = WorkflowCommitSource(
            workflow_id="workflow-commit-source",
            workflow_kind="workflow_native_surrogate",
            workflow_version="phase-k-workflow-payload",
            execution_kind="commit",
            schema=session.current_schema,
            backend_kind="live_backend",
            workflow_profile="default",
            dissatisfaction_axes=["style"],
            preserve_constraints=["composition"],
            accepted_patch=session.accepted_patch,
        )

        request = build_workflow_commit_request_from_source(source)

        self.assertEqual(request.workflow_payload.execution_kind, "commit")
        self.assertEqual(request.committed_patch_spec["patch_id"], "cp_001")
        self.assertEqual(request.committed_patch_spec["changes"]["model"], "sdxl-base-patched")
        self.assertEqual(request.graph_patch_spec["patch_id"], "cp_001")
        self.assertEqual(request.commit_source_payload["commit_execution_mode"], "schema_execution_fallback")


if __name__ == "__main__":
    unittest.main()
