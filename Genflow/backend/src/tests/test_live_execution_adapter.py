import unittest

from app.agent.default_live_backend_client import DefaultLiveBackendClient
from app.agent.live_backend_config import LiveBackendConfig
from app.agent.live_execution_adapter import LiveExecutionAdapter
from app.agent.live_execution_models import (
    CommitExecutionRequest,
    ExecutionRequest,
    ExecutionResponse,
    PreviewExecutionRequest,
)
from app.agent.local_workflow_facade import LocalWorkflowFacade
from app.agent.runtime_models import CommittedPatch, NormalizedSchema, PreviewProbe
from app.agent.workflow_backend_transport import WorkflowBackendTransport
from app.agent.workflow_execution_source_models import (
    WorkflowCommitSource,
    WorkflowExecutionSource,
    WorkflowPreviewSource,
)


class FakeLiveBackendClient:
    def __init__(self):
        self.initial_requests = []
        self.preview_requests = []
        self.commit_requests = []

    def run_initial(self, request: ExecutionRequest) -> ExecutionResponse:
        self.initial_requests.append(request)
        return ExecutionResponse(
            response_id="live-initial-1",
            execution_kind="initial",
            output_payload={"image_id": "img-initial-1", "model": request.schema_snapshot.get("model")},
            summary_text="Live initial result completed.",
            changed_axes=["initial_generation"],
            preserved_axes=[],
            backend_artifacts={"artifact_uri": "memory://initial/1"},
            backend_metadata={"result_type": "live_initial_result", "backend": "fake"},
            comparison_notes=["reference_count=2"],
        )

    def run_preview(self, request: PreviewExecutionRequest) -> ExecutionResponse:
        self.preview_requests.append(request)
        return ExecutionResponse(
            response_id="live-preview-1",
            execution_kind="preview",
            output_payload={"preview_id": "pv-1", "probe_id": request.preview_spec.get("probe_id")},
            summary_text="Live preview completed.",
            changed_axes=list(request.preview_spec.get("target_axes", [])),
            preserved_axes=list(request.preview_spec.get("preserve_axes", [])),
            backend_artifacts={"artifact_uri": "memory://preview/1"},
            backend_metadata={"result_type": "live_preview_result", "backend": "fake"},
            comparison_notes=["preview_source=schema_variation"],
        )

    def run_commit(self, request: CommitExecutionRequest) -> ExecutionResponse:
        self.commit_requests.append(request)
        return ExecutionResponse(
            response_id="live-commit-1",
            execution_kind="commit",
            output_payload={"image_id": "img-commit-1", "patch_id": request.patch_spec.get("patch_id")},
            summary_text="Live committed execution completed.",
            changed_axes=list(request.patch_spec.get("target_axes", [])),
            preserved_axes=list(request.patch_spec.get("preserve_axes", [])),
            backend_artifacts={"artifact_uri": "memory://commit/1"},
            backend_metadata={"result_type": "live_committed_result", "backend": "fake"},
            comparison_notes=["commit_change_keys=style,model"],
        )


class LiveExecutionAdapterTest(unittest.TestCase):
    def test_live_adapter_requires_backend_client(self):
        adapter = LiveExecutionAdapter()
        schema = NormalizedSchema(prompt="portrait", model="sdxl")

        with self.assertRaisesRegex(NotImplementedError, "Live execution adapter is not wired yet."):
            adapter.produce_initial_result(schema)

    def test_live_adapter_maps_initial_request_and_response(self):
        client = FakeLiveBackendClient()
        adapter = LiveExecutionAdapter(backend_client=client)
        schema = NormalizedSchema(
            prompt="a vivid portrait",
            negative_prompt="blurry",
            model="sdxl-base",
            sampler="DPM++ 2M",
            style=["cinematic", "vivid"],
            lora=["portrait-helper"],
        )
        reference_bundle = {
            "query_index": 7,
            "counts": {"best": 1},
            "references": [{"id": 101}, {"id": 202}],
        }

        payload, summary = adapter.produce_initial_result(schema, reference_bundle)

        request = client.initial_requests[0]
        self.assertEqual(request.execution_kind, "initial")
        self.assertEqual(request.schema_snapshot["model"], "sdxl-base")
        self.assertEqual(request.workflow_payload["workflow_kind"], "workflow_native_surrogate")
        self.assertIn("nodes", request.workflow_payload)
        self.assertIn("edges", request.workflow_payload)
        self.assertNotIn("schema", request.workflow_payload)
        self.assertEqual(request.reference_info["reference_ids"], [101, 202])
        self.assertEqual(payload.result_id, "live-initial-1")
        self.assertEqual(payload.result_type, "live_initial_result")
        self.assertEqual(payload.content["model"], "sdxl-base")
        self.assertIn("backend_artifacts", payload.artifacts)
        self.assertEqual(summary.summary_text, "Live initial result completed.")

    def test_live_adapter_builds_typed_initial_execution_source(self):
        schema = NormalizedSchema(prompt="portrait", model="sdxl-base")
        source = LiveExecutionAdapter._build_initial_execution_source(
            schema,
            {"query_index": 7, "references": [{"id": 101}, {"id": 202}]},
        )

        self.assertIsInstance(source, WorkflowExecutionSource)
        self.assertEqual(source.execution_kind, "initial")
        self.assertEqual(source.selected_reference_ids, [101, 202])

    def test_live_adapter_maps_preview_request_and_response(self):
        client = FakeLiveBackendClient()
        adapter = LiveExecutionAdapter(backend_client=client)
        schema = NormalizedSchema(prompt="a vivid portrait", model="sdxl-base")
        probe = PreviewProbe(
            probe_id="p_001",
            summary="push style",
            target_axes=["style"],
            preserve_axes=["composition"],
            preview_execution_spec={"patch_family": "resource_shift"},
            source_kind="schema_variation",
        )

        preview_result = adapter.execute_preview_probe(schema, probe)

        request = client.preview_requests[0]
        self.assertEqual(request.execution_kind, "preview")
        self.assertEqual(request.workflow_payload["execution_kind"], "preview")
        self.assertTrue(request.workflow_payload["preview"])
        self.assertIn("nodes", request.workflow_payload)
        self.assertEqual(request.preview_spec["probe_id"], "p_001")
        self.assertEqual(request.preview_spec["target_axes"], ["style"])
        self.assertEqual(request.preview_spec["graph_patch_spec"]["patch_kind"], "graph_preview_projection")
        self.assertTrue(request.preview_spec["graph_patch_spec"]["node_patches"])
        self.assertEqual(preview_result.probe_id, "p_001")
        self.assertEqual(preview_result.payload.result_type, "live_preview_result")
        self.assertEqual(preview_result.summary.changed_axes, ["style"])
        self.assertEqual(preview_result.comparison_notes, ["preview_source=schema_variation"])

    def test_live_adapter_builds_typed_preview_execution_source(self):
        schema = NormalizedSchema(prompt="portrait", model="sdxl-base")
        probe = PreviewProbe(probe_id="p_001", target_axes=["style"], preserve_axes=["composition"])

        source = LiveExecutionAdapter._build_preview_execution_source(schema, probe)

        self.assertIsInstance(source, WorkflowPreviewSource)
        self.assertEqual(source.selected_probe.probe_id, "p_001")
        self.assertTrue(source.preview)

    def test_live_adapter_maps_commit_request_and_response(self):
        client = FakeLiveBackendClient()
        adapter = LiveExecutionAdapter(backend_client=client)
        schema = NormalizedSchema(prompt="a vivid portrait", model="sdxl-base-patched")
        patch = CommittedPatch(
            patch_id="cp_001",
            target_fields=["style", "model"],
            target_axes=["style"],
            preserve_axes=["composition"],
            changes={"style": ["cinematic", "vivid"], "model": "sdxl-base-patched"},
            rationale="apply stronger style shift",
        )

        payload, summary = adapter.execute_committed_patch(schema, patch)

        request = client.commit_requests[0]
        self.assertEqual(request.execution_kind, "commit")
        self.assertEqual(request.workflow_payload["execution_kind"], "commit")
        self.assertIn("nodes", request.workflow_payload)
        self.assertEqual(request.patch_spec["patch_id"], "cp_001")
        self.assertEqual(request.patch_spec["changes"]["model"], "sdxl-base-patched")
        self.assertEqual(request.patch_spec["graph_patch_spec"]["patch_id"], "cp_001")
        self.assertTrue(request.patch_spec["graph_patch_spec"]["node_patches"])
        self.assertEqual(payload.result_type, "live_committed_result")
        self.assertEqual(payload.content["patch_id"], "cp_001")
        self.assertEqual(summary.changed_axes, ["style"])
        self.assertEqual(summary.preserved_axes, ["composition"])

    def test_live_adapter_builds_typed_commit_execution_source(self):
        schema = NormalizedSchema(prompt="portrait", model="sdxl-base")
        patch = CommittedPatch(
            patch_id="cp_001",
            target_fields=["style"],
            target_axes=["style"],
            preserve_axes=["composition"],
        )

        source = LiveExecutionAdapter._build_commit_execution_source(schema, patch)

        self.assertIsInstance(source, WorkflowCommitSource)
        self.assertEqual(source.accepted_patch.patch_id, "cp_001")

    def test_live_adapter_works_with_concrete_default_client(self):
        client = DefaultLiveBackendClient(
            transport=lambda execution_kind, request: {
                "response_id": "resp-initial-1",
                "execution_kind": execution_kind,
                "output_payload": {"image_id": "img-1", "model": request.schema_snapshot.get("model")},
                "summary_text": "Concrete live client path completed.",
                "changed_axes": ["initial_generation"],
                "preserved_axes": [],
                "backend_artifacts": {"artifact_uri": "memory://concrete/1"},
                "backend_metadata": {"result_type": "live_initial_result", "backend": "fake-transport"},
                "comparison_notes": ["roundtrip=ok"],
            }
        )
        adapter = LiveExecutionAdapter(backend_client=client)
        schema = NormalizedSchema(prompt="portrait", model="sdxl-base")

        payload, summary = adapter.produce_initial_result(schema, {"references": [{"id": 1}]})

        self.assertEqual(payload.result_id, "resp-initial-1")
        self.assertEqual(payload.result_type, "live_initial_result")
        self.assertEqual(payload.content["model"], "sdxl-base")
        self.assertEqual(summary.summary_text, "Concrete live client path completed.")

    def test_live_adapter_works_with_transport_client_and_local_facade(self):
        transport = WorkflowBackendTransport(
            LiveBackendConfig(enabled=True, backend_kind="workflow_shell", endpoint="memory://shell"),
            workflow_facade=LocalWorkflowFacade(),
        )
        client = DefaultLiveBackendClient(transport=transport)
        adapter = LiveExecutionAdapter(backend_client=client)
        schema = NormalizedSchema(prompt="portrait", model="sdxl-base")

        payload, summary = adapter.produce_initial_result(schema, {"references": [{"id": 1}, {"id": 2}]})

        self.assertEqual(payload.result_id, "local-initial")
        self.assertEqual(payload.result_type, "live_initial_result")
        self.assertEqual(payload.content["reference_count"], 2)
        self.assertEqual(summary.summary_text, "Local workflow facade produced initial result with model=sdxl-base and references=2.")


if __name__ == "__main__":
    unittest.main()
