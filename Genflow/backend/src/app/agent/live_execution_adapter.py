from __future__ import annotations

from typing import Dict, Optional
from dataclasses import asdict

from app.agent.execution_adapter import ExecutionAdapter
from app.agent.live_backend_client import LiveBackendClient
from app.agent.live_backend_errors import LiveBackendResponseError
from app.agent.live_execution_models import (
    CommitExecutionRequest,
    ExecutionRequest,
    ExecutionResponse,
    PreviewExecutionRequest,
)
from app.agent.runtime_models import (
    CommittedPatch,
    NormalizedSchema,
    PreviewProbe,
    PreviewResult,
    ResultPayload,
    ResultSummary,
)
from app.agent.workflow_execution_builder import (
    build_workflow_commit_request_from_source,
    build_workflow_execution_payload_from_source,
    build_workflow_preview_request_from_source,
)
from app.agent.workflow_execution_source_models import (
    WorkflowCommitSource,
    WorkflowExecutionSource,
    WorkflowPreviewSource,
)


class LiveExecutionAdapter(ExecutionAdapter):
    def __init__(
        self,
        backend_client: LiveBackendClient | None = None,
        artifact_store=None,
    ):
        self.backend_client = backend_client
        self.artifact_store = artifact_store

    def produce_initial_result(
        self,
        schema: NormalizedSchema,
        reference_bundle: Optional[Dict[str, object]] = None,
    ) -> tuple[ResultPayload, ResultSummary]:
        client = self._require_backend_client()
        source = self._build_initial_execution_source(schema, reference_bundle)
        workflow_payload = build_workflow_execution_payload_from_source(source)
        request = ExecutionRequest(
            execution_kind="initial",
            schema_snapshot=self._build_schema_snapshot(schema),
            workflow_payload=asdict(workflow_payload),
            reference_info=self._build_reference_info(reference_bundle),
        )
        response = client.run_initial(request)
        return self._map_result_response(response, default_result_type="live_initial_result")

    def execute_preview_probe(
        self,
        schema: NormalizedSchema,
        probe: PreviewProbe,
    ) -> PreviewResult:
        client = self._require_backend_client()
        source = self._build_preview_execution_source(schema, probe)
        workflow_request = build_workflow_preview_request_from_source(source)
        request = PreviewExecutionRequest(
            execution_kind="preview",
            schema_snapshot=self._build_schema_snapshot(schema),
            workflow_payload=asdict(workflow_request.workflow_payload),
            reference_info=dict(workflow_request.reference_info),
            preview_spec=dict(workflow_request.preview_patch_spec),
        )
        response = client.run_preview(request)
        payload, summary = self._map_result_response(
            response,
            default_result_type="live_preview_result",
        )
        return PreviewResult(
            probe_id=probe.probe_id,
            summary=summary,
            payload=payload,
            comparison_notes=list(response.comparison_notes),
        )

    def execute_committed_patch(
        self,
        schema: NormalizedSchema,
        patch: CommittedPatch,
    ) -> tuple[ResultPayload, ResultSummary]:
        client = self._require_backend_client()
        source = self._build_commit_execution_source(schema, patch)
        workflow_request = build_workflow_commit_request_from_source(source)
        request = CommitExecutionRequest(
            execution_kind="commit",
            schema_snapshot=self._build_schema_snapshot(schema),
            workflow_payload=asdict(workflow_request.workflow_payload),
            reference_info=dict(workflow_request.reference_info),
            patch_spec=dict(workflow_request.committed_patch_spec),
        )
        response = client.run_commit(request)
        return self._map_result_response(response, default_result_type="live_committed_result")

    def _require_backend_client(self) -> LiveBackendClient:
        if self.backend_client is None:
            raise NotImplementedError("Live execution adapter is not wired yet.")
        return self.backend_client

    @staticmethod
    def _build_schema_snapshot(schema: NormalizedSchema) -> Dict[str, object]:
        return {
            "prompt": schema.prompt,
            "negative_prompt": schema.negative_prompt,
            "cfgscale": schema.cfgscale,
            "steps": schema.steps,
            "sampler": schema.sampler,
            "seed": schema.seed,
            "model": schema.model,
            "clipskip": schema.clipskip,
            "style": list(schema.style),
            "lora": list(schema.lora),
        }

    @staticmethod
    def _build_initial_execution_source(
        schema: NormalizedSchema,
        reference_bundle: Optional[Dict[str, object]],
    ) -> WorkflowExecutionSource:
        reference_bundle = reference_bundle or {}
        return WorkflowExecutionSource(
            workflow_id="workflow-live-adapter-initial",
            workflow_kind="workflow_native_surrogate",
            workflow_version="phase-k-workflow-payload",
            execution_kind="initial",
            preview=False,
            schema=schema,
            selected_gallery_index=reference_bundle.get("query_index"),
            selected_reference_ids=[
                item["id"]
                for item in reference_bundle.get("references", [])
                if isinstance(item, dict) and "id" in item
            ],
            selected_reference_bundle=dict(reference_bundle),
            backend_kind="live_backend",
            workflow_profile="default",
        )

    @staticmethod
    def _build_preview_execution_source(
        schema: NormalizedSchema,
        probe: PreviewProbe,
    ) -> WorkflowPreviewSource:
        return WorkflowPreviewSource(
            workflow_id="workflow-live-adapter-preview",
            workflow_kind="workflow_native_surrogate",
            workflow_version="phase-k-workflow-payload",
            execution_kind="preview",
            preview=True,
            schema=schema,
            backend_kind="live_backend",
            workflow_profile="default",
            dissatisfaction_axes=list(probe.target_axes),
            preserve_constraints=list(probe.preserve_axes),
            selected_probe=probe,
        )

    @staticmethod
    def _build_commit_execution_source(
        schema: NormalizedSchema,
        patch: CommittedPatch,
    ) -> WorkflowCommitSource:
        return WorkflowCommitSource(
            workflow_id="workflow-live-adapter-commit",
            workflow_kind="workflow_native_surrogate",
            workflow_version="phase-k-workflow-payload",
            execution_kind="commit",
            preview=False,
            schema=schema,
            backend_kind="live_backend",
            workflow_profile="default",
            dissatisfaction_axes=list(patch.target_axes),
            preserve_constraints=list(patch.preserve_axes),
            accepted_patch=patch,
        )

    @staticmethod
    def _build_reference_info(reference_bundle: Optional[Dict[str, object]]) -> Dict[str, object]:
        reference_bundle = reference_bundle or {}
        references = reference_bundle.get("references", [])
        reference_ids = []
        for item in references:
            if isinstance(item, dict) and "id" in item:
                reference_ids.append(item["id"])
        return {
            "query_index": reference_bundle.get("query_index"),
            "counts": dict(reference_bundle.get("counts", {})),
            "reference_ids": reference_ids,
            "reference_count": len(references),
        }

    @staticmethod
    def _map_result_response(
        response: ExecutionResponse,
        default_result_type: str,
    ) -> tuple[ResultPayload, ResultSummary]:
        if not response.response_id:
            raise LiveBackendResponseError("Live adapter received response without response_id.")
        payload = ResultPayload(
            result_id=response.response_id,
            result_type=response.backend_metadata.get("result_type", default_result_type),
            content=dict(response.output_payload),
            artifacts={
                "backend_artifacts": dict(response.backend_artifacts),
                "backend_metadata": dict(response.backend_metadata),
            },
        )
        summary = ResultSummary(
            summary_text=response.summary_text,
            changed_axes=list(response.changed_axes),
            preserved_axes=list(response.preserved_axes),
            notes=list(response.comparison_notes),
        )
        return payload, summary
