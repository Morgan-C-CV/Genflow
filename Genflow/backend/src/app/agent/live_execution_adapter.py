from __future__ import annotations

from typing import Dict, Optional

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
        request = ExecutionRequest(
            execution_kind="initial",
            schema_snapshot=self._build_schema_snapshot(schema),
            workflow_payload=self._build_workflow_payload(schema),
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
        request = PreviewExecutionRequest(
            execution_kind="preview",
            schema_snapshot=self._build_schema_snapshot(schema),
            workflow_payload=self._build_workflow_payload(schema),
            reference_info={},
            preview_spec={
                "probe_id": probe.probe_id,
                "summary": probe.summary,
                "target_axes": list(probe.target_axes),
                "preserve_axes": list(probe.preserve_axes),
                "source_kind": probe.source_kind,
                "preview_execution_spec": dict(probe.preview_execution_spec),
            },
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
        request = CommitExecutionRequest(
            execution_kind="commit",
            schema_snapshot=self._build_schema_snapshot(schema),
            workflow_payload=self._build_workflow_payload(schema),
            reference_info={},
            patch_spec={
                "patch_id": patch.patch_id,
                "target_fields": list(patch.target_fields),
                "target_axes": list(patch.target_axes),
                "preserve_axes": list(patch.preserve_axes),
                "changes": dict(patch.changes),
                "rationale": patch.rationale,
            },
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
    def _build_workflow_payload(schema: NormalizedSchema) -> Dict[str, object]:
        return {
            "workflow_kind": "normalized_schema_surrogate",
            "schema": {
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
                "full_metadata_string": schema.full_metadata_string,
                "raw_fields": dict(schema.raw_fields),
            },
        }

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
