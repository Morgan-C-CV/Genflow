from __future__ import annotations

from app.agent.live_execution_models import (
    CommitExecutionRequest,
    ExecutionRequest,
    ExecutionResponse,
    PreviewExecutionRequest,
)


class LocalWorkflowFacade:
    def run(self, execution_kind: str, request: ExecutionRequest) -> ExecutionResponse:
        if execution_kind == "initial":
            return self._run_initial(request)
        if execution_kind == "preview":
            return self._run_preview(request)
        if execution_kind == "commit":
            return self._run_commit(request)
        raise ValueError(f"Unsupported local workflow execution kind: {execution_kind}.")

    @staticmethod
    def _run_initial(request: ExecutionRequest) -> ExecutionResponse:
        schema = request.schema_snapshot
        reference_ids = list(request.reference_info.get("reference_ids", []))
        model = str(schema.get("model", ""))
        return ExecutionResponse(
            response_id="local-initial",
            execution_kind="initial",
            output_payload={
                "image_id": "local-image-initial",
                "model": model,
                "prompt": schema.get("prompt", ""),
                "reference_count": len(reference_ids),
            },
            summary_text=(
                f"Local workflow facade produced initial result with model={model or 'unknown'} "
                f"and references={len(reference_ids)}."
            ),
            changed_axes=["initial_generation"],
            preserved_axes=[],
            backend_artifacts={"artifact_uri": "memory://local-workflow/initial"},
            backend_metadata={"result_type": "live_initial_result", "backend": "local_workflow_facade"},
            comparison_notes=[f"reference_count={len(reference_ids)}"],
        )

    @staticmethod
    def _run_preview(request: PreviewExecutionRequest) -> ExecutionResponse:
        preview_spec = dict(request.preview_spec)
        probe_id = str(preview_spec.get("probe_id", "preview"))
        target_axes = list(preview_spec.get("target_axes", []))
        preserve_axes = list(preview_spec.get("preserve_axes", []))
        return ExecutionResponse(
            response_id=f"local-preview-{probe_id}",
            execution_kind="preview",
            output_payload={
                "preview_id": f"preview-{probe_id}",
                "probe_id": probe_id,
                "source_kind": preview_spec.get("source_kind", ""),
            },
            summary_text=f"Local workflow facade produced preview for probe={probe_id}.",
            changed_axes=target_axes,
            preserved_axes=preserve_axes,
            backend_artifacts={"artifact_uri": f"memory://local-workflow/preview/{probe_id}"},
            backend_metadata={"result_type": "live_preview_result", "backend": "local_workflow_facade"},
            comparison_notes=[f"preview_source={preview_spec.get('source_kind', '')}"],
        )

    @staticmethod
    def _run_commit(request: CommitExecutionRequest) -> ExecutionResponse:
        patch_spec = dict(request.patch_spec)
        patch_id = str(patch_spec.get("patch_id", "commit"))
        target_axes = list(patch_spec.get("target_axes", []))
        preserve_axes = list(patch_spec.get("preserve_axes", []))
        return ExecutionResponse(
            response_id=f"local-commit-{patch_id}",
            execution_kind="commit",
            output_payload={
                "image_id": f"commit-{patch_id}",
                "patch_id": patch_id,
                "target_fields": list(patch_spec.get("target_fields", [])),
            },
            summary_text=f"Local workflow facade committed patch={patch_id}.",
            changed_axes=target_axes,
            preserved_axes=preserve_axes,
            backend_artifacts={"artifact_uri": f"memory://local-workflow/commit/{patch_id}"},
            backend_metadata={"result_type": "live_committed_result", "backend": "local_workflow_facade"},
            comparison_notes=[f"commit_rationale={patch_spec.get('rationale', '')}"],
        )
