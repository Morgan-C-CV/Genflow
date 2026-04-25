from __future__ import annotations

from app.agent.live_execution_models import (
    CommitExecutionRequest,
    ExecutionRequest,
    ExecutionResponse,
    PreviewExecutionRequest,
)


class LocalWorkflowFacade:
    def run(self, execution_kind: str, request: ExecutionRequest) -> ExecutionResponse:
        self._validate_workflow_payload(request.workflow_payload)
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
    def _validate_workflow_payload(workflow_payload: dict) -> None:
        if not isinstance(workflow_payload, dict):
            raise ValueError("workflow_payload must be a dict.")
        if not isinstance(workflow_payload.get("nodes"), list):
            raise ValueError("workflow_payload must include a nodes list.")
        if not isinstance(workflow_payload.get("edges"), list):
            raise ValueError("workflow_payload must include an edges list.")
        if not isinstance(workflow_payload.get("execution_config"), dict):
            raise ValueError("workflow_payload must include execution_config.")

    @staticmethod
    def _run_preview(request: PreviewExecutionRequest) -> ExecutionResponse:
        preview_spec = dict(request.preview_spec)
        graph_patch_spec = dict(preview_spec.get("graph_patch_spec", {}))
        LocalWorkflowFacade._validate_graph_patch_spec(graph_patch_spec, "preview")
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
            backend_metadata={
                "result_type": "live_preview_result",
                "backend": "local_workflow_facade",
                "graph_patch_id": graph_patch_spec.get("patch_id", ""),
            },
            comparison_notes=[
                f"preview_source={preview_spec.get('source_kind', '')}",
                f"graph_patch_id={graph_patch_spec.get('patch_id', '')}",
            ],
        )

    @staticmethod
    def _run_commit(request: CommitExecutionRequest) -> ExecutionResponse:
        patch_spec = dict(request.patch_spec)
        graph_patch_spec = dict(patch_spec.get("graph_patch_spec", {}))
        primary_commit_plan = dict(patch_spec.get("primary_commit_plan", {}))
        backend_execution_mode = str(patch_spec.get("backend_execution_mode", ""))
        commit_source_payload = dict(patch_spec.get("commit_source_payload", {}))
        LocalWorkflowFacade._validate_graph_patch_spec(graph_patch_spec, "commit")
        patch_id = str(patch_spec.get("patch_id", "commit"))
        target_axes = list(patch_spec.get("target_axes", []))
        preserve_axes = list(patch_spec.get("preserve_axes", []))
        primary_plan_kind = str(primary_commit_plan.get("plan_kind", "schema_primary"))
        implementation_mode = str(
            commit_source_payload.get("commit_execution_implementation_mode", "schema_compatible_execution")
        )
        execution_behavior_branch = (
            "graph_primary_execution_branch"
            if primary_plan_kind == "graph_primary"
            else "schema_primary_execution_branch"
        )
        return ExecutionResponse(
            response_id=f"local-commit-{patch_id}",
            execution_kind="commit",
            output_payload={
                "image_id": f"commit-{patch_id}",
                "patch_id": patch_id,
                "target_fields": list(patch_spec.get("target_fields", [])),
                "graph_native_artifact_input_received": bool(
                    commit_source_payload.get("selected_workflow_graph_patch_id", "")
                ),
                "request_primary_plan_kind": primary_plan_kind,
                "commit_execution_implementation_mode": implementation_mode,
                "backend_execution_mode": backend_execution_mode,
                "execution_behavior_branch": execution_behavior_branch,
                "graph_driven_node_count": len(graph_patch_spec.get("node_patches", []))
                if primary_plan_kind == "graph_primary"
                else 0,
            },
            summary_text=(
                f"Local workflow facade ran graph-primary execution branch for patch={patch_id}."
                if primary_plan_kind == "graph_primary"
                else f"Local workflow facade ran schema-primary execution branch for patch={patch_id}."
            ),
            changed_axes=target_axes,
            preserved_axes=preserve_axes,
            backend_artifacts={"artifact_uri": f"memory://local-workflow/commit/{patch_id}"},
            backend_metadata={
                "result_type": "live_committed_result",
                "backend": "local_workflow_facade",
                "graph_patch_id": graph_patch_spec.get("patch_id", ""),
                "commit_execution_mode": commit_source_payload.get("commit_execution_mode", ""),
                "commit_execution_authority": commit_source_payload.get("commit_execution_authority", ""),
                "request_primary_plan_kind": primary_plan_kind,
                "commit_execution_implementation_mode": implementation_mode,
                "backend_execution_mode": backend_execution_mode,
                "execution_behavior_branch": execution_behavior_branch,
                "graph_primary_behavior_applied": primary_plan_kind == "graph_primary",
                "preferred_commit_source": commit_source_payload.get("preferred_commit_source", ""),
                "graph_native_artifact_input_received": bool(
                    commit_source_payload.get("selected_workflow_graph_patch_id", "")
                ),
                "selected_workflow_graph_patch_id": commit_source_payload.get(
                    "selected_workflow_graph_patch_id", ""
                ),
                "top_schema_patch_id": commit_source_payload.get("top_schema_patch_id", ""),
                "top_graph_patch_candidate_id": commit_source_payload.get("top_graph_patch_candidate_id", ""),
            },
            comparison_notes=[
                f"commit_rationale={patch_spec.get('rationale', '')}",
                f"graph_patch_id={graph_patch_spec.get('patch_id', '')}",
                f"request_primary_plan_kind={primary_plan_kind}",
                f"commit_execution_implementation_mode={implementation_mode}",
                f"backend_execution_mode={backend_execution_mode}",
                f"execution_behavior_branch={execution_behavior_branch}",
                f"commit_execution_mode={commit_source_payload.get('commit_execution_mode', '')}",
                f"commit_execution_authority={commit_source_payload.get('commit_execution_authority', '')}",
                f"preferred_commit_source={commit_source_payload.get('preferred_commit_source', '')}",
                "graph_native_artifact_input_received="
                f"{bool(commit_source_payload.get('selected_workflow_graph_patch_id', ''))}",
                f"selected_workflow_graph_patch_id={commit_source_payload.get('selected_workflow_graph_patch_id', '')}",
            ],
        )

    @staticmethod
    def _validate_graph_patch_spec(graph_patch_spec: dict, execution_kind: str) -> None:
        if not isinstance(graph_patch_spec, dict):
            raise ValueError(f"{execution_kind} request must include graph_patch_spec.")
        if not graph_patch_spec.get("patch_id"):
            raise ValueError(f"{execution_kind} request graph_patch_spec must include patch_id.")
        if not isinstance(graph_patch_spec.get("node_patches"), list):
            raise ValueError(f"{execution_kind} request graph_patch_spec must include node_patches.")
