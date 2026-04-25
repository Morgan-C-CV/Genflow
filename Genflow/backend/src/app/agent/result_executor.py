from __future__ import annotations

from typing import Callable, Dict, Optional
from uuid import uuid4

from app.agent.execution_adapter import ExecutionAdapter
from app.agent.runtime_models import (
    CommittedPatch,
    NormalizedSchema,
    PreviewProbe,
    PreviewResult,
    ResultPayload,
    ResultSummary,
)
from app.agent.workflow_graph_patch_models import WorkflowGraphPatch


class ResultExecutor(ExecutionAdapter):
    def __init__(self, id_factory: Optional[Callable[[], str]] = None):
        self._id_factory = id_factory or (lambda: str(uuid4()))

    def produce_initial_result(
        self,
        schema: NormalizedSchema,
        reference_bundle: Optional[Dict[str, object]] = None,
    ) -> tuple[ResultPayload, ResultSummary]:
        reference_bundle = reference_bundle or {}
        reference_count = len(reference_bundle.get("references", []))
        result_id = self._id_factory()
        payload = ResultPayload(
            result_id=result_id,
            result_type="mock_initial_result",
            content={
                "prompt": schema.prompt,
                "negative_prompt": schema.negative_prompt,
                "model": schema.model,
                "sampler": schema.sampler,
                "style": list(schema.style),
                "lora": list(schema.lora),
                "reference_count": reference_count,
            },
            artifacts={
                "render_mode": "mock",
                "schema_snapshot": {
                    "model": schema.model,
                    "sampler": schema.sampler,
                    "seed": schema.seed,
                },
            },
        )
        summary = ResultSummary(
            summary_text=(
                f"Mock initial result generated with model={schema.model or 'unknown'}, "
                f"sampler={schema.sampler or 'unknown'}, references={reference_count}."
            ),
            changed_axes=["initial_generation"],
            preserved_axes=[],
            notes=[
                f"prompt_length={len(schema.prompt)}",
                f"style_count={len(schema.style)}",
                f"lora_count={len(schema.lora)}",
            ],
        )
        return payload, summary

    def execute_preview_probe(
        self,
        schema: NormalizedSchema,
        probe: PreviewProbe,
    ) -> PreviewResult:
        payload = ResultPayload(
            result_id=self._id_factory(),
            result_type="mock_preview_result",
            content={
                "prompt": schema.prompt,
                "probe_id": probe.probe_id,
                "target_axes": list(probe.target_axes),
                "source_kind": probe.source_kind,
                "preview_execution_spec": dict(probe.preview_execution_spec),
            },
            artifacts={
                "render_mode": "mock_preview",
                "preserve_axes": list(probe.preserve_axes),
            },
        )
        summary = ResultSummary(
            summary_text=(
                f"Mock preview for probe={probe.probe_id}, "
                f"target_axes={','.join(probe.target_axes) or 'none'}, "
                f"source_kind={probe.source_kind or 'unknown'}."
            ),
            changed_axes=list(probe.target_axes),
            preserved_axes=list(probe.preserve_axes),
            notes=[
                f"patch_family={probe.preview_execution_spec.get('patch_family', '')}",
                f"reference_anchor={probe.preview_execution_spec.get('reference_anchor', '')}",
            ],
        )
        return PreviewResult(
            probe_id=probe.probe_id,
            summary=summary,
            payload=payload,
            comparison_notes=[
                f"preview_source={probe.source_kind}",
                f"target_axes={','.join(probe.target_axes)}",
            ],
        )

    def execute_committed_patch(
        self,
        schema: NormalizedSchema,
        patch: CommittedPatch,
        graph_patch: WorkflowGraphPatch | None = None,
        commit_execution_mode: str = "",
        commit_execution_authority: str = "",
        commit_execution_implementation_mode: str = "",
    ) -> tuple[ResultPayload, ResultSummary]:
        changed_axes = list(patch.target_axes or patch.target_fields)
        graph_patch = graph_patch or WorkflowGraphPatch()
        effective_mode = commit_execution_mode or "schema_execution_fallback"
        effective_authority = commit_execution_authority or "schema_authoritative"
        implementation_mode = commit_execution_implementation_mode or "schema_compatible_execution"
        primary_plan_kind = "graph_primary" if effective_authority == "graph_authoritative" else "schema_primary"
        backend_graph_commit_payload_supplied = bool(graph_patch.patch_id)
        requested_backend_execution_mode = (
            "graph_primary_backend_execution"
            if implementation_mode == "graph_primary_execution"
            else "schema_compatible_backend_execution"
        )
        backend_graph_primary_capable = bool(graph_patch.patch_id and graph_patch.node_patches)
        accepted_backend_execution_mode = (
            "schema_compatible_backend_execution"
            if (
                requested_backend_execution_mode == "graph_primary_backend_execution"
                and not backend_graph_primary_capable
            )
            else requested_backend_execution_mode
        )
        realized_backend_execution_mode = (
            "schema_compatible_backend_execution"
            if (
                accepted_backend_execution_mode == "graph_primary_backend_execution"
                and not (graph_patch.edge_patches or graph_patch.region_patches)
            )
            else accepted_backend_execution_mode
        )
        execution_behavior_branch = (
            "graph_primary_execution_branch"
            if realized_backend_execution_mode == "graph_primary_backend_execution"
            else "schema_primary_execution_branch"
        )
        backend_graph_commit_payload_consumed = (
            backend_graph_commit_payload_supplied and primary_plan_kind == "graph_primary"
        )
        backend_graph_native_execution_realized = (
            realized_backend_execution_mode == "graph_primary_backend_execution"
        )
        payload = ResultPayload(
            result_id=self._id_factory(),
            result_type="mock_committed_result",
            content={
                "prompt": schema.prompt,
                "patch_id": patch.patch_id,
                "target_fields": list(patch.target_fields),
                "target_axes": changed_axes,
                "rationale": patch.rationale,
                "graph_patch_input_id": graph_patch.patch_id,
                "commit_execution_authority": effective_authority,
                "request_primary_plan_kind": primary_plan_kind,
                "commit_execution_implementation_mode": implementation_mode,
                "requested_backend_execution_mode": requested_backend_execution_mode,
                "backend_graph_primary_capable": backend_graph_primary_capable,
                "backend_graph_commit_payload_supplied": backend_graph_commit_payload_supplied,
                "backend_graph_commit_payload_consumed": backend_graph_commit_payload_consumed,
                "backend_graph_native_execution_realized": backend_graph_native_execution_realized,
                "accepted_backend_execution_mode": accepted_backend_execution_mode,
                "realized_backend_execution_mode": realized_backend_execution_mode,
                "execution_behavior_branch": execution_behavior_branch,
                "graph_driven_node_count": (
                    len(graph_patch.node_patches)
                    if realized_backend_execution_mode == "graph_primary_backend_execution"
                    else 0
                ),
            },
            artifacts={
                "render_mode": "mock_commit",
                "changes": dict(patch.changes),
                "preserve_axes": list(patch.preserve_axes),
                "backend_metadata": {
                    "graph_patch_id": graph_patch.patch_id,
                    "commit_execution_mode": effective_mode,
                    "commit_execution_authority": effective_authority,
                    "request_primary_plan_kind": primary_plan_kind,
                    "commit_execution_implementation_mode": implementation_mode,
                    "backend_graph_primary_capable": backend_graph_primary_capable,
                    "backend_graph_commit_payload_supplied": backend_graph_commit_payload_supplied,
                    "backend_graph_commit_payload_consumed": backend_graph_commit_payload_consumed,
                    "backend_graph_native_execution_realized": backend_graph_native_execution_realized,
                    "backend_execution_mode": accepted_backend_execution_mode,
                    "accepted_backend_execution_mode": accepted_backend_execution_mode,
                    "realized_backend_execution_mode": realized_backend_execution_mode,
                    "execution_behavior_branch": execution_behavior_branch,
                    "graph_primary_behavior_applied": (
                        realized_backend_execution_mode == "graph_primary_backend_execution"
                    ),
                    "graph_native_artifact_input_received": bool(graph_patch.patch_id),
                },
            },
        )
        summary = ResultSummary(
            summary_text=(
                (
                    f"Mock graph-primary execution branch for patch={patch.patch_id}, "
                    f"graph_patch={graph_patch.patch_id or 'none'}, "
                    f"graph_nodes={len(graph_patch.node_patches)}, "
                    f"implementation_mode={implementation_mode}."
                    if realized_backend_execution_mode == "graph_primary_backend_execution"
                    else f"Mock schema-primary execution branch for patch={patch.patch_id}, "
                    f"target_fields={','.join(patch.target_fields) or 'none'}, "
                    f"target_axes={','.join(changed_axes) or 'none'}, "
                    f"implementation_mode={implementation_mode}."
                )
            ),
            changed_axes=changed_axes,
            preserved_axes=list(patch.preserve_axes),
            notes=[
                patch.rationale,
                f"change_keys={','.join(patch.changes.keys())}",
                f"graph_native_artifact_input_received={bool(graph_patch.patch_id)}",
                f"commit_execution_authority={effective_authority}",
                f"request_primary_plan_kind={primary_plan_kind}",
                f"commit_execution_implementation_mode={implementation_mode}",
                f"backend_graph_primary_capable={backend_graph_primary_capable}",
                f"backend_graph_commit_payload_supplied={backend_graph_commit_payload_supplied}",
                f"backend_graph_commit_payload_consumed={backend_graph_commit_payload_consumed}",
                f"backend_graph_native_execution_realized={backend_graph_native_execution_realized}",
                f"requested_backend_execution_mode={requested_backend_execution_mode}",
                f"accepted_backend_execution_mode={accepted_backend_execution_mode}",
                f"realized_backend_execution_mode={realized_backend_execution_mode}",
                f"execution_behavior_branch={execution_behavior_branch}",
            ] if patch.rationale else [
                f"change_keys={','.join(patch.changes.keys())}",
                f"graph_native_artifact_input_received={bool(graph_patch.patch_id)}",
                f"commit_execution_authority={effective_authority}",
                f"request_primary_plan_kind={primary_plan_kind}",
                f"commit_execution_implementation_mode={implementation_mode}",
                f"backend_graph_primary_capable={backend_graph_primary_capable}",
                f"backend_graph_commit_payload_supplied={backend_graph_commit_payload_supplied}",
                f"backend_graph_commit_payload_consumed={backend_graph_commit_payload_consumed}",
                f"backend_graph_native_execution_realized={backend_graph_native_execution_realized}",
                f"requested_backend_execution_mode={requested_backend_execution_mode}",
                f"accepted_backend_execution_mode={accepted_backend_execution_mode}",
                f"realized_backend_execution_mode={realized_backend_execution_mode}",
                f"execution_behavior_branch={execution_behavior_branch}",
            ],
        )
        return payload, summary
