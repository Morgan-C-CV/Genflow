from __future__ import annotations

from app.agent.memory import AgentSessionState
from app.agent.workflow_descriptor_models import (
    SurrogateBenchmarkComparisonFootprint,
    SurrogateExecutionDescriptor,
    SurrogateRepairDescriptor,
    SurrogateWorkflowDescriptor,
)
from app.agent.workflow_execution_source_models import (
    WorkflowCommitSource,
    WorkflowExecutionSource,
    WorkflowPreviewSource,
)


def build_surrogate_workflow_descriptor(
    session: AgentSessionState,
    execution_kind: str = "",
    preview: bool = False,
) -> SurrogateWorkflowDescriptor:
    workflow_id = session.workflow_id or f"workflow-{session.session_id}"
    return SurrogateWorkflowDescriptor(
        workflow_id=workflow_id,
        workflow_kind=session.workflow_identity.workflow_kind or "surrogate_workflow",
        schema_prompt=session.current_schema.prompt,
        schema_model=session.current_schema.model,
        schema_negative_prompt=session.current_schema.negative_prompt,
        schema_sampler=session.current_schema.sampler,
        schema_style=list(session.current_schema.style),
        schema_lora=list(session.current_schema.lora),
        selected_gallery_index=session.selected_gallery_index,
        selected_reference_ids=list(session.selected_reference_ids),
        reference_bundle=dict(session.selected_reference_bundle),
        execution=SurrogateExecutionDescriptor(
            execution_kind=execution_kind,
            preview=preview,
            backend_kind=str(session.workflow_metadata.get("backend_kind", "")),
            workflow_profile=str(session.workflow_metadata.get("workflow_profile", "")),
            current_result_id=session.current_result_id,
            metadata={
                "has_schema": bool(session.current_schema_raw),
                "reference_count": len(session.selected_reference_ids),
            },
        ),
        repair=SurrogateRepairDescriptor(
            has_feedback=bool(session.latest_feedback),
            feedback_count=len(session.feedback_history),
            dissatisfaction_axes=list(session.dissatisfaction_axes),
            preserve_constraints=list(session.preserve_constraints),
            selected_probe_id=session.selected_probe.probe_id,
            probe_target_axes=list(session.selected_probe.target_axes),
            probe_preserve_axes=list(session.selected_probe.preserve_axes),
            accepted_patch_id=session.accepted_patch.patch_id,
            patch_target_fields=list(session.accepted_patch.target_fields),
            current_uncertainty_estimate=session.current_uncertainty_estimate,
            metadata={
                "latest_feedback": session.latest_feedback,
            },
        ),
        benchmark_comparison=SurrogateBenchmarkComparisonFootprint(
            benchmark_source=str(
                session.benchmark_comparison_summary.metadata.get("benchmark_source", "")
            ),
            compared_anchor_ids=list(session.benchmark_comparison_summary.compared_anchor_ids),
            compared_candidate_ids=list(session.benchmark_comparison_summary.compared_candidate_ids),
            focus_axes=list(session.benchmark_comparison_summary.focus_axes),
            preserve_axes=list(session.benchmark_comparison_summary.preserve_axes),
            confidence_hint=session.benchmark_comparison_summary.confidence_hint,
        ),
        metadata={
            "selected_reference_ids": list(session.selected_reference_ids),
            "has_reference_bundle": bool(session.selected_reference_bundle),
            "workflow_version": session.workflow_identity.workflow_version,
        },
    )


def build_surrogate_workflow_descriptor_from_execution_source(
    source: WorkflowExecutionSource,
) -> SurrogateWorkflowDescriptor:
    selected_probe = (
        source.selected_probe if isinstance(source, (WorkflowPreviewSource, WorkflowCommitSource)) else None
    )
    accepted_patch = source.accepted_patch if isinstance(source, WorkflowCommitSource) else None
    workflow_id = source.workflow_id or "workflow-execution-source"
    return SurrogateWorkflowDescriptor(
        workflow_id=workflow_id,
        workflow_kind=source.workflow_kind or "workflow_native_surrogate",
        schema_prompt=source.schema.prompt,
        schema_model=source.schema.model,
        schema_negative_prompt=source.schema.negative_prompt,
        schema_sampler=source.schema.sampler,
        schema_style=list(source.schema.style),
        schema_lora=list(source.schema.lora),
        selected_gallery_index=source.selected_gallery_index,
        selected_reference_ids=list(source.selected_reference_ids),
        reference_bundle=dict(source.selected_reference_bundle),
        execution=SurrogateExecutionDescriptor(
            execution_kind=source.execution_kind,
            preview=source.preview,
            backend_kind=source.backend_kind,
            workflow_profile=source.workflow_profile,
            current_result_id=source.current_result_id,
            metadata={
                "has_schema": bool(source.schema.prompt or source.schema.model or source.schema.raw_fields),
                "reference_count": len(source.selected_reference_ids),
            },
        ),
        repair=SurrogateRepairDescriptor(
            has_feedback=bool(source.latest_feedback),
            feedback_count=source.feedback_count,
            dissatisfaction_axes=list(source.dissatisfaction_axes),
            preserve_constraints=list(source.preserve_constraints),
            selected_probe_id=selected_probe.probe_id if selected_probe else "",
            probe_target_axes=list(selected_probe.target_axes) if selected_probe else [],
            probe_preserve_axes=list(selected_probe.preserve_axes) if selected_probe else [],
            accepted_patch_id=accepted_patch.patch_id if accepted_patch else "",
            patch_target_fields=list(accepted_patch.target_fields) if accepted_patch else [],
            current_uncertainty_estimate=source.current_uncertainty_estimate,
            metadata={
                "latest_feedback": source.latest_feedback,
            },
        ),
        benchmark_comparison=SurrogateBenchmarkComparisonFootprint(
            benchmark_source=source.benchmark_source,
            compared_anchor_ids=list(source.compared_anchor_ids),
            compared_candidate_ids=list(source.compared_candidate_ids),
            focus_axes=list(source.benchmark_focus_axes),
            preserve_axes=list(source.benchmark_preserve_axes),
            confidence_hint=source.benchmark_confidence_hint,
        ),
        metadata={
            "selected_reference_ids": list(source.selected_reference_ids),
            "has_reference_bundle": bool(source.selected_reference_bundle),
            "workflow_version": source.workflow_version,
            **dict(source.metadata),
        },
    )
