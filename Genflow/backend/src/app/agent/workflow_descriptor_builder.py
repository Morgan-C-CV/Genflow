from __future__ import annotations

from app.agent.memory import AgentSessionState
from app.agent.workflow_descriptor_models import (
    SurrogateExecutionDescriptor,
    SurrogateRepairDescriptor,
    SurrogateWorkflowDescriptor,
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
        metadata={
            "selected_reference_ids": list(session.selected_reference_ids),
            "has_reference_bundle": bool(session.selected_reference_bundle),
        },
    )
