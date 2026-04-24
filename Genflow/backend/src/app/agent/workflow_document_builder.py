from __future__ import annotations

from app.agent.memory import AgentSessionState
from app.agent.workflow_document_models import (
    SurrogateWorkflowDocument,
    SurrogateWorkflowEdge,
    SurrogateWorkflowNode,
    SurrogateWorkflowRegion,
)


def build_surrogate_workflow_document(
    session: AgentSessionState,
    execution_kind: str = "",
    preview: bool = False,
) -> SurrogateWorkflowDocument:
    workflow_id = session.workflow_id or f"workflow-{session.session_id}"
    effective_execution_kind = execution_kind or "idle"
    region_label = "repair_region" if session.selected_probe.probe_id or session.accepted_patch.patch_id else "initial_region"
    entry_node_ids = _build_entry_node_ids(session)
    exit_node_ids = _build_exit_node_ids(session)

    nodes = [
        SurrogateWorkflowNode(
            node_id="intent.prompt",
            node_kind="surrogate_input",
            role="input",
            label="Prompt Input",
            metadata={"value_present": bool(session.current_schema.prompt)},
        ),
        SurrogateWorkflowNode(
            node_id="render.model",
            node_kind="surrogate_compute",
            role="compute",
            label="Model Selection",
            metadata={"model": session.current_schema.model},
        ),
        SurrogateWorkflowNode(
            node_id="result.output",
            node_kind="surrogate_output",
            role="output",
            label="Result Output",
            metadata={"result_id": session.current_result_id},
        ),
    ]
    edges = [
        SurrogateWorkflowEdge(
            edge_id="intent.prompt->render.model",
            source_node_id="intent.prompt",
            target_node_id="render.model",
            edge_kind="prompt_conditions_model",
            metadata={"hint": "prompt_conditions_model"},
        ),
        SurrogateWorkflowEdge(
            edge_id="render.model->result.output",
            source_node_id="render.model",
            target_node_id="result.output",
            edge_kind="model_emits_result",
            metadata={"hint": "model_emits_result"},
        ),
    ]

    if session.selected_gallery_index is not None:
        nodes.append(
            SurrogateWorkflowNode(
                node_id="reference.bundle",
                node_kind="surrogate_reference",
                role="reference",
                label="Reference Bundle",
                metadata={"gallery_index": session.selected_gallery_index},
            )
        )
        edges.append(
            SurrogateWorkflowEdge(
                edge_id="reference.bundle->intent.prompt",
                source_node_id="reference.bundle",
                target_node_id="intent.prompt",
                edge_kind="references_inform_prompt",
                metadata={"hint": "references_inform_prompt"},
            )
        )

    if session.selected_probe.probe_id:
        probe_node_id = f"probe.{session.selected_probe.probe_id}"
        nodes.append(
            SurrogateWorkflowNode(
                node_id=probe_node_id,
                node_kind="surrogate_probe",
                role="repair_probe",
                label="Selected Probe",
                metadata={"target_axes": list(session.selected_probe.target_axes)},
            )
        )
        edges.append(
            SurrogateWorkflowEdge(
                edge_id=f"{probe_node_id}->render.model",
                source_node_id=probe_node_id,
                target_node_id="render.model",
                edge_kind="probe_targets_render",
                metadata={"hint": "probe_targets_render"},
            )
        )

    if session.accepted_patch.patch_id:
        patch_node_id = f"patch.{session.accepted_patch.patch_id}"
        nodes.append(
            SurrogateWorkflowNode(
                node_id=patch_node_id,
                node_kind="surrogate_patch",
                role="repair_patch",
                label="Accepted Patch",
                metadata={"target_fields": list(session.accepted_patch.target_fields)},
            )
        )
        edges.append(
            SurrogateWorkflowEdge(
                edge_id=f"{patch_node_id}->render.model",
                source_node_id=patch_node_id,
                target_node_id="render.model",
                edge_kind="patch_updates_render",
                metadata={"hint": "patch_updates_render"},
            )
        )
        edges.append(
            SurrogateWorkflowEdge(
                edge_id=f"{patch_node_id}->result.output",
                source_node_id=patch_node_id,
                target_node_id="result.output",
                edge_kind="patch_updates_result",
                metadata={"hint": "patch_updates_result"},
            )
        )

    region = SurrogateWorkflowRegion(
        region_id=f"{workflow_id}:{effective_execution_kind}",
        region_label=region_label,
        region_kind=region_label,
        node_ids=[node.node_id for node in nodes],
        entry_node_ids=list(entry_node_ids),
        exit_node_ids=list(exit_node_ids),
        metadata={
            "execution_kind": execution_kind,
            "preview": preview,
            "has_feedback": bool(session.latest_feedback),
        },
    )
    return SurrogateWorkflowDocument(
        document_id=f"{workflow_id}:{effective_execution_kind}",
        workflow_id=workflow_id,
        workflow_kind=session.workflow_identity.workflow_kind or "surrogate_workflow",
        execution_kind=execution_kind,
        preview=preview,
        backend_kind=str(session.workflow_metadata.get("backend_kind", "")),
        workflow_profile=str(session.workflow_metadata.get("workflow_profile", "")),
        nodes=nodes,
        edges=edges,
        regions=[region],
        entry_node_ids=list(entry_node_ids),
        exit_node_ids=list(exit_node_ids),
        metadata={
            "region_label": region_label,
            "graph_regions": [region_label],
            "selected_probe_id": session.selected_probe.probe_id,
            "accepted_patch_id": session.accepted_patch.patch_id,
            "has_feedback": bool(session.latest_feedback),
        },
    )


def _build_entry_node_ids(session: AgentSessionState) -> list[str]:
    if session.selected_gallery_index is not None:
        return ["reference.bundle", "intent.prompt"]
    return ["intent.prompt"]


def _build_exit_node_ids(session: AgentSessionState) -> list[str]:
    if session.accepted_patch.patch_id:
        return [f"patch.{session.accepted_patch.patch_id}", "result.output"]
    if session.selected_probe.probe_id:
        return [f"probe.{session.selected_probe.probe_id}", "result.output"]
    return ["result.output"]
