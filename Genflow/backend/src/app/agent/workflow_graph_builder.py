from __future__ import annotations

from app.agent.memory import AgentSessionState
from app.agent.workflow_runtime_models import (
    WorkflowGraphPlaceholder,
    WorkflowNodeRef,
    WorkflowTopologySlice,
)


def build_surrogate_workflow_graph(
    session: AgentSessionState,
    execution_kind: str = "",
    preview: bool = False,
) -> tuple[WorkflowGraphPlaceholder, dict]:
    graph_id = session.workflow_id or f"workflow-{session.session_id}"
    region_label = "repair_region" if session.selected_probe.probe_id or session.accepted_patch.patch_id else "initial_region"
    entry_node_ids = _build_graph_entry_node_ids(session)
    exit_node_ids = _build_graph_exit_node_ids(session)
    node_refs = [
        WorkflowNodeRef(
            node_id="intent.prompt",
            node_kind="surrogate_input",
            role="input",
            label="Prompt Input",
            upstream_ids=[],
            downstream_ids=["render.model"],
            metadata={"value_present": bool(session.current_schema.prompt)},
        ),
        WorkflowNodeRef(
            node_id="render.model",
            node_kind="surrogate_compute",
            role="compute",
            label="Model Selection",
            upstream_ids=["intent.prompt"],
            downstream_ids=["result.output"],
            metadata={"model": session.current_schema.model},
        ),
        WorkflowNodeRef(
            node_id="result.output",
            node_kind="surrogate_output",
            role="output",
            label="Result Output",
            upstream_ids=["render.model"],
            downstream_ids=[],
            metadata={"result_id": session.current_result_id},
        ),
    ]
    if session.selected_gallery_index is not None:
        node_refs.append(
            WorkflowNodeRef(
                node_id="reference.bundle",
                node_kind="surrogate_reference",
                role="reference",
                label="Reference Bundle",
                upstream_ids=[],
                downstream_ids=["intent.prompt"],
                metadata={"gallery_index": session.selected_gallery_index},
            )
        )
    if session.selected_probe.probe_id:
        node_refs.append(
            WorkflowNodeRef(
                node_id=f"probe.{session.selected_probe.probe_id}",
                node_kind="surrogate_probe",
                role="repair_probe",
                label="Selected Probe",
                upstream_ids=["render.model"],
                downstream_ids=["render.model"],
                metadata={"target_axes": list(session.selected_probe.target_axes)},
            )
        )
    if session.accepted_patch.patch_id:
        node_refs.append(
            WorkflowNodeRef(
                node_id=f"patch.{session.accepted_patch.patch_id}",
                node_kind="surrogate_patch",
                role="repair_patch",
                label="Accepted Patch",
                upstream_ids=["render.model"],
                downstream_ids=["result.output"],
                metadata={"target_fields": list(session.accepted_patch.target_fields)},
            )
        )

    adjacency_hints = [
        {"from": "intent.prompt", "to": "render.model", "hint": "prompt_conditions_model"},
    ]
    if session.selected_gallery_index is not None:
        adjacency_hints.append({"from": "reference.bundle", "to": "intent.prompt", "hint": "references_inform_prompt"})
    if session.selected_probe.probe_id:
        adjacency_hints.append(
            {"from": f"probe.{session.selected_probe.probe_id}", "to": "render.model", "hint": "probe_targets_render"}
        )
    if session.accepted_patch.patch_id:
        adjacency_hints.append(
            {"from": f"patch.{session.accepted_patch.patch_id}", "to": "render.model", "hint": "patch_updates_render"}
        )

    topology_slice = WorkflowTopologySlice(
        slice_id=f"{graph_id}:{execution_kind or 'idle'}",
        region_label=region_label,
        slice_kind="repair_region" if region_label == "repair_region" else "initial_region",
        node_refs=list(node_refs),
        entry_node_ids=list(entry_node_ids),
        exit_node_ids=list(exit_node_ids),
        edge_hints=list(adjacency_hints),
        scope_partitions=[scope.scope_id for scope in session.editable_scopes + session.protected_scopes],
        metadata={
            "execution_kind": execution_kind,
            "preview": preview,
            "has_feedback": bool(session.latest_feedback),
        },
    )
    placeholder = WorkflowGraphPlaceholder(
        graph_id=graph_id,
        graph_kind="surrogate_topology",
        entry_node_ids=list(entry_node_ids),
        exit_node_ids=list(exit_node_ids),
        node_refs=list(node_refs),
        topology_slices=[topology_slice],
        adjacency_hints=list(adjacency_hints),
        metadata={
            "backend_kind": session.workflow_metadata.get("backend_kind", ""),
            "workflow_profile": session.workflow_metadata.get("workflow_profile", ""),
            "execution_kind": execution_kind,
            "preview": preview,
            "region_label": region_label,
            "graph_regions": [region_label],
        },
    )
    topology_hints = {
        "region_label": region_label,
        "node_count": len(node_refs),
        "adjacency_hint_count": len(adjacency_hints),
        "selected_probe_id": session.selected_probe.probe_id,
        "accepted_patch_id": session.accepted_patch.patch_id,
        "has_feedback": bool(session.latest_feedback),
        "entry_node_ids": list(entry_node_ids),
        "exit_node_ids": list(exit_node_ids),
    }
    return placeholder, topology_hints


def _build_graph_entry_node_ids(session: AgentSessionState) -> list[str]:
    if session.selected_gallery_index is not None:
        return ["reference.bundle", "intent.prompt"]
    return ["intent.prompt"]


def _build_graph_exit_node_ids(session: AgentSessionState) -> list[str]:
    if session.accepted_patch.patch_id:
        return [f"patch.{session.accepted_patch.patch_id}", "result.output"]
    if session.selected_probe.probe_id:
        return [f"probe.{session.selected_probe.probe_id}", "result.output"]
    return ["result.output"]
