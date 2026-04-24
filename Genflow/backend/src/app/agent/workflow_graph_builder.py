from __future__ import annotations

from app.agent.memory import AgentSessionState
from app.agent.workflow_document_builder import build_surrogate_workflow_document
from app.agent.workflow_document_models import SurrogateWorkflowDocument
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
    document = build_surrogate_workflow_document(
        session=session,
        execution_kind=execution_kind,
        preview=preview,
    )
    return build_surrogate_workflow_graph_from_document(
        document=document,
        scope_partitions=[scope.scope_id for scope in session.editable_scopes + session.protected_scopes],
    )


def build_surrogate_workflow_graph_from_document(
    document: SurrogateWorkflowDocument,
    scope_partitions: list[str] | None = None,
) -> tuple[WorkflowGraphPlaceholder, dict]:
    scope_partitions = scope_partitions or []
    adjacency_hints = [
        {
            "from": edge.source_node_id,
            "to": edge.target_node_id,
            "hint": str(edge.metadata.get("hint", edge.edge_kind)),
        }
        for edge in document.edges
    ]
    upstream_map: dict[str, list[str]] = {}
    downstream_map: dict[str, list[str]] = {}
    for edge in document.edges:
        upstream_map.setdefault(edge.target_node_id, []).append(edge.source_node_id)
        downstream_map.setdefault(edge.source_node_id, []).append(edge.target_node_id)

    node_refs = [
        WorkflowNodeRef(
            node_id=node.node_id,
            node_kind=node.node_kind,
            role=node.role,
            label=node.label,
            upstream_ids=list(upstream_map.get(node.node_id, [])),
            downstream_ids=list(downstream_map.get(node.node_id, [])),
            metadata=dict(node.metadata),
        )
        for node in document.nodes
    ]
    topology_slices = [
        WorkflowTopologySlice(
            slice_id=region.region_id,
            region_label=region.region_label,
            slice_kind=region.region_kind,
            node_refs=[node_ref for node_ref in node_refs if node_ref.node_id in set(region.node_ids)],
            entry_node_ids=list(region.entry_node_ids),
            exit_node_ids=list(region.exit_node_ids),
            edge_hints=[
                hint
                for hint in adjacency_hints
                if hint["from"] in set(region.node_ids) and hint["to"] in set(region.node_ids)
            ],
            scope_partitions=list(scope_partitions),
            metadata=dict(region.metadata),
        )
        for region in document.regions
    ]
    placeholder = WorkflowGraphPlaceholder(
        graph_id=document.workflow_id,
        graph_kind="surrogate_topology",
        entry_node_ids=list(document.entry_node_ids),
        exit_node_ids=list(document.exit_node_ids),
        node_refs=node_refs,
        topology_slices=topology_slices,
        adjacency_hints=adjacency_hints,
        metadata={
            "backend_kind": document.backend_kind,
            "workflow_profile": document.workflow_profile,
            "execution_kind": document.execution_kind,
            "preview": document.preview,
            "region_label": document.metadata.get("region_label", ""),
            "graph_regions": list(document.metadata.get("graph_regions", [])),
        },
    )
    topology_hints = {
        "region_label": document.metadata.get("region_label", ""),
        "node_count": len(node_refs),
        "adjacency_hint_count": len(adjacency_hints),
        "selected_probe_id": document.metadata.get("selected_probe_id", ""),
        "accepted_patch_id": document.metadata.get("accepted_patch_id", ""),
        "has_feedback": bool(document.metadata.get("has_feedback", False)),
        "entry_node_ids": list(document.entry_node_ids),
        "exit_node_ids": list(document.exit_node_ids),
    }
    return placeholder, topology_hints
