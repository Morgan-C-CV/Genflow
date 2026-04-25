from __future__ import annotations

"""Project surrogate workflow structures into a graph-native source container."""

from app.agent.memory import AgentSessionState
from app.agent.workflow_descriptor_builder import build_surrogate_workflow_descriptor_from_execution_source
from app.agent.workflow_document_builder import (
    build_surrogate_workflow_document,
    build_surrogate_workflow_document_from_descriptor,
)
from app.agent.workflow_execution_source_models import WorkflowExecutionSource
from app.agent.workflow_graph_source_models import (
    WorkflowGraphEdge,
    WorkflowGraphNode,
    WorkflowGraphRegion,
    WorkflowGraphSource,
)


def build_workflow_graph_source(
    session: AgentSessionState,
    execution_kind: str = "",
    preview: bool = False,
) -> WorkflowGraphSource:
    document = build_surrogate_workflow_document(
        session=session,
        execution_kind=execution_kind,
        preview=preview,
    )
    workflow_version = session.workflow_identity.workflow_version or "phase-k-workflow-payload"
    return build_workflow_graph_source_from_document(document, workflow_version=workflow_version)


def build_workflow_graph_source_from_execution_source(
    source: WorkflowExecutionSource,
) -> WorkflowGraphSource:
    descriptor = build_surrogate_workflow_descriptor_from_execution_source(source)
    document = build_surrogate_workflow_document_from_descriptor(descriptor)
    workflow_version = source.workflow_version or "phase-k-workflow-payload"
    return build_workflow_graph_source_from_document(document, workflow_version=workflow_version)


def build_workflow_graph_source_from_document(
    document,
    workflow_version: str = "phase-k-workflow-payload",
) -> WorkflowGraphSource:
    return WorkflowGraphSource(
        workflow_id=document.workflow_id,
        workflow_kind=document.workflow_kind or "workflow_native_surrogate",
        workflow_version=workflow_version,
        backend_kind=document.backend_kind,
        workflow_profile=document.workflow_profile,
        nodes=[
            WorkflowGraphNode(
                node_id=node.node_id,
                node_type=node.node_kind,
                role=node.role,
                label=node.label,
                config={},
                metadata=dict(node.metadata),
            )
            for node in document.nodes
        ],
        edges=[
            WorkflowGraphEdge(
                edge_id=edge.edge_id,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                edge_type=edge.edge_kind,
                metadata=dict(edge.metadata),
            )
            for edge in document.edges
        ],
        regions=[
            WorkflowGraphRegion(
                region_id=region.region_id,
                region_type=region.region_kind,
                label=region.region_label,
                node_ids=list(region.node_ids),
                entry_node_ids=list(region.entry_node_ids),
                exit_node_ids=list(region.exit_node_ids),
                metadata=dict(region.metadata),
            )
            for region in document.regions
        ],
        entry_node_ids=list(document.entry_node_ids),
        exit_node_ids=list(document.exit_node_ids),
        metadata={
            "source_document_id": document.document_id,
            "graph_regions": list(document.metadata.get("graph_regions", [])),
            "region_label": document.metadata.get("region_label", ""),
            "selected_probe_id": document.metadata.get("selected_probe_id", ""),
            "accepted_patch_id": document.metadata.get("accepted_patch_id", ""),
        },
    )
