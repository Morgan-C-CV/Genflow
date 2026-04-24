from __future__ import annotations

"""Build workflow-native execution payloads from the current surrogate stack."""

from app.agent.memory import AgentSessionState
from app.agent.workflow_descriptor_builder import (
    build_surrogate_workflow_descriptor,
    build_surrogate_workflow_descriptor_from_execution_source,
)
from app.agent.workflow_document_builder import build_surrogate_workflow_document_from_descriptor
from app.agent.workflow_execution_models import (
    WorkflowCommitRequest,
    WorkflowExecutionArtifact,
    WorkflowExecutionPayload,
    WorkflowPreviewRequest,
)
from app.agent.workflow_execution_source_models import (
    WorkflowCommitSource,
    WorkflowExecutionSource,
    WorkflowPreviewSource,
)


def build_workflow_execution_payload(
    session: AgentSessionState,
    execution_kind: str,
    preview: bool = False,
) -> WorkflowExecutionPayload:
    descriptor = build_surrogate_workflow_descriptor(
        session=session,
        execution_kind=execution_kind,
        preview=preview,
    )
    return _build_workflow_execution_payload_from_descriptor(descriptor)


def build_workflow_execution_payload_from_source(
    source: WorkflowExecutionSource,
) -> WorkflowExecutionPayload:
    descriptor = build_surrogate_workflow_descriptor_from_execution_source(source)
    return _build_workflow_execution_payload_from_descriptor(descriptor)


def _build_workflow_execution_payload_from_descriptor(
    descriptor,
) -> WorkflowExecutionPayload:
    document = build_surrogate_workflow_document_from_descriptor(descriptor)
    return WorkflowExecutionPayload(
        workflow_id=document.workflow_id,
        workflow_kind=document.workflow_kind or "workflow_native_surrogate",
        workflow_version=descriptor.metadata.get("workflow_version", "phase-k-workflow-payload"),
        execution_kind=descriptor.execution.execution_kind,
        preview=descriptor.execution.preview,
        nodes=[
            {
                "node_id": node.node_id,
                "node_kind": node.node_kind,
                "role": node.role,
                "label": node.label,
                "metadata": dict(node.metadata),
            }
            for node in document.nodes
        ],
        edges=[
            {
                "edge_id": edge.edge_id,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "edge_kind": edge.edge_kind,
                "metadata": dict(edge.metadata),
            }
            for edge in document.edges
        ],
        entry_node_ids=list(document.entry_node_ids),
        exit_node_ids=list(document.exit_node_ids),
        execution_config={
            "execution_kind": descriptor.execution.execution_kind,
            "preview": descriptor.execution.preview,
            "backend_kind": document.backend_kind,
            "workflow_profile": document.workflow_profile,
            "region_label": document.metadata.get("region_label", ""),
            "selected_probe_id": document.metadata.get("selected_probe_id", ""),
            "accepted_patch_id": document.metadata.get("accepted_patch_id", ""),
        },
        backend_metadata={
            "backend_kind": document.backend_kind,
            "workflow_profile": document.workflow_profile,
            "graph_regions": list(document.metadata.get("graph_regions", [])),
            "source_document_id": document.document_id,
        },
        artifacts=[
            WorkflowExecutionArtifact(
                artifact_id=document.document_id,
                artifact_kind="workflow_document",
                uri=f"memory://workflow-execution/{document.document_id}",
                metadata={"region_label": document.metadata.get("region_label", "")},
            )
        ],
    )


def build_workflow_preview_request(session: AgentSessionState) -> WorkflowPreviewRequest:
    payload = build_workflow_execution_payload(session, execution_kind="preview", preview=True)
    return WorkflowPreviewRequest(
        workflow_payload=payload,
        preview_patch_spec={
            "probe_id": session.selected_probe.probe_id,
            "summary": session.selected_probe.summary,
            "target_axes": list(session.selected_probe.target_axes),
            "preserve_axes": list(session.selected_probe.preserve_axes),
            "source_kind": session.selected_probe.source_kind,
            "preview_execution_spec": dict(session.selected_probe.preview_execution_spec),
        },
        reference_info=_build_reference_info(session),
    )


def build_workflow_preview_request_from_source(
    source: WorkflowPreviewSource,
) -> WorkflowPreviewRequest:
    payload = build_workflow_execution_payload_from_source(source)
    return WorkflowPreviewRequest(
        workflow_payload=payload,
        preview_patch_spec={
            "probe_id": source.selected_probe.probe_id,
            "summary": source.selected_probe.summary,
            "target_axes": list(source.selected_probe.target_axes),
            "preserve_axes": list(source.selected_probe.preserve_axes),
            "source_kind": source.selected_probe.source_kind,
            "preview_execution_spec": dict(source.selected_probe.preview_execution_spec),
        },
        reference_info=_build_reference_info_from_source(source),
    )


def build_workflow_commit_request(session: AgentSessionState) -> WorkflowCommitRequest:
    payload = build_workflow_execution_payload(session, execution_kind="commit", preview=False)
    return WorkflowCommitRequest(
        workflow_payload=payload,
        committed_patch_spec={
            "patch_id": session.accepted_patch.patch_id,
            "target_fields": list(session.accepted_patch.target_fields),
            "target_axes": list(session.accepted_patch.target_axes),
            "preserve_axes": list(session.accepted_patch.preserve_axes),
            "changes": dict(session.accepted_patch.changes),
            "rationale": session.accepted_patch.rationale,
        },
        reference_info=_build_reference_info(session),
    )


def build_workflow_commit_request_from_source(
    source: WorkflowCommitSource,
) -> WorkflowCommitRequest:
    payload = build_workflow_execution_payload_from_source(source)
    return WorkflowCommitRequest(
        workflow_payload=payload,
        committed_patch_spec={
            "patch_id": source.accepted_patch.patch_id,
            "target_fields": list(source.accepted_patch.target_fields),
            "target_axes": list(source.accepted_patch.target_axes),
            "preserve_axes": list(source.accepted_patch.preserve_axes),
            "changes": dict(source.accepted_patch.changes),
            "rationale": source.accepted_patch.rationale,
        },
        reference_info=_build_reference_info_from_source(source),
    )


def _build_reference_info(session: AgentSessionState) -> dict:
    return {
        "selected_gallery_index": session.selected_gallery_index,
        "reference_ids": list(session.selected_reference_ids),
        "reference_count": len(session.selected_reference_ids),
    }


def _build_reference_info_from_source(source: WorkflowExecutionSource) -> dict:
    return {
        "selected_gallery_index": source.selected_gallery_index,
        "reference_ids": list(source.selected_reference_ids),
        "reference_count": len(source.selected_reference_ids),
    }
