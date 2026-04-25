from __future__ import annotations

"""Build workflow-native execution payloads from the current surrogate stack."""

from app.agent.memory import AgentSessionState
from app.agent.workflow_execution_models import (
    WorkflowCommitRequest,
    WorkflowExecutionArtifact,
    WorkflowExecutionPayload,
    WorkflowPreviewRequest,
)
from app.agent.workflow_graph_patch_builder import (
    build_workflow_graph_patch_from_committed_patch,
    build_workflow_graph_patch_from_preview_probe,
)
from app.agent.workflow_execution_source_models import (
    WorkflowCommitSource,
    WorkflowExecutionSource,
    WorkflowPreviewSource,
)
from app.agent.workflow_graph_source_builder import (
    build_workflow_graph_source,
    build_workflow_graph_source_from_execution_source,
)
from app.agent.workflow_graph_source_models import WorkflowGraphSource


def build_workflow_execution_payload(
    session: AgentSessionState,
    execution_kind: str,
    preview: bool = False,
) -> WorkflowExecutionPayload:
    graph_source = build_workflow_graph_source(
        session=session,
        execution_kind=execution_kind,
        preview=preview,
    )
    return build_workflow_execution_payload_from_graph_source(
        graph_source,
        execution_kind=execution_kind,
        preview=preview,
    )


def build_workflow_execution_payload_from_source(
    source: WorkflowExecutionSource,
) -> WorkflowExecutionPayload:
    graph_source = build_workflow_graph_source_from_execution_source(source)
    return build_workflow_execution_payload_from_graph_source(
        graph_source,
        execution_kind=source.execution_kind,
        preview=source.preview,
    )


def build_workflow_execution_payload_from_graph_source(
    graph_source: WorkflowGraphSource,
    execution_kind: str,
    preview: bool = False,
) -> WorkflowExecutionPayload:
    return WorkflowExecutionPayload(
        workflow_id=graph_source.workflow_id,
        workflow_kind=graph_source.workflow_kind or "workflow_native_surrogate",
        workflow_version=graph_source.workflow_version or "phase-k-workflow-payload",
        execution_kind=execution_kind,
        preview=preview,
        nodes=[
            {
                "node_id": node.node_id,
                "node_kind": node.node_type,
                "role": node.role,
                "label": node.label,
                "metadata": {
                    **dict(node.metadata),
                    "graph_node_type": node.node_type,
                },
            }
            for node in graph_source.nodes
        ],
        edges=[
            {
                "edge_id": edge.edge_id,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "edge_kind": edge.edge_type,
                "metadata": {
                    **dict(edge.metadata),
                    "graph_edge_type": edge.edge_type,
                },
            }
            for edge in graph_source.edges
        ],
        entry_node_ids=list(graph_source.entry_node_ids),
        exit_node_ids=list(graph_source.exit_node_ids),
        execution_config={
            "execution_kind": execution_kind,
            "preview": preview,
            "backend_kind": graph_source.backend_kind,
            "workflow_profile": graph_source.workflow_profile,
            "region_label": graph_source.metadata.get("region_label", ""),
            "selected_probe_id": graph_source.metadata.get("selected_probe_id", ""),
            "accepted_patch_id": graph_source.metadata.get("accepted_patch_id", ""),
        },
        backend_metadata={
            "backend_kind": graph_source.backend_kind,
            "workflow_profile": graph_source.workflow_profile,
            "graph_regions": list(graph_source.metadata.get("graph_regions", [])),
            "source_document_id": graph_source.metadata.get("source_document_id", ""),
            "source_graph_id": graph_source.workflow_id,
        },
        artifacts=[
            WorkflowExecutionArtifact(
                artifact_id=graph_source.workflow_id,
                artifact_kind="workflow_graph_source",
                uri=f"memory://workflow-execution/{graph_source.workflow_id}",
                metadata={"region_label": graph_source.metadata.get("region_label", "")},
            )
        ],
    )


def build_workflow_preview_request(session: AgentSessionState) -> WorkflowPreviewRequest:
    payload = build_workflow_execution_payload(session, execution_kind="preview", preview=True)
    graph_source = build_workflow_graph_source(session, execution_kind="preview", preview=True)
    graph_patch = build_workflow_graph_patch_from_preview_probe(
        session.selected_probe,
        graph_source=graph_source,
        session=session,
    )
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
        graph_patch_spec=_serialize_graph_patch(graph_patch),
        reference_info=_build_reference_info(session),
    )


def build_workflow_preview_request_from_source(
    source: WorkflowPreviewSource,
) -> WorkflowPreviewRequest:
    payload = build_workflow_execution_payload_from_source(source)
    graph_source = build_workflow_graph_source_from_execution_source(source)
    graph_patch = build_workflow_graph_patch_from_preview_probe(
        source.selected_probe,
        graph_source=graph_source,
    )
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
        graph_patch_spec=_serialize_graph_patch(graph_patch),
        reference_info=_build_reference_info_from_source(source),
    )


def build_workflow_commit_request(session: AgentSessionState) -> WorkflowCommitRequest:
    payload = build_workflow_execution_payload(session, execution_kind="commit", preview=False)
    graph_source = build_workflow_graph_source(session, execution_kind="commit", preview=False)
    graph_patch = (
        session.selected_workflow_graph_patch
        if session.preferred_commit_source == "graph" and session.selected_workflow_graph_patch.patch_id
        else session.current_workflow_graph_patch
    )
    if not graph_patch.patch_id:
        graph_patch = build_workflow_graph_patch_from_committed_patch(
            committed_patch=session.accepted_patch,
            graph_source=graph_source,
            session=session,
        )
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
        graph_patch_spec=_serialize_graph_patch(graph_patch),
        commit_source_payload=_build_commit_source_payload(session),
        reference_info=_build_reference_info(session),
    )


def build_workflow_commit_request_from_source(
    source: WorkflowCommitSource,
) -> WorkflowCommitRequest:
    payload = build_workflow_execution_payload_from_source(source)
    graph_source = build_workflow_graph_source_from_execution_source(source)
    graph_patch = (
        source.selected_workflow_graph_patch
        if source.commit_execution_mode == "graph_native_execution_handoff"
        and source.selected_workflow_graph_patch.patch_id
        else build_workflow_graph_patch_from_committed_patch(
            committed_patch=source.accepted_patch,
            graph_source=graph_source,
        )
    )
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
        graph_patch_spec=_serialize_graph_patch(graph_patch),
        commit_source_payload=_build_commit_source_payload_from_source(source, graph_patch),
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


def _build_commit_source_payload(session: AgentSessionState) -> dict:
    return {
        "commit_execution_mode": session.commit_execution_mode,
        "preferred_commit_source": session.preferred_commit_source,
        "selected_workflow_graph_patch_id": session.selected_workflow_graph_patch.patch_id,
        "top_schema_patch_id": session.top_schema_patch_candidate.patch_id,
        "top_graph_patch_candidate_id": session.top_workflow_graph_patch_candidate.candidate_id,
    }


def _build_commit_source_payload_from_source(source: WorkflowCommitSource, graph_patch) -> dict:
    metadata = dict(source.accepted_patch.metadata)
    return {
        "commit_execution_mode": str(
            source.commit_execution_mode
            or metadata.get("commit_execution_mode", "schema_execution_fallback")
            or "schema_execution_fallback"
        ),
        "preferred_commit_source": str(metadata.get("preferred_commit_source", "schema") or "schema"),
        "selected_workflow_graph_patch_id": str(
            metadata.get("selected_workflow_graph_patch_id", graph_patch.patch_id)
        ),
        "top_schema_patch_id": str(metadata.get("top_schema_patch_id", source.accepted_patch.patch_id)),
        "top_graph_patch_candidate_id": str(metadata.get("top_graph_patch_candidate_id", "")),
    }


def _serialize_graph_patch(graph_patch) -> dict:
    return {
        "workflow_id": graph_patch.workflow_id,
        "patch_id": graph_patch.patch_id,
        "patch_kind": graph_patch.patch_kind,
        "node_patches": [
            {
                "node_id": patch.node_id,
                "operation": patch.operation,
                "target_fields": list(patch.target_fields),
                "target_axes": list(patch.target_axes),
                "changes": dict(patch.changes),
                "rationale": patch.rationale,
                "metadata": dict(patch.metadata),
            }
            for patch in graph_patch.node_patches
        ],
        "edge_patches": [
            {
                "edge_id": patch.edge_id,
                "operation": patch.operation,
                "target_axes": list(patch.target_axes),
                "preserve_axes": list(patch.preserve_axes),
                "rationale": patch.rationale,
                "metadata": dict(patch.metadata),
            }
            for patch in graph_patch.edge_patches
        ],
        "region_patches": [
            {
                "region_id": patch.region_id,
                "operation": patch.operation,
                "target_axes": list(patch.target_axes),
                "preserve_axes": list(patch.preserve_axes),
                "rationale": patch.rationale,
                "metadata": dict(patch.metadata),
            }
            for patch in graph_patch.region_patches
        ],
        "metadata": dict(graph_patch.metadata),
    }
