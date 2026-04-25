from __future__ import annotations

"""Build graph-native patch intent from committed schema-level repair decisions."""

from app.agent.memory import AgentSessionState
from app.agent.runtime_models import CommittedPatch, PreviewProbe
from app.agent.workflow_graph_patch_models import (
    WorkflowEdgePatch,
    WorkflowGraphPatch,
    WorkflowGraphPatchCandidate,
    WorkflowNodePatch,
    WorkflowRegionPatch,
)
from app.agent.workflow_graph_source_builder import build_workflow_graph_source
from app.agent.workflow_graph_source_models import WorkflowGraphSource


def build_workflow_graph_patch(session: AgentSessionState) -> WorkflowGraphPatch:
    graph_source = build_workflow_graph_source(session, execution_kind="commit_plan", preview=False)
    return build_workflow_graph_patch_from_committed_patch(
        committed_patch=session.accepted_patch,
        graph_source=graph_source,
        session=session,
    )


def build_workflow_graph_patch_from_committed_patch(
    committed_patch: CommittedPatch,
    graph_source: WorkflowGraphSource,
    session: AgentSessionState | None = None,
) -> WorkflowGraphPatch:
    if not committed_patch.patch_id:
        return WorkflowGraphPatch(workflow_id=graph_source.workflow_id)

    node_targets = _map_node_targets(committed_patch.target_fields)
    if not node_targets:
        node_targets = {"render.model": list(committed_patch.target_fields)}

    node_patches = [
        WorkflowNodePatch(
            node_id=node_id,
            operation="update_node_config",
            target_fields=list(fields),
            target_axes=list(committed_patch.target_axes),
            changes={field: committed_patch.changes.get(field) for field in fields if field in committed_patch.changes},
            rationale=committed_patch.rationale,
            metadata={
                "patch_id": committed_patch.patch_id,
                "preserve_axes": list(committed_patch.preserve_axes),
                "candidate_kind": str(committed_patch.metadata.get("candidate_kind", "")),
            },
        )
        for node_id, fields in node_targets.items()
    ]

    patched_node_ids = {patch.node_id for patch in node_patches}
    edge_patches = [
        WorkflowEdgePatch(
            edge_id=edge.edge_id,
            operation="rebind_flow_constraints",
            target_axes=list(committed_patch.target_axes),
            preserve_axes=list(committed_patch.preserve_axes),
            rationale=f"Align edge flow with patch {committed_patch.patch_id}.",
            metadata={
                "patch_id": committed_patch.patch_id,
                "edge_type": edge.edge_type,
            },
        )
        for edge in graph_source.edges
        if edge.source_node_id in patched_node_ids
        or edge.target_node_id in patched_node_ids
        or edge.source_node_id == f"patch.{committed_patch.patch_id}"
    ]

    region_patches = [
        WorkflowRegionPatch(
            region_id=region.region_id,
            operation="update_region_intent",
            target_axes=list(committed_patch.target_axes),
            preserve_axes=list(committed_patch.preserve_axes),
            rationale=f"Repair region intent updated by patch {committed_patch.patch_id}.",
            metadata={
                "patch_id": committed_patch.patch_id,
                "region_type": region.region_type,
            },
        )
        for region in graph_source.regions
        if region.region_type == "repair_region" or f"patch.{committed_patch.patch_id}" in region.node_ids
    ]

    return WorkflowGraphPatch(
        workflow_id=graph_source.workflow_id,
        patch_id=committed_patch.patch_id,
        patch_kind="graph_intent_projection",
        node_patches=node_patches,
        edge_patches=edge_patches,
        region_patches=region_patches,
        metadata={
            "workflow_kind": graph_source.workflow_kind,
            "workflow_version": graph_source.workflow_version,
            "target_axes": list(committed_patch.target_axes),
            "preserve_axes": list(committed_patch.preserve_axes),
            "target_fields": list(committed_patch.target_fields),
            "selected_probe_id": session.selected_probe.probe_id if session is not None else "",
            "source_graph_id": graph_source.workflow_id,
        },
    )


def build_workflow_graph_patch_from_preview_probe(
    selected_probe: PreviewProbe,
    graph_source: WorkflowGraphSource,
    session: AgentSessionState | None = None,
) -> WorkflowGraphPatch:
    if not selected_probe.probe_id:
        return WorkflowGraphPatch(workflow_id=graph_source.workflow_id)

    node_patches = [
        WorkflowNodePatch(
            node_id="render.model",
            operation="preview_node_adjustment",
            target_fields=["style_direction"],
            target_axes=list(selected_probe.target_axes),
            changes={
                "target_axes": list(selected_probe.target_axes),
                "preserve_axes": list(selected_probe.preserve_axes),
                "source_kind": selected_probe.source_kind,
            },
            rationale=selected_probe.summary,
            metadata={
                "probe_id": selected_probe.probe_id,
                "preview_execution_spec": dict(selected_probe.preview_execution_spec),
            },
        )
    ]
    edge_patches = [
        WorkflowEdgePatch(
            edge_id=edge.edge_id,
            operation="preview_flow_adjustment",
            target_axes=list(selected_probe.target_axes),
            preserve_axes=list(selected_probe.preserve_axes),
            rationale=f"Preview graph intent for probe {selected_probe.probe_id}.",
            metadata={"probe_id": selected_probe.probe_id, "edge_type": edge.edge_type},
        )
        for edge in graph_source.edges
        if edge.target_node_id == "render.model" or edge.source_node_id == "render.model"
    ]
    region_patches = [
        WorkflowRegionPatch(
            region_id=region.region_id,
            operation="preview_region_adjustment",
            target_axes=list(selected_probe.target_axes),
            preserve_axes=list(selected_probe.preserve_axes),
            rationale=f"Preview region intent for probe {selected_probe.probe_id}.",
            metadata={"probe_id": selected_probe.probe_id, "region_type": region.region_type},
        )
        for region in graph_source.regions
        if region.region_type in {"initial_region", "repair_region"}
    ]
    return WorkflowGraphPatch(
        workflow_id=graph_source.workflow_id,
        patch_id=f"preview:{selected_probe.probe_id}",
        patch_kind="graph_preview_projection",
        node_patches=node_patches,
        edge_patches=edge_patches,
        region_patches=region_patches,
        metadata={
            "probe_id": selected_probe.probe_id,
            "workflow_kind": graph_source.workflow_kind,
            "workflow_version": graph_source.workflow_version,
            "target_axes": list(selected_probe.target_axes),
            "preserve_axes": list(selected_probe.preserve_axes),
            "selected_probe_id": selected_probe.probe_id,
            "source_graph_id": graph_source.workflow_id,
            "session_id": session.session_id if session is not None else "",
        },
    )


def materialize_workflow_graph_patch_from_candidate(
    candidate: WorkflowGraphPatchCandidate,
    session: AgentSessionState | None = None,
) -> WorkflowGraphPatch:
    if not candidate.candidate_id:
        return WorkflowGraphPatch(workflow_id="")
    patch_id = candidate.candidate_id.replace("gpc:", "wgp:", 1)
    return WorkflowGraphPatch(
        workflow_id=candidate.workflow_id,
        patch_id=patch_id,
        patch_kind="graph_candidate_selected_commit_artifact",
        node_patches=[
            WorkflowNodePatch(
                node_id=patch.node_id,
                operation=patch.operation,
                target_fields=list(patch.target_fields),
                target_axes=list(patch.target_axes),
                changes=dict(patch.changes),
                rationale=patch.rationale,
                metadata=dict(patch.metadata),
            )
            for patch in candidate.node_patches
        ],
        edge_patches=[
            WorkflowEdgePatch(
                edge_id=patch.edge_id,
                operation=patch.operation,
                target_axes=list(patch.target_axes),
                preserve_axes=list(patch.preserve_axes),
                rationale=patch.rationale,
                metadata=dict(patch.metadata),
            )
            for patch in candidate.edge_patches
        ],
        region_patches=[
            WorkflowRegionPatch(
                region_id=patch.region_id,
                operation=patch.operation,
                target_axes=list(patch.target_axes),
                preserve_axes=list(patch.preserve_axes),
                rationale=patch.rationale,
                metadata=dict(patch.metadata),
            )
            for patch in candidate.region_patches
        ],
        metadata={
            "candidate_id": candidate.candidate_id,
            "candidate_kind": candidate.candidate_kind,
            "target_axes": list(candidate.target_axes),
            "preserve_axes": list(candidate.preserve_axes),
            "candidate_rationale": candidate.candidate_rationale,
            "selected_probe_id": session.selected_probe.probe_id if session is not None else "",
            "pbo_score": candidate.metadata.get("pbo_score", 0.0),
            "pbo_rationale": list(candidate.metadata.get("pbo_rationale", [])),
        },
    )


def _map_node_targets(target_fields: list[str]) -> dict[str, list[str]]:
    node_targets: dict[str, list[str]] = {}
    for field in target_fields:
        if field in {"prompt", "negative_prompt"}:
            node_targets.setdefault("intent.prompt", []).append(field)
        else:
            node_targets.setdefault("render.model", []).append(field)
    return node_targets
