from __future__ import annotations

"""Build deterministic local graph-native patch candidates from repair context."""

from app.agent.memory import AgentSessionState
from app.agent.workflow_graph_patch_builder import build_workflow_graph_patch_from_preview_probe
from app.agent.workflow_graph_patch_models import WorkflowGraphPatchCandidate
from app.agent.workflow_graph_source_builder import build_workflow_graph_source
from app.agent.workflow_graph_source_models import WorkflowGraphSource


def build_workflow_graph_patch_candidates(session: AgentSessionState) -> list[WorkflowGraphPatchCandidate]:
    if not session.selected_probe.probe_id:
        return []
    graph_source = build_workflow_graph_source(session, execution_kind="commit_plan", preview=False)
    return build_workflow_graph_patch_candidates_from_probe(
        selected_probe=session.selected_probe,
        graph_source=graph_source,
        repair_context={
            "preserve_constraints": list(session.preserve_constraints),
            "dissatisfaction_axes": list(session.dissatisfaction_axes),
            "repair_hypothesis_count": len(session.repair_hypotheses),
        },
    )


def build_workflow_graph_patch_candidates_from_probe(
    selected_probe,
    graph_source: WorkflowGraphSource,
    repair_context: dict | None = None,
) -> list[WorkflowGraphPatchCandidate]:
    if not selected_probe.probe_id:
        return []
    repair_context = repair_context or {}
    base_patch = build_workflow_graph_patch_from_preview_probe(selected_probe, graph_source=graph_source)
    variants = [
        ("balanced_graph_shift", "balanced", list(selected_probe.target_axes), list(selected_probe.preserve_axes)),
        (
            "target_amplified",
            "aggressive",
            list(selected_probe.target_axes) + _extra_axes(repair_context.get("dissatisfaction_axes", []), selected_probe.target_axes),
            list(selected_probe.preserve_axes),
        ),
        (
            "preserve_safe",
            "conservative",
            list(selected_probe.target_axes),
            list(dict.fromkeys(list(selected_probe.preserve_axes) + list(repair_context.get("preserve_constraints", [])))),
        ),
    ]
    candidates: list[WorkflowGraphPatchCandidate] = []
    for index, (variant_id, variant_kind, target_axes, preserve_axes) in enumerate(variants, start=1):
        candidates.append(
            WorkflowGraphPatchCandidate(
                workflow_id=graph_source.workflow_id,
                candidate_id=f"gpc:{selected_probe.probe_id}:{index:02d}",
                candidate_kind=variant_kind,
                node_patches=[
                    _clone_node_patch(node_patch, variant_id, target_axes, preserve_axes)
                    for node_patch in base_patch.node_patches
                ],
                edge_patches=[
                    _clone_edge_patch(edge_patch, variant_id, target_axes, preserve_axes)
                    for edge_patch in base_patch.edge_patches
                ],
                region_patches=[
                    _clone_region_patch(region_patch, variant_id, target_axes, preserve_axes)
                    for region_patch in base_patch.region_patches
                ],
                target_axes=target_axes,
                preserve_axes=preserve_axes,
                candidate_rationale=_build_candidate_rationale(variant_id, selected_probe.summary),
                metadata={
                    "variant_id": variant_id,
                    "probe_id": selected_probe.probe_id,
                    "source_graph_id": graph_source.workflow_id,
                    "repair_hypothesis_count": int(repair_context.get("repair_hypothesis_count", 0)),
                },
            )
        )
    return candidates


def _extra_axes(dissatisfaction_axes: list[str], current_target_axes: list[str]) -> list[str]:
    return [axis for axis in dissatisfaction_axes if axis not in current_target_axes][:1]


def _clone_node_patch(node_patch, variant_id: str, target_axes: list[str], preserve_axes: list[str]):
    from app.agent.workflow_graph_patch_models import WorkflowNodePatch

    return WorkflowNodePatch(
        node_id=node_patch.node_id,
        operation=f"{node_patch.operation}:{variant_id}",
        target_fields=list(node_patch.target_fields),
        target_axes=list(target_axes),
        changes={**dict(node_patch.changes), "preserve_axes": list(preserve_axes)},
        rationale=node_patch.rationale,
        metadata={**dict(node_patch.metadata), "variant_id": variant_id},
    )


def _clone_edge_patch(edge_patch, variant_id: str, target_axes: list[str], preserve_axes: list[str]):
    from app.agent.workflow_graph_patch_models import WorkflowEdgePatch

    return WorkflowEdgePatch(
        edge_id=edge_patch.edge_id,
        operation=f"{edge_patch.operation}:{variant_id}",
        target_axes=list(target_axes),
        preserve_axes=list(preserve_axes),
        rationale=edge_patch.rationale,
        metadata={**dict(edge_patch.metadata), "variant_id": variant_id},
    )


def _clone_region_patch(region_patch, variant_id: str, target_axes: list[str], preserve_axes: list[str]):
    from app.agent.workflow_graph_patch_models import WorkflowRegionPatch

    return WorkflowRegionPatch(
        region_id=region_patch.region_id,
        operation=f"{region_patch.operation}:{variant_id}",
        target_axes=list(target_axes),
        preserve_axes=list(preserve_axes),
        rationale=region_patch.rationale,
        metadata={**dict(region_patch.metadata), "variant_id": variant_id},
    )


def _build_candidate_rationale(variant_id: str, probe_summary: str) -> str:
    if variant_id == "target_amplified":
        return f"Amplify target repair direction from probe: {probe_summary}"
    if variant_id == "preserve_safe":
        return f"Bias graph repair toward preservation while following probe: {probe_summary}"
    return f"Balance target and preserve graph repair for probe: {probe_summary}"
