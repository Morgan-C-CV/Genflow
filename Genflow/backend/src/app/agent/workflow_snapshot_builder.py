from __future__ import annotations

"""Consolidate surrogate workflow-aware outputs into one snapshot.

Current surrogate workflow-aware stack:
session state -> descriptor -> document -> graph placeholder -> snapshot.
Runtime sync should consume this snapshot as a write-back payload rather than
rebuilding workflow-aware structures inline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.agent.memory import AgentSessionState
from app.agent.workflow_descriptor_builder import build_surrogate_workflow_descriptor
from app.agent.workflow_document_builder import build_surrogate_workflow_document_from_descriptor
from app.agent.workflow_graph_builder import build_surrogate_workflow_graph_from_document
from app.agent.workflow_runtime_models import WorkflowGraphPlaceholder, WorkflowIdentity, WorkflowScope
from app.agent.workflow_scope_materializer import (
    materialize_editable_scopes,
    materialize_protected_scopes,
)


@dataclass
class SurrogateWorkflowSnapshot:
    workflow_identity: WorkflowIdentity = field(default_factory=WorkflowIdentity)
    workflow_metadata: Dict[str, Any] = field(default_factory=dict)
    surrogate_payload: Dict[str, Any] = field(default_factory=dict)
    workflow_graph_placeholder: WorkflowGraphPlaceholder = field(default_factory=WorkflowGraphPlaceholder)
    workflow_topology_hints: Dict[str, Any] = field(default_factory=dict)
    workflow_topology_entry_node_ids: List[str] = field(default_factory=list)
    workflow_topology_exit_node_ids: List[str] = field(default_factory=list)
    editable_scopes: List[WorkflowScope] = field(default_factory=list)
    protected_scopes: List[WorkflowScope] = field(default_factory=list)


def build_surrogate_workflow_snapshot(
    session: AgentSessionState,
    execution_kind: str = "",
    preview: bool = False,
    backend_kind: str = "",
    workflow_profile: str = "",
) -> SurrogateWorkflowSnapshot:
    workflow_identity = WorkflowIdentity(
        workflow_id=session.workflow_id or f"workflow-{session.session_id}",
        workflow_kind=session.workflow_identity.workflow_kind or "normalized_schema_surrogate",
        workflow_version=session.workflow_identity.workflow_version or "phase-g-skeleton",
    )
    editable_scopes = materialize_editable_scopes(session)
    protected_scopes = materialize_protected_scopes(session)
    descriptor = build_surrogate_workflow_descriptor(
        session=session,
        execution_kind=execution_kind,
        preview=preview,
    )
    effective_backend_kind = backend_kind or descriptor.execution.backend_kind
    effective_workflow_profile = workflow_profile or descriptor.execution.workflow_profile
    descriptor.execution.backend_kind = effective_backend_kind
    descriptor.execution.workflow_profile = effective_workflow_profile
    document = build_surrogate_workflow_document_from_descriptor(descriptor)
    graph_placeholder, workflow_topology_hints = build_surrogate_workflow_graph_from_document(
        document=document,
        scope_partitions=[scope.scope_id for scope in editable_scopes + protected_scopes],
    )
    workflow_metadata = {
        "backend_kind": effective_backend_kind,
        "workflow_profile": effective_workflow_profile,
        "surrogate_kind": "normalized_schema",
        "has_schema": bool(descriptor.execution.metadata.get("has_schema", False)),
        "has_preview_results": bool(session.preview_probe_results),
        "has_committed_patch": bool(descriptor.repair.accepted_patch_id),
        "has_feedback": descriptor.repair.has_feedback,
        "feedback_count": descriptor.repair.feedback_count,
        "dissatisfaction_axes": list(descriptor.repair.dissatisfaction_axes),
        "preserve_constraints": list(descriptor.repair.preserve_constraints),
        "repair_hypothesis_count": len(session.repair_hypotheses),
        "probe_count": len(session.preview_probe_candidates),
        "selected_probe_id": descriptor.repair.selected_probe_id,
        "current_uncertainty_estimate": descriptor.repair.current_uncertainty_estimate,
        "graph_entry_node_ids": list(document.entry_node_ids),
        "graph_exit_node_ids": list(document.exit_node_ids),
        "document_region_label": str(document.metadata.get("region_label", "")),
        "benchmark_comparison": {
            "benchmark_source": descriptor.benchmark_comparison.benchmark_source,
            "compared_anchor_ids": list(descriptor.benchmark_comparison.compared_anchor_ids),
            "compared_candidate_ids": list(descriptor.benchmark_comparison.compared_candidate_ids),
            "focus_axes": list(descriptor.benchmark_comparison.focus_axes),
            "preserve_axes": list(descriptor.benchmark_comparison.preserve_axes),
            "confidence_hint": descriptor.benchmark_comparison.confidence_hint,
        },
        "verifier_signal_summary": {
            "target_alignment_score": session.latest_verifier_signal_summary.target_alignment_score,
            "preserve_risk_score": session.latest_verifier_signal_summary.preserve_risk_score,
            "benchmark_support_score": session.latest_verifier_signal_summary.benchmark_support_score,
            "execution_evidence_score": session.latest_verifier_signal_summary.execution_evidence_score,
            "total_score": session.latest_verifier_signal_summary.total_score,
            "notes": list(session.latest_verifier_signal_summary.notes),
            "regression_notes": list(session.latest_verifier_signal_summary.regression_notes),
        },
        "verifier_repair_recommendation": {
            "recommended_action": session.latest_verifier_repair_recommendation.recommended_action,
            "rationale": list(session.latest_verifier_repair_recommendation.rationale),
            "priority": session.latest_verifier_repair_recommendation.priority,
            "supporting_signals": list(session.latest_verifier_repair_recommendation.supporting_signals),
        },
        "workflow_graph_patch": {
            "patch_id": session.current_workflow_graph_patch.patch_id,
            "patch_kind": session.current_workflow_graph_patch.patch_kind,
            "node_patch_count": len(session.current_workflow_graph_patch.node_patches),
            "edge_patch_count": len(session.current_workflow_graph_patch.edge_patches),
            "region_patch_count": len(session.current_workflow_graph_patch.region_patches),
            "target_fields": list(session.current_workflow_graph_patch.metadata.get("target_fields", [])),
            "target_axes": list(session.current_workflow_graph_patch.metadata.get("target_axes", [])),
        },
        "workflow_graph_patch_candidates": {
            "candidate_count": len(session.workflow_graph_patch_candidates),
            "candidate_ids": [candidate.candidate_id for candidate in session.workflow_graph_patch_candidates],
            "candidate_kinds": [candidate.candidate_kind for candidate in session.workflow_graph_patch_candidates],
            "top_candidate_id": (
                session.workflow_graph_patch_candidates[0].candidate_id
                if session.workflow_graph_patch_candidates
                else ""
            ),
            "top_candidate_score": (
                session.workflow_graph_patch_candidates[0].metadata.get("pbo_score", 0.0)
                if session.workflow_graph_patch_candidates
                else 0.0
            ),
        },
        "patch_winner_comparison": {
            "top_schema_patch_id": session.top_schema_patch_candidate.patch_id,
            "top_schema_patch_axes": list(session.top_schema_patch_candidate.target_axes),
            "top_graph_patch_candidate_id": session.top_workflow_graph_patch_candidate.candidate_id,
            "top_graph_patch_candidate_axes": list(session.top_workflow_graph_patch_candidate.target_axes),
            "preferred_commit_source": session.preferred_commit_source,
            "selected_graph_native_patch_candidate_id": session.selected_graph_native_patch_candidate.candidate_id,
            "graph_native_aligned_winner": bool(
                session.accepted_patch.metadata.get("graph_native_aligned_winner", False)
            ),
            "aligned_graph_candidate_id": str(
                session.accepted_patch.metadata.get("aligned_graph_candidate_id", "")
            ),
        },
    }
    surrogate_payload = {
        "schema": {
            "prompt": descriptor.schema_prompt,
            "negative_prompt": descriptor.schema_negative_prompt,
            "model": descriptor.schema_model,
            "sampler": descriptor.schema_sampler,
            "style": list(descriptor.schema_style),
            "lora": list(descriptor.schema_lora),
        },
        "selected_gallery_index": descriptor.selected_gallery_index,
        "selected_reference_ids": list(descriptor.selected_reference_ids),
        "selected_probe_id": descriptor.repair.selected_probe_id,
        "accepted_patch_id": descriptor.repair.accepted_patch_id,
        "current_result_id": descriptor.execution.current_result_id,
        "latest_feedback": str(descriptor.repair.metadata.get("latest_feedback", "")),
        "feedback_count": descriptor.repair.feedback_count,
        "dissatisfaction_axes": list(descriptor.repair.dissatisfaction_axes),
        "preserve_constraints": list(descriptor.repair.preserve_constraints),
        "repair_hypothesis_count": len(session.repair_hypotheses),
        "probe_count": len(session.preview_probe_candidates),
        "current_uncertainty_estimate": descriptor.repair.current_uncertainty_estimate,
        "workflow_document_id": document.document_id,
        "workflow_document_region_label": document.metadata.get("region_label", ""),
        "workflow_document_entry_node_ids": list(document.entry_node_ids),
        "workflow_document_exit_node_ids": list(document.exit_node_ids),
        "workflow_topology_graph_id": graph_placeholder.graph_id,
        "workflow_topology_slice_count": len(graph_placeholder.topology_slices),
        "workflow_topology_entry_node_ids": list(graph_placeholder.entry_node_ids),
        "workflow_topology_exit_node_ids": list(graph_placeholder.exit_node_ids),
        "benchmark_comparison": {
            "benchmark_source": descriptor.benchmark_comparison.benchmark_source,
            "compared_anchor_ids": list(descriptor.benchmark_comparison.compared_anchor_ids),
            "compared_candidate_ids": list(descriptor.benchmark_comparison.compared_candidate_ids),
            "focus_axes": list(descriptor.benchmark_comparison.focus_axes),
            "preserve_axes": list(descriptor.benchmark_comparison.preserve_axes),
            "confidence_hint": descriptor.benchmark_comparison.confidence_hint,
        },
        "verifier_signal_summary": {
            "target_alignment_score": session.latest_verifier_signal_summary.target_alignment_score,
            "preserve_risk_score": session.latest_verifier_signal_summary.preserve_risk_score,
            "benchmark_support_score": session.latest_verifier_signal_summary.benchmark_support_score,
            "execution_evidence_score": session.latest_verifier_signal_summary.execution_evidence_score,
            "total_score": session.latest_verifier_signal_summary.total_score,
            "notes": list(session.latest_verifier_signal_summary.notes),
            "regression_notes": list(session.latest_verifier_signal_summary.regression_notes),
        },
        "verifier_repair_recommendation": {
            "recommended_action": session.latest_verifier_repair_recommendation.recommended_action,
            "rationale": list(session.latest_verifier_repair_recommendation.rationale),
            "priority": session.latest_verifier_repair_recommendation.priority,
            "supporting_signals": list(session.latest_verifier_repair_recommendation.supporting_signals),
        },
        "workflow_graph_patch": {
            "patch_id": session.current_workflow_graph_patch.patch_id,
            "patch_kind": session.current_workflow_graph_patch.patch_kind,
            "node_patches": [
                {
                    "node_id": patch.node_id,
                    "operation": patch.operation,
                    "target_fields": list(patch.target_fields),
                    "target_axes": list(patch.target_axes),
                }
                for patch in session.current_workflow_graph_patch.node_patches
            ],
            "edge_patches": [
                {
                    "edge_id": patch.edge_id,
                    "operation": patch.operation,
                    "target_axes": list(patch.target_axes),
                    "preserve_axes": list(patch.preserve_axes),
                }
                for patch in session.current_workflow_graph_patch.edge_patches
            ],
            "region_patches": [
                {
                    "region_id": patch.region_id,
                    "operation": patch.operation,
                    "target_axes": list(patch.target_axes),
                    "preserve_axes": list(patch.preserve_axes),
                }
                for patch in session.current_workflow_graph_patch.region_patches
            ],
        },
        "workflow_graph_patch_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "candidate_kind": candidate.candidate_kind,
                "target_axes": list(candidate.target_axes),
                "preserve_axes": list(candidate.preserve_axes),
                "candidate_rationale": candidate.candidate_rationale,
                "node_patch_count": len(candidate.node_patches),
                "edge_patch_count": len(candidate.edge_patches),
                "region_patch_count": len(candidate.region_patches),
                "pbo_score": candidate.metadata.get("pbo_score", 0.0),
                "pbo_rationale": list(candidate.metadata.get("pbo_rationale", [])),
            }
            for candidate in session.workflow_graph_patch_candidates
        ],
        "patch_winner_comparison": {
            "top_schema_patch_candidate": {
                "patch_id": session.top_schema_patch_candidate.patch_id,
                "target_axes": list(session.top_schema_patch_candidate.target_axes),
                "preserve_axes": list(session.top_schema_patch_candidate.preserve_axes),
                "target_fields": list(session.top_schema_patch_candidate.target_fields),
                "metadata": dict(session.top_schema_patch_candidate.metadata),
            },
            "top_workflow_graph_patch_candidate": {
                "candidate_id": session.top_workflow_graph_patch_candidate.candidate_id,
                "candidate_kind": session.top_workflow_graph_patch_candidate.candidate_kind,
                "target_axes": list(session.top_workflow_graph_patch_candidate.target_axes),
                "preserve_axes": list(session.top_workflow_graph_patch_candidate.preserve_axes),
                "metadata": dict(session.top_workflow_graph_patch_candidate.metadata),
            },
            "preferred_commit_source": session.preferred_commit_source,
            "selected_graph_native_patch_candidate": {
                "candidate_id": session.selected_graph_native_patch_candidate.candidate_id,
                "candidate_kind": session.selected_graph_native_patch_candidate.candidate_kind,
                "target_axes": list(session.selected_graph_native_patch_candidate.target_axes),
                "preserve_axes": list(session.selected_graph_native_patch_candidate.preserve_axes),
                "metadata": dict(session.selected_graph_native_patch_candidate.metadata),
            },
            "graph_native_aligned_winner": bool(
                session.accepted_patch.metadata.get("graph_native_aligned_winner", False)
            ),
            "aligned_graph_candidate_id": str(
                session.accepted_patch.metadata.get("aligned_graph_candidate_id", "")
            ),
        },
    }
    workflow_topology_hints = {
        **workflow_topology_hints,
        "document_id": document.document_id,
        "document_region_label": document.metadata.get("region_label", ""),
    }
    return SurrogateWorkflowSnapshot(
        workflow_identity=workflow_identity,
        workflow_metadata=workflow_metadata,
        surrogate_payload=surrogate_payload,
        workflow_graph_placeholder=graph_placeholder,
        workflow_topology_hints=workflow_topology_hints,
        workflow_topology_entry_node_ids=list(graph_placeholder.entry_node_ids),
        workflow_topology_exit_node_ids=list(graph_placeholder.exit_node_ids),
        editable_scopes=editable_scopes,
        protected_scopes=protected_scopes,
    )
