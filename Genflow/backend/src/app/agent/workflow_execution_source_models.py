from __future__ import annotations

"""Typed execution input sources for workflow execution payload builders."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.agent.runtime_models import CommittedPatch, NormalizedSchema, PreviewProbe


@dataclass
class WorkflowExecutionSource:
    workflow_id: str = ""
    workflow_kind: str = ""
    workflow_version: str = ""
    execution_kind: str = ""
    preview: bool = False
    schema: NormalizedSchema = field(default_factory=NormalizedSchema)
    selected_gallery_index: int | None = None
    selected_reference_ids: List[int] = field(default_factory=list)
    selected_reference_bundle: Dict[str, Any] = field(default_factory=dict)
    current_result_id: str = ""
    backend_kind: str = ""
    workflow_profile: str = ""
    latest_feedback: str = ""
    feedback_count: int = 0
    dissatisfaction_axes: List[str] = field(default_factory=list)
    preserve_constraints: List[str] = field(default_factory=list)
    current_uncertainty_estimate: float = 0.0
    benchmark_source: str = ""
    compared_anchor_ids: List[int] = field(default_factory=list)
    compared_candidate_ids: List[str] = field(default_factory=list)
    benchmark_focus_axes: List[str] = field(default_factory=list)
    benchmark_preserve_axes: List[str] = field(default_factory=list)
    benchmark_confidence_hint: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowPreviewSource(WorkflowExecutionSource):
    selected_probe: PreviewProbe = field(default_factory=PreviewProbe)


@dataclass
class WorkflowCommitSource(WorkflowExecutionSource):
    selected_probe: PreviewProbe = field(default_factory=PreviewProbe)
    accepted_patch: CommittedPatch = field(default_factory=CommittedPatch)
