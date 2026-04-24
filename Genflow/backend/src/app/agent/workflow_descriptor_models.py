"""Surrogate workflow descriptor layer.

This layer is the compact workflow-aware summary extracted from session state.
It intentionally sits between session state and the richer surrogate document /
graph / snapshot builders so those downstream builders do not re-read scattered
session fields independently.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SurrogateBenchmarkComparisonFootprint:
    benchmark_source: str = ""
    compared_anchor_ids: List[int] = field(default_factory=list)
    compared_candidate_ids: List[str] = field(default_factory=list)
    focus_axes: List[str] = field(default_factory=list)
    preserve_axes: List[str] = field(default_factory=list)
    confidence_hint: float = 0.0


@dataclass
class SurrogateExecutionDescriptor:
    execution_kind: str = ""
    preview: bool = False
    backend_kind: str = ""
    workflow_profile: str = ""
    current_result_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurrogateRepairDescriptor:
    has_feedback: bool = False
    feedback_count: int = 0
    dissatisfaction_axes: List[str] = field(default_factory=list)
    preserve_constraints: List[str] = field(default_factory=list)
    selected_probe_id: str = ""
    probe_target_axes: List[str] = field(default_factory=list)
    probe_preserve_axes: List[str] = field(default_factory=list)
    accepted_patch_id: str = ""
    patch_target_fields: List[str] = field(default_factory=list)
    current_uncertainty_estimate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurrogateWorkflowDescriptor:
    workflow_id: str = ""
    workflow_kind: str = ""
    schema_prompt: str = ""
    schema_model: str = ""
    schema_negative_prompt: str = ""
    schema_sampler: str = ""
    schema_style: List[str] = field(default_factory=list)
    schema_lora: List[str] = field(default_factory=list)
    selected_gallery_index: int | None = None
    selected_reference_ids: List[int] = field(default_factory=list)
    reference_bundle: Dict[str, Any] = field(default_factory=dict)
    execution: SurrogateExecutionDescriptor = field(default_factory=SurrogateExecutionDescriptor)
    repair: SurrogateRepairDescriptor = field(default_factory=SurrogateRepairDescriptor)
    benchmark_comparison: SurrogateBenchmarkComparisonFootprint = field(
        default_factory=SurrogateBenchmarkComparisonFootprint
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
