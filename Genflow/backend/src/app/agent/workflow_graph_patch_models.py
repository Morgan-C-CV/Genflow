from __future__ import annotations

"""Graph-native patch intent models projected from the current repair loop."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class WorkflowNodePatch:
    node_id: str = ""
    operation: str = ""
    target_fields: List[str] = field(default_factory=list)
    target_axes: List[str] = field(default_factory=list)
    changes: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowEdgePatch:
    edge_id: str = ""
    operation: str = ""
    target_axes: List[str] = field(default_factory=list)
    preserve_axes: List[str] = field(default_factory=list)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowRegionPatch:
    region_id: str = ""
    operation: str = ""
    target_axes: List[str] = field(default_factory=list)
    preserve_axes: List[str] = field(default_factory=list)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowGraphPatch:
    workflow_id: str = ""
    patch_id: str = ""
    patch_kind: str = ""
    node_patches: List[WorkflowNodePatch] = field(default_factory=list)
    edge_patches: List[WorkflowEdgePatch] = field(default_factory=list)
    region_patches: List[WorkflowRegionPatch] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
