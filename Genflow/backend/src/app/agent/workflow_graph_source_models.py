from __future__ import annotations

"""Workflow-native graph source models used before execution payload shaping."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class WorkflowGraphNode:
    node_id: str = ""
    node_type: str = ""
    role: str = ""
    label: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowGraphEdge:
    edge_id: str = ""
    source_node_id: str = ""
    target_node_id: str = ""
    edge_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowGraphRegion:
    region_id: str = ""
    region_type: str = ""
    label: str = ""
    node_ids: List[str] = field(default_factory=list)
    entry_node_ids: List[str] = field(default_factory=list)
    exit_node_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowGraphSource:
    workflow_id: str = ""
    workflow_kind: str = ""
    workflow_version: str = ""
    backend_kind: str = ""
    workflow_profile: str = ""
    nodes: List[WorkflowGraphNode] = field(default_factory=list)
    edges: List[WorkflowGraphEdge] = field(default_factory=list)
    regions: List[WorkflowGraphRegion] = field(default_factory=list)
    entry_node_ids: List[str] = field(default_factory=list)
    exit_node_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
