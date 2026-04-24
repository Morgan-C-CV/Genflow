from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SurrogateWorkflowNode:
    node_id: str = ""
    node_kind: str = ""
    role: str = ""
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurrogateWorkflowEdge:
    edge_id: str = ""
    source_node_id: str = ""
    target_node_id: str = ""
    edge_kind: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurrogateWorkflowRegion:
    region_id: str = ""
    region_label: str = ""
    region_kind: str = ""
    node_ids: List[str] = field(default_factory=list)
    entry_node_ids: List[str] = field(default_factory=list)
    exit_node_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurrogateWorkflowDocument:
    document_id: str = ""
    workflow_id: str = ""
    workflow_kind: str = ""
    execution_kind: str = ""
    preview: bool = False
    backend_kind: str = ""
    workflow_profile: str = ""
    nodes: List[SurrogateWorkflowNode] = field(default_factory=list)
    edges: List[SurrogateWorkflowEdge] = field(default_factory=list)
    regions: List[SurrogateWorkflowRegion] = field(default_factory=list)
    entry_node_ids: List[str] = field(default_factory=list)
    exit_node_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
