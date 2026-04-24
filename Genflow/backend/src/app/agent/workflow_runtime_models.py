from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class WorkflowIdentity:
    workflow_id: str = ""
    workflow_kind: str = ""
    workflow_version: str = ""


@dataclass
class WorkflowScope:
    scope_id: str = ""
    scope_kind: str = ""
    label: str = ""
    node_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecutionConfig:
    execution_kind: str = ""
    preview: bool = False
    backend_kind: str = ""
    workflow_profile: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowNodeRef:
    node_id: str = ""
    node_kind: str = ""
    role: str = ""
    label: str = ""
    upstream_ids: List[str] = field(default_factory=list)
    downstream_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTopologySlice:
    slice_id: str = ""
    region_label: str = ""
    slice_kind: str = ""
    node_refs: List[WorkflowNodeRef] = field(default_factory=list)
    entry_node_ids: List[str] = field(default_factory=list)
    exit_node_ids: List[str] = field(default_factory=list)
    edge_hints: List[Dict[str, str]] = field(default_factory=list)
    scope_partitions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowGraphPlaceholder:
    graph_id: str = ""
    graph_kind: str = ""
    entry_node_ids: List[str] = field(default_factory=list)
    exit_node_ids: List[str] = field(default_factory=list)
    node_refs: List[WorkflowNodeRef] = field(default_factory=list)
    topology_slices: List[WorkflowTopologySlice] = field(default_factory=list)
    adjacency_hints: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStateSnapshot:
    identity: WorkflowIdentity = field(default_factory=WorkflowIdentity)
    editable_scopes: List[WorkflowScope] = field(default_factory=list)
    protected_scopes: List[WorkflowScope] = field(default_factory=list)
    last_execution_config: WorkflowExecutionConfig = field(default_factory=WorkflowExecutionConfig)
    workflow_metadata: Dict[str, Any] = field(default_factory=dict)
    surrogate_payload: Dict[str, Any] = field(default_factory=dict)
    workflow_graph_placeholder: WorkflowGraphPlaceholder = field(default_factory=WorkflowGraphPlaceholder)
