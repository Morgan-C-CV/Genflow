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
class WorkflowStateSnapshot:
    identity: WorkflowIdentity = field(default_factory=WorkflowIdentity)
    editable_scopes: List[WorkflowScope] = field(default_factory=list)
    protected_scopes: List[WorkflowScope] = field(default_factory=list)
    last_execution_config: WorkflowExecutionConfig = field(default_factory=WorkflowExecutionConfig)
    workflow_metadata: Dict[str, Any] = field(default_factory=dict)
    surrogate_payload: Dict[str, Any] = field(default_factory=dict)
