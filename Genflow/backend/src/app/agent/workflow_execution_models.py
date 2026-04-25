from __future__ import annotations

"""Workflow-native shaped execution payloads for workflow substrate adapters."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class WorkflowExecutionArtifact:
    artifact_id: str = ""
    artifact_kind: str = ""
    uri: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecutionPayload:
    workflow_id: str = ""
    workflow_kind: str = ""
    workflow_version: str = ""
    execution_kind: str = ""
    preview: bool = False
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    entry_node_ids: List[str] = field(default_factory=list)
    exit_node_ids: List[str] = field(default_factory=list)
    execution_config: Dict[str, Any] = field(default_factory=dict)
    backend_metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[WorkflowExecutionArtifact] = field(default_factory=list)


@dataclass
class WorkflowPreviewRequest:
    workflow_payload: WorkflowExecutionPayload = field(default_factory=WorkflowExecutionPayload)
    preview_patch_spec: Dict[str, Any] = field(default_factory=dict)
    graph_patch_spec: Dict[str, Any] = field(default_factory=dict)
    reference_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowCommitRequest:
    workflow_payload: WorkflowExecutionPayload = field(default_factory=WorkflowExecutionPayload)
    committed_patch_spec: Dict[str, Any] = field(default_factory=dict)
    graph_patch_spec: Dict[str, Any] = field(default_factory=dict)
    reference_info: Dict[str, Any] = field(default_factory=dict)
