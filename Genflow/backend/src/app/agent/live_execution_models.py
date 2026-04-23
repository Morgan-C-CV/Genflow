from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExecutionRequest:
    execution_kind: str = ""
    schema_snapshot: Dict[str, Any] = field(default_factory=dict)
    workflow_payload: Dict[str, Any] = field(default_factory=dict)
    reference_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreviewExecutionRequest(ExecutionRequest):
    preview_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommitExecutionRequest(ExecutionRequest):
    patch_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResponse:
    response_id: str = ""
    execution_kind: str = ""
    output_payload: Dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""
    changed_axes: List[str] = field(default_factory=list)
    preserved_axes: List[str] = field(default_factory=list)
    backend_artifacts: Dict[str, Any] = field(default_factory=dict)
    backend_metadata: Dict[str, Any] = field(default_factory=dict)
    comparison_notes: List[str] = field(default_factory=list)
