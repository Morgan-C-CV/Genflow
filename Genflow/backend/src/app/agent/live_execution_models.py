from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.agent.live_backend_errors import LiveBackendResponseError


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


def normalize_execution_response(raw_response: Any, expected_kind: str) -> ExecutionResponse:
    if isinstance(raw_response, ExecutionResponse):
        response = raw_response
    elif isinstance(raw_response, dict):
        response = ExecutionResponse(
            response_id=str(raw_response.get("response_id", "")),
            execution_kind=str(raw_response.get("execution_kind", "")),
            output_payload=raw_response.get("output_payload", {}),
            summary_text=str(raw_response.get("summary_text", "")),
            changed_axes=_normalize_string_list(raw_response.get("changed_axes", [])),
            preserved_axes=_normalize_string_list(raw_response.get("preserved_axes", [])),
            backend_artifacts=_normalize_dict(raw_response.get("backend_artifacts", {}), "backend_artifacts"),
            backend_metadata=_normalize_dict(raw_response.get("backend_metadata", {}), "backend_metadata"),
            comparison_notes=_normalize_string_list(raw_response.get("comparison_notes", [])),
        )
    else:
        raise LiveBackendResponseError(
            f"Backend response must be ExecutionResponse or dict, got {type(raw_response).__name__}."
        )

    if not response.response_id:
        raise LiveBackendResponseError("Backend response is missing response_id.")
    if response.execution_kind != expected_kind:
        raise LiveBackendResponseError(
            f"Backend response execution_kind mismatch: expected {expected_kind}, got {response.execution_kind or 'empty'}."
        )
    if not isinstance(response.output_payload, dict):
        raise LiveBackendResponseError("Backend response output_payload must be a dict.")

    response.changed_axes = _normalize_string_list(response.changed_axes)
    response.preserved_axes = _normalize_string_list(response.preserved_axes)
    response.comparison_notes = _normalize_string_list(response.comparison_notes)
    response.backend_artifacts = _normalize_dict(response.backend_artifacts, "backend_artifacts")
    response.backend_metadata = _normalize_dict(response.backend_metadata, "backend_metadata")
    return response


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value] if value else []
    raise LiveBackendResponseError(f"Expected list-like string field, got {type(value).__name__}.")


def _normalize_dict(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LiveBackendResponseError(f"Backend response {field_name} must be a dict.")
    return dict(value)
