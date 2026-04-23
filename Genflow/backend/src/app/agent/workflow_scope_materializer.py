from __future__ import annotations

from app.agent.memory import AgentSessionState
from app.agent.workflow_runtime_models import WorkflowScope


def materialize_editable_scopes(session: AgentSessionState) -> list[WorkflowScope]:
    target_fields = list(session.accepted_patch.target_fields)
    if not target_fields and session.selected_probe.probe_id:
        target_fields = list(session.selected_probe.target_axes)
    if not target_fields and session.current_schema_raw:
        target_fields = ["prompt", "model", "style", "lora"]
    if not target_fields:
        return []
    return [
        WorkflowScope(
            scope_id="editable-surrogate",
            scope_kind="schema_fields",
            label="Editable surrogate workflow fields",
            node_ids=list(target_fields),
            metadata={"source": "surrogate", "count": len(target_fields)},
        )
    ]


def materialize_protected_scopes(session: AgentSessionState) -> list[WorkflowScope]:
    protected_items = list(session.preserve_constraints)
    if not protected_items and session.selected_probe.probe_id:
        protected_items = list(session.selected_probe.preserve_axes)
    if not protected_items:
        return []
    return [
        WorkflowScope(
            scope_id="protected-surrogate",
            scope_kind="preserve_constraints",
            label="Protected surrogate workflow fields",
            node_ids=list(protected_items),
            metadata={"source": "constraints", "count": len(protected_items)},
        )
    ]


def materialize_workflow_scopes(session: AgentSessionState) -> tuple[list[WorkflowScope], list[WorkflowScope]]:
    return materialize_editable_scopes(session), materialize_protected_scopes(session)
