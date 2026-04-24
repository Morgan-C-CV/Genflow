import unittest
from dataclasses import asdict

from app.agent.memory import AgentMemoryService
from app.agent.workflow_runtime_models import WorkflowScope
from run_agent_demo import build_session_artifact_payload


class MemoryCompatibilityTest(unittest.TestCase):
    def test_create_session_keeps_existing_constructor_flow(self):
        memory = AgentMemoryService()
        session = memory.create_session("test intent")

        self.assertEqual(session.original_intent, "test intent")
        self.assertEqual(session.clarified_intent, "test intent")
        self.assertEqual(session.current_schema.prompt, "")
        self.assertEqual(session.current_schema_raw, "")
        self.assertEqual(session.selected_reference_bundle, {})
        self.assertEqual(session.workflow_id, "")
        self.assertEqual(session.workflow_state.identity.workflow_id, "")
        self.assertEqual(session.editable_scopes, [])
        self.assertEqual(session.protected_scopes, [])
        self.assertEqual(session.last_execution_config.execution_kind, "")
        self.assertEqual(session.workflow_metadata, {})
        self.assertEqual(session.workflow_graph_placeholder.graph_id, "")
        self.assertEqual(session.workflow_topology_hints, {})
        self.assertEqual(session.workflow_topology_entry_node_ids, [])
        self.assertEqual(session.workflow_topology_exit_node_ids, [])
        self.assertEqual(session.feedback_history, [])
        self.assertEqual(session.patch_history, [])
        self.assertFalse(session.continue_recommended)

    def test_save_and_get_session_preserve_new_default_fields(self):
        memory = AgentMemoryService()
        session = memory.create_session("another intent")
        session.feedback_history.append("too dark")
        session.selected_reference_ids.extend([1, 2, 3])
        memory.save_session(session)

        loaded = memory.get_session(session.session_id)
        self.assertEqual(loaded.feedback_history, ["too dark"])
        self.assertEqual(loaded.selected_reference_ids, [1, 2, 3])
        self.assertEqual(loaded.current_result_summary.summary_text, "")

    def test_session_artifact_payload_preserves_workflow_placeholder_fields(self):
        memory = AgentMemoryService()
        session = memory.create_session("workflow aware intent")
        session.workflow_id = "wf-local-1"
        session.workflow_identity.workflow_kind = "normalized_schema_surrogate"
        session.workflow_metadata["backend_kind"] = "workflow_shell"
        session.editable_scopes.append(WorkflowScope(scope_id="editable-1"))
        session.last_execution_config.execution_kind = "preview"

        payload = build_session_artifact_payload(session)
        serialized = asdict(session.workflow_state)

        self.assertEqual(payload["workflow_id"], "wf-local-1")
        self.assertEqual(payload["workflow_identity"]["workflow_kind"], "normalized_schema_surrogate")
        self.assertEqual(payload["workflow_metadata"]["backend_kind"], "workflow_shell")
        self.assertEqual(payload["editable_scopes"][0]["scope_id"], "editable-1")
        self.assertEqual(payload["last_execution_config"]["execution_kind"], "preview")
        self.assertIn("identity", payload["workflow_state"])
        self.assertEqual(serialized["identity"]["workflow_id"], "")


if __name__ == "__main__":
    unittest.main()
