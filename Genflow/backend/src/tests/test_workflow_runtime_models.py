import unittest
from dataclasses import asdict

from app.agent.workflow_runtime_models import (
    WorkflowExecutionConfig,
    WorkflowIdentity,
    WorkflowScope,
    WorkflowStateSnapshot,
)


class WorkflowRuntimeModelsTest(unittest.TestCase):
    def test_workflow_runtime_models_construct_with_defaults(self):
        identity = WorkflowIdentity()
        scope = WorkflowScope()
        config = WorkflowExecutionConfig()
        snapshot = WorkflowStateSnapshot()

        self.assertEqual(identity.workflow_id, "")
        self.assertEqual(scope.node_ids, [])
        self.assertEqual(config.parameters, {})
        self.assertEqual(snapshot.editable_scopes, [])
        self.assertEqual(snapshot.workflow_metadata, {})

    def test_workflow_runtime_models_default_factories_are_not_shared(self):
        scope_a = WorkflowScope()
        scope_b = WorkflowScope()
        scope_a.node_ids.append("node-1")

        snapshot_a = WorkflowStateSnapshot()
        snapshot_b = WorkflowStateSnapshot()
        snapshot_a.workflow_metadata["kind"] = "surrogate"

        self.assertEqual(scope_b.node_ids, [])
        self.assertEqual(snapshot_b.workflow_metadata, {})

    def test_workflow_snapshot_is_serializable(self):
        snapshot = WorkflowStateSnapshot(
            identity=WorkflowIdentity(workflow_id="wf-1", workflow_kind="surrogate"),
            editable_scopes=[WorkflowScope(scope_id="editable-1", node_ids=["n1"])],
        )

        payload = asdict(snapshot)

        self.assertEqual(payload["identity"]["workflow_id"], "wf-1")
        self.assertEqual(payload["editable_scopes"][0]["node_ids"], ["n1"])


if __name__ == "__main__":
    unittest.main()
