import unittest
from dataclasses import asdict

from app.agent.workflow_runtime_models import (
    WorkflowExecutionConfig,
    WorkflowGraphPlaceholder,
    WorkflowIdentity,
    WorkflowNodeRef,
    WorkflowScope,
    WorkflowStateSnapshot,
    WorkflowTopologySlice,
)


class WorkflowRuntimeModelsTest(unittest.TestCase):
    def test_workflow_runtime_models_construct_with_defaults(self):
        identity = WorkflowIdentity()
        node = WorkflowNodeRef()
        scope = WorkflowScope()
        config = WorkflowExecutionConfig()
        topology_slice = WorkflowTopologySlice()
        graph = WorkflowGraphPlaceholder()
        snapshot = WorkflowStateSnapshot()

        self.assertEqual(identity.workflow_id, "")
        self.assertEqual(node.node_id, "")
        self.assertEqual(scope.node_ids, [])
        self.assertEqual(config.parameters, {})
        self.assertEqual(topology_slice.node_refs, [])
        self.assertEqual(graph.adjacency_hints, [])
        self.assertEqual(snapshot.editable_scopes, [])
        self.assertEqual(snapshot.workflow_metadata, {})
        self.assertEqual(snapshot.workflow_graph_placeholder.graph_id, "")

    def test_workflow_runtime_models_default_factories_are_not_shared(self):
        scope_a = WorkflowScope()
        scope_b = WorkflowScope()
        scope_a.node_ids.append("node-1")

        snapshot_a = WorkflowStateSnapshot()
        snapshot_b = WorkflowStateSnapshot()
        snapshot_a.workflow_metadata["kind"] = "surrogate"
        snapshot_a.workflow_graph_placeholder.metadata["region"] = "initial"

        self.assertEqual(scope_b.node_ids, [])
        self.assertEqual(snapshot_b.workflow_metadata, {})
        self.assertEqual(snapshot_b.workflow_graph_placeholder.metadata, {})

    def test_workflow_snapshot_is_serializable(self):
        snapshot = WorkflowStateSnapshot(
            identity=WorkflowIdentity(workflow_id="wf-1", workflow_kind="surrogate"),
            editable_scopes=[WorkflowScope(scope_id="editable-1", node_ids=["n1"])],
            workflow_graph_placeholder=WorkflowGraphPlaceholder(
                graph_id="graph-1",
                node_refs=[WorkflowNodeRef(node_id="node-1")],
                topology_slices=[WorkflowTopologySlice(slice_id="slice-1")],
            ),
        )

        payload = asdict(snapshot)

        self.assertEqual(payload["identity"]["workflow_id"], "wf-1")
        self.assertEqual(payload["editable_scopes"][0]["node_ids"], ["n1"])
        self.assertEqual(payload["workflow_graph_placeholder"]["graph_id"], "graph-1")
        self.assertEqual(payload["workflow_graph_placeholder"]["node_refs"][0]["node_id"], "node-1")


if __name__ == "__main__":
    unittest.main()
