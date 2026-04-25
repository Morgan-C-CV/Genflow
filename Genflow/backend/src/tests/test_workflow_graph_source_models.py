import unittest

from app.agent.workflow_graph_source_models import (
    WorkflowGraphEdge,
    WorkflowGraphNode,
    WorkflowGraphRegion,
    WorkflowGraphSource,
)


class WorkflowGraphSourceModelsTest(unittest.TestCase):
    def test_graph_source_defaults_are_safe(self):
        graph_source = WorkflowGraphSource()

        self.assertEqual(graph_source.workflow_id, "")
        self.assertEqual(graph_source.nodes, [])
        self.assertEqual(graph_source.edges, [])
        self.assertEqual(graph_source.regions, [])
        self.assertEqual(graph_source.metadata, {})

    def test_graph_source_components_are_typed_and_serializable(self):
        node = WorkflowGraphNode(node_id="intent.prompt", node_type="prompt_input", role="input")
        edge = WorkflowGraphEdge(edge_id="e1", source_node_id="intent.prompt", target_node_id="result.output")
        region = WorkflowGraphRegion(region_id="r1", region_type="initial", label="Initial")
        graph_source = WorkflowGraphSource(
            workflow_id="workflow-1",
            nodes=[node],
            edges=[edge],
            regions=[region],
        )

        self.assertEqual(graph_source.nodes[0].node_type, "prompt_input")
        self.assertEqual(graph_source.edges[0].source_node_id, "intent.prompt")
        self.assertEqual(graph_source.regions[0].label, "Initial")


if __name__ == "__main__":
    unittest.main()
