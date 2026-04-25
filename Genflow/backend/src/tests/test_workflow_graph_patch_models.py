import unittest

from app.agent.workflow_graph_patch_models import (
    WorkflowEdgePatch,
    WorkflowGraphPatch,
    WorkflowNodePatch,
    WorkflowRegionPatch,
)


class WorkflowGraphPatchModelsTest(unittest.TestCase):
    def test_graph_patch_defaults_are_safe(self):
        patch = WorkflowGraphPatch()

        self.assertEqual(patch.workflow_id, "")
        self.assertEqual(patch.patch_id, "")
        self.assertEqual(patch.node_patches, [])
        self.assertEqual(patch.edge_patches, [])
        self.assertEqual(patch.region_patches, [])

    def test_graph_patch_components_are_typed(self):
        patch = WorkflowGraphPatch(
            workflow_id="workflow-1",
            patch_id="cp_001",
            node_patches=[WorkflowNodePatch(node_id="render.model", operation="update_node_config")],
            edge_patches=[WorkflowEdgePatch(edge_id="e1", operation="rebind_flow_constraints")],
            region_patches=[WorkflowRegionPatch(region_id="r1", operation="update_region_intent")],
        )

        self.assertEqual(patch.node_patches[0].node_id, "render.model")
        self.assertEqual(patch.edge_patches[0].edge_id, "e1")
        self.assertEqual(patch.region_patches[0].region_id, "r1")


if __name__ == "__main__":
    unittest.main()
