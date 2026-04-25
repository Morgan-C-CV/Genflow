import unittest

from app.agent.live_execution_models import CommitExecutionRequest, ExecutionRequest, PreviewExecutionRequest
from app.agent.local_workflow_facade import LocalWorkflowFacade


class LocalWorkflowFacadeTest(unittest.TestCase):
    def test_initial_path_returns_normalized_execution_response(self):
        facade = LocalWorkflowFacade()

        response = facade.run(
            "initial",
            ExecutionRequest(
                execution_kind="initial",
                schema_snapshot={"prompt": "portrait", "model": "sdxl-base"},
                workflow_payload={
                    "workflow_id": "workflow-1",
                    "workflow_kind": "workflow_native_surrogate",
                    "nodes": [{"node_id": "intent.prompt"}],
                    "edges": [{"edge_id": "intent.prompt->result.output"}],
                    "execution_config": {"execution_kind": "initial"},
                },
                reference_info={"reference_ids": [101, 202]},
            ),
        )

        self.assertEqual(response.execution_kind, "initial")
        self.assertEqual(response.response_id, "local-initial")
        self.assertEqual(response.output_payload["model"], "sdxl-base")
        self.assertEqual(response.output_payload["reference_count"], 2)
        self.assertEqual(response.backend_metadata["backend"], "local_workflow_facade")

    def test_preview_path_returns_deterministic_preview_response(self):
        facade = LocalWorkflowFacade()

        response = facade.run(
            "preview",
            PreviewExecutionRequest(
                execution_kind="preview",
                workflow_payload={
                    "workflow_id": "workflow-1",
                    "workflow_kind": "workflow_native_surrogate",
                    "nodes": [{"node_id": "probe.p_001"}],
                    "edges": [{"edge_id": "probe.p_001->result.output"}],
                    "execution_config": {"execution_kind": "preview", "preview": True},
                },
                preview_spec={
                    "probe_id": "p_001",
                    "target_axes": ["style"],
                    "preserve_axes": ["composition"],
                    "source_kind": "schema_variation",
                    "graph_patch_spec": {
                        "patch_id": "preview:p_001",
                        "node_patches": [{"node_id": "render.model"}],
                    },
                },
            ),
        )

        self.assertEqual(response.execution_kind, "preview")
        self.assertEqual(response.response_id, "local-preview-p_001")
        self.assertEqual(response.changed_axes, ["style"])
        self.assertEqual(response.preserved_axes, ["composition"])
        self.assertEqual(response.backend_metadata["graph_patch_id"], "preview:p_001")
        self.assertIn("graph_patch_id=preview:p_001", response.comparison_notes)

    def test_commit_path_returns_deterministic_commit_response(self):
        facade = LocalWorkflowFacade()

        response = facade.run(
            "commit",
            CommitExecutionRequest(
                execution_kind="commit",
                workflow_payload={
                    "workflow_id": "workflow-1",
                    "workflow_kind": "workflow_native_surrogate",
                    "nodes": [{"node_id": "patch.cp_001"}],
                    "edges": [{"edge_id": "patch.cp_001->result.output"}],
                    "execution_config": {"execution_kind": "commit"},
                },
                patch_spec={
                    "patch_id": "cp_001",
                    "target_fields": ["style"],
                    "target_axes": ["style"],
                    "preserve_axes": ["composition"],
                    "rationale": "style shift",
                    "commit_source_payload": {
                        "commit_execution_mode": "graph_native_execution_handoff",
                        "commit_execution_authority": "graph_authoritative",
                        "request_primary_plan_kind": "graph_primary",
                        "preferred_commit_source": "graph",
                        "selected_workflow_graph_patch_id": "wgp_001",
                        "top_schema_patch_id": "cp_001",
                        "top_graph_patch_candidate_id": "wgc_001",
                    },
                    "primary_commit_plan": {
                        "plan_kind": "graph_primary",
                        "graph_patch_id": "wgp_001",
                    },
                    "graph_patch_spec": {
                        "patch_id": "cp_001",
                        "node_patches": [{"node_id": "render.model"}],
                    },
                },
            ),
        )

        self.assertEqual(response.execution_kind, "commit")
        self.assertEqual(response.response_id, "local-commit-cp_001")
        self.assertEqual(response.output_payload["patch_id"], "cp_001")
        self.assertTrue(response.output_payload["graph_native_artifact_input_received"])
        self.assertEqual(response.output_payload["request_primary_plan_kind"], "graph_primary")
        self.assertEqual(response.output_payload["execution_behavior_branch"], "graph_primary_execution_branch")
        self.assertEqual(response.output_payload["graph_driven_node_count"], 1)
        self.assertEqual(response.changed_axes, ["style"])
        self.assertEqual(response.backend_metadata["graph_patch_id"], "cp_001")
        self.assertEqual(response.backend_metadata["commit_execution_mode"], "graph_native_execution_handoff")
        self.assertEqual(response.backend_metadata["commit_execution_authority"], "graph_authoritative")
        self.assertEqual(response.backend_metadata["request_primary_plan_kind"], "graph_primary")
        self.assertEqual(response.backend_metadata["execution_behavior_branch"], "graph_primary_execution_branch")
        self.assertTrue(response.backend_metadata["graph_primary_behavior_applied"])
        self.assertTrue(response.backend_metadata["graph_native_artifact_input_received"])
        self.assertEqual(response.backend_metadata["preferred_commit_source"], "graph")
        self.assertEqual(response.backend_metadata["selected_workflow_graph_patch_id"], "wgp_001")
        self.assertIn("Local workflow facade ran graph-primary execution branch", response.summary_text)
        self.assertIn("graph_patch_id=cp_001", response.comparison_notes)
        self.assertIn("request_primary_plan_kind=graph_primary", response.comparison_notes)
        self.assertIn("execution_behavior_branch=graph_primary_execution_branch", response.comparison_notes)
        self.assertIn("commit_execution_mode=graph_native_execution_handoff", response.comparison_notes)
        self.assertIn("commit_execution_authority=graph_authoritative", response.comparison_notes)
        self.assertIn("graph_native_artifact_input_received=True", response.comparison_notes)
        self.assertIn("preferred_commit_source=graph", response.comparison_notes)

    def test_facade_rejects_preview_without_graph_patch_spec(self):
        facade = LocalWorkflowFacade()

        with self.assertRaisesRegex(ValueError, "preview request graph_patch_spec must include patch_id."):
            facade.run(
                "preview",
                PreviewExecutionRequest(
                    execution_kind="preview",
                    workflow_payload={
                        "workflow_id": "workflow-1",
                        "workflow_kind": "workflow_native_surrogate",
                        "nodes": [{"node_id": "probe.p_001"}],
                        "edges": [{"edge_id": "probe.p_001->result.output"}],
                        "execution_config": {"execution_kind": "preview", "preview": True},
                    },
                    preview_spec={
                        "probe_id": "p_001",
                        "target_axes": ["style"],
                        "preserve_axes": ["composition"],
                    },
                ),
            )

    def test_facade_rejects_missing_workflow_native_payload_shape(self):
        facade = LocalWorkflowFacade()

        with self.assertRaisesRegex(ValueError, "workflow_payload must include a nodes list."):
            facade.run(
                "initial",
                ExecutionRequest(
                    execution_kind="initial",
                    workflow_payload={},
                ),
            )


if __name__ == "__main__":
    unittest.main()
