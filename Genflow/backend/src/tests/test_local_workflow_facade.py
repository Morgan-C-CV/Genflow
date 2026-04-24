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
                },
            ),
        )

        self.assertEqual(response.execution_kind, "preview")
        self.assertEqual(response.response_id, "local-preview-p_001")
        self.assertEqual(response.changed_axes, ["style"])
        self.assertEqual(response.preserved_axes, ["composition"])

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
                },
            ),
        )

        self.assertEqual(response.execution_kind, "commit")
        self.assertEqual(response.response_id, "local-commit-cp_001")
        self.assertEqual(response.output_payload["patch_id"], "cp_001")
        self.assertEqual(response.changed_axes, ["style"])

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
