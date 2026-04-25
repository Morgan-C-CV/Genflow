import unittest

from app.agent.result_executor import ResultExecutor
from app.agent.runtime_models import NormalizedSchema, PreviewProbe, ResultPayload, ResultSummary
from app.agent.workflow_graph_patch_models import (
    WorkflowEdgePatch,
    WorkflowGraphPatch,
    WorkflowNodePatch,
)


class ResultExecutorTest(unittest.TestCase):
    def test_produce_initial_result_returns_structured_objects(self):
        executor = ResultExecutor(id_factory=lambda: "result-1")
        schema = NormalizedSchema(
            prompt="a vivid portrait",
            negative_prompt="blurry",
            model="sdxl-base",
            sampler="DPM++ 2M",
            style=["cinematic", "vivid"],
            lora=["portrait-helper"],
        )
        reference_bundle = {"references": [{"id": 101}, {"id": 202}]}

        payload, summary = executor.produce_initial_result(schema, reference_bundle)

        self.assertIsInstance(payload, ResultPayload)
        self.assertIsInstance(summary, ResultSummary)
        self.assertEqual(payload.result_id, "result-1")
        self.assertEqual(payload.result_type, "mock_initial_result")
        self.assertEqual(payload.content["model"], "sdxl-base")
        self.assertEqual(payload.content["reference_count"], 2)
        self.assertIn("model=sdxl-base", summary.summary_text)
        self.assertIn("style_count=2", summary.notes)

    def test_execute_preview_probe_returns_preview_result_without_mutating_schema(self):
        executor = ResultExecutor(id_factory=lambda: "preview-1")
        schema = NormalizedSchema(
            prompt="a vivid portrait",
            negative_prompt="blurry",
            model="sdxl-base",
            sampler="DPM++ 2M",
            style=["cinematic", "vivid"],
            lora=["portrait-helper"],
        )
        original_prompt = schema.prompt
        probe = PreviewProbe(
            probe_id="p_001",
            summary="style mismatch",
            target_axes=["style"],
            preserve_axes=["composition"],
            preview_execution_spec={"patch_family": "resource_shift", "reference_anchor": 7},
            source_kind="resource_shift",
        )

        preview_result = executor.execute_preview_probe(schema, probe)

        self.assertEqual(schema.prompt, original_prompt)
        self.assertEqual(preview_result.probe_id, "p_001")
        self.assertEqual(preview_result.payload.result_id, "preview-1")
        self.assertIn("style", preview_result.summary.changed_axes)
        self.assertIn("composition", preview_result.summary.preserved_axes)
        self.assertIn("resource_shift", preview_result.summary.summary_text)

    def test_execute_committed_patch_returns_updated_result_objects(self):
        executor = ResultExecutor(id_factory=lambda: "commit-1")
        schema = NormalizedSchema(
            prompt="a vivid portrait | make the style more vivid",
            model="sdxl-base-patched",
            style=["cinematic", "vivid"],
        )
        from app.agent.runtime_models import CommittedPatch

        patch = CommittedPatch(
            patch_id="cp_p_001",
            target_fields=["style", "model"],
            target_axes=["style"],
            preserve_axes=["composition"],
            changes={"style": ["cinematic", "vivid"], "model": "sdxl-base-patched"},
            rationale="adjust style and preserve composition",
        )

        payload, summary = executor.execute_committed_patch(schema, patch)

        self.assertEqual(payload.result_id, "commit-1")
        self.assertEqual(payload.result_type, "mock_committed_result")
        self.assertEqual(payload.content["patch_id"], "cp_p_001")
        self.assertIn("style", summary.changed_axes)
        self.assertIn("composition", summary.preserved_axes)
        self.assertIn("cp_p_001", summary.summary_text)

    def test_execute_committed_patch_reports_graph_native_input_when_provided(self):
        executor = ResultExecutor(id_factory=lambda: "commit-2")
        schema = NormalizedSchema(prompt="portrait", model="sdxl-base")
        from app.agent.runtime_models import CommittedPatch

        patch = CommittedPatch(patch_id="cp_p_002", target_fields=["style"], target_axes=["style"])
        graph_patch = WorkflowGraphPatch(
            workflow_id="workflow-1",
            patch_id="wgp_002",
            node_patches=[WorkflowNodePatch(node_id="render.model", operation="update")],
            edge_patches=[WorkflowEdgePatch(edge_id="edge-1", operation="retarget")],
        )

        payload, summary = executor.execute_committed_patch(
            schema,
            patch,
            graph_patch=graph_patch,
            commit_execution_mode="graph_native_execution_handoff",
            commit_execution_authority="graph_authoritative",
            commit_execution_implementation_mode="graph_primary_execution",
        )

        self.assertEqual(payload.content["graph_patch_input_id"], "wgp_002")
        self.assertEqual(payload.content["commit_execution_authority"], "graph_authoritative")
        self.assertEqual(payload.content["request_primary_plan_kind"], "graph_primary")
        self.assertEqual(payload.content["commit_execution_implementation_mode"], "graph_primary_execution")
        self.assertTrue(payload.content["request_graph_native_realization"])
        self.assertEqual(payload.content["requested_backend_execution_mode"], "graph_primary_backend_execution")
        self.assertTrue(payload.content["backend_graph_primary_capable"])
        self.assertTrue(payload.content["backend_graph_native_realization_supported"])
        self.assertTrue(payload.content["backend_graph_commit_payload_supplied"])
        self.assertTrue(payload.content["backend_graph_commit_payload_consumed"])
        self.assertTrue(payload.content["backend_graph_native_execution_realized"])
        self.assertEqual(
            payload.content["backend_graph_native_realization_reason"],
            "graph_native_realization_achieved",
        )
        self.assertEqual(payload.content["accepted_backend_execution_mode"], "graph_primary_backend_execution")
        self.assertEqual(payload.content["realized_backend_execution_mode"], "graph_primary_backend_execution")
        self.assertEqual(payload.content["execution_behavior_branch"], "graph_primary_execution_branch")
        self.assertEqual(payload.content["graph_driven_node_count"], 1)
        self.assertTrue(payload.artifacts["backend_metadata"]["graph_native_artifact_input_received"])
        self.assertEqual(
            payload.artifacts["backend_metadata"]["commit_execution_mode"],
            "graph_native_execution_handoff",
        )
        self.assertEqual(
            payload.artifacts["backend_metadata"]["commit_execution_authority"],
            "graph_authoritative",
        )
        self.assertEqual(
            payload.artifacts["backend_metadata"]["commit_execution_implementation_mode"],
            "graph_primary_execution",
        )
        self.assertEqual(payload.artifacts["backend_metadata"]["request_primary_plan_kind"], "graph_primary")
        self.assertTrue(payload.artifacts["backend_metadata"]["request_graph_native_realization"])
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_primary_capable"])
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_native_realization_supported"])
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_commit_payload_supplied"])
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_commit_payload_consumed"])
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_native_execution_realized"])
        self.assertEqual(
            payload.artifacts["backend_metadata"]["backend_graph_native_realization_reason"],
            "graph_native_realization_achieved",
        )
        self.assertEqual(
            payload.artifacts["backend_metadata"]["accepted_backend_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            payload.artifacts["backend_metadata"]["realized_backend_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            payload.artifacts["backend_metadata"]["execution_behavior_branch"],
            "graph_primary_execution_branch",
        )
        self.assertTrue(payload.artifacts["backend_metadata"]["graph_primary_behavior_applied"])
        self.assertIn("graph_native_artifact_input_received=True", summary.notes)
        self.assertIn("commit_execution_authority=graph_authoritative", summary.notes)
        self.assertIn("commit_execution_implementation_mode=graph_primary_execution", summary.notes)
        self.assertIn("request_graph_native_realization=True", summary.notes)
        self.assertIn("request_primary_plan_kind=graph_primary", summary.notes)
        self.assertIn("backend_graph_primary_capable=True", summary.notes)
        self.assertIn("backend_graph_native_realization_supported=True", summary.notes)
        self.assertIn("backend_graph_commit_payload_supplied=True", summary.notes)
        self.assertIn("backend_graph_commit_payload_consumed=True", summary.notes)
        self.assertIn("backend_graph_native_execution_realized=True", summary.notes)
        self.assertIn("backend_graph_native_realization_reason=graph_native_realization_achieved", summary.notes)
        self.assertIn("requested_backend_execution_mode=graph_primary_backend_execution", summary.notes)
        self.assertIn("accepted_backend_execution_mode=graph_primary_backend_execution", summary.notes)
        self.assertIn("realized_backend_execution_mode=graph_primary_backend_execution", summary.notes)
        self.assertIn("execution_behavior_branch=graph_primary_execution_branch", summary.notes)
        self.assertIn("Mock graph-primary execution branch", summary.summary_text)

    def test_execute_committed_patch_can_downgrade_realized_backend_mode(self):
        executor = ResultExecutor(id_factory=lambda: "commit-3")
        schema = NormalizedSchema(prompt="portrait", model="sdxl-base")
        from app.agent.runtime_models import CommittedPatch

        patch = CommittedPatch(patch_id="cp_p_003", target_fields=["style"], target_axes=["style"])
        graph_patch = WorkflowGraphPatch(
            workflow_id="workflow-1",
            patch_id="wgp_003",
            node_patches=[WorkflowNodePatch(node_id="render.model", operation="update")],
        )

        payload, summary = executor.execute_committed_patch(
            schema,
            patch,
            graph_patch=graph_patch,
            commit_execution_mode="graph_native_execution_handoff",
            commit_execution_authority="graph_authoritative",
            commit_execution_implementation_mode="graph_primary_execution",
        )

        self.assertEqual(payload.content["requested_backend_execution_mode"], "graph_primary_backend_execution")
        self.assertTrue(payload.content["request_graph_native_realization"])
        self.assertTrue(payload.content["backend_graph_primary_capable"])
        self.assertTrue(payload.content["backend_graph_native_realization_supported"])
        self.assertTrue(payload.content["backend_graph_commit_payload_supplied"])
        self.assertTrue(payload.content["backend_graph_commit_payload_consumed"])
        self.assertFalse(payload.content["backend_graph_native_execution_realized"])
        self.assertEqual(
            payload.content["backend_graph_native_realization_reason"],
            "insufficient_graph_payload_completeness",
        )
        self.assertEqual(payload.content["accepted_backend_execution_mode"], "graph_primary_backend_execution")
        self.assertEqual(payload.content["realized_backend_execution_mode"], "schema_compatible_backend_execution")
        self.assertEqual(payload.content["execution_behavior_branch"], "schema_primary_execution_branch")
        self.assertEqual(payload.content["graph_driven_node_count"], 0)
        self.assertEqual(
            payload.artifacts["backend_metadata"]["realized_backend_execution_mode"],
            "schema_compatible_backend_execution",
        )
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_primary_capable"])
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_native_realization_supported"])
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_commit_payload_supplied"])
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_commit_payload_consumed"])
        self.assertFalse(payload.artifacts["backend_metadata"]["backend_graph_native_execution_realized"])
        self.assertEqual(
            payload.artifacts["backend_metadata"]["backend_graph_native_realization_reason"],
            "insufficient_graph_payload_completeness",
        )
        self.assertIn("realized_backend_execution_mode=schema_compatible_backend_execution", summary.notes)
        self.assertIn("backend_graph_native_realization_supported=True", summary.notes)
        self.assertIn("backend_graph_native_execution_realized=False", summary.notes)
        self.assertIn(
            "backend_graph_native_realization_reason=insufficient_graph_payload_completeness",
            summary.notes,
        )

    def test_execute_committed_patch_distinguishes_backend_not_capable_from_not_realized(self):
        executor = ResultExecutor(id_factory=lambda: "commit-4")
        schema = NormalizedSchema(prompt="portrait", model="sdxl-base")
        from app.agent.runtime_models import CommittedPatch

        patch = CommittedPatch(patch_id="cp_p_004", target_fields=["style"], target_axes=["style"])
        graph_patch = WorkflowGraphPatch(
            workflow_id="workflow-1",
            patch_id="wgp_004",
        )

        payload, summary = executor.execute_committed_patch(
            schema,
            patch,
            graph_patch=graph_patch,
            commit_execution_mode="graph_native_execution_handoff",
            commit_execution_authority="graph_authoritative",
            commit_execution_implementation_mode="graph_primary_execution",
        )

        self.assertEqual(payload.content["requested_backend_execution_mode"], "graph_primary_backend_execution")
        self.assertTrue(payload.content["request_graph_native_realization"])
        self.assertFalse(payload.content["backend_graph_primary_capable"])
        self.assertFalse(payload.content["backend_graph_native_realization_supported"])
        self.assertTrue(payload.content["backend_graph_commit_payload_supplied"])
        self.assertTrue(payload.content["backend_graph_commit_payload_consumed"])
        self.assertFalse(payload.content["backend_graph_native_execution_realized"])
        self.assertEqual(
            payload.content["backend_graph_native_realization_reason"],
            "unsupported_backend_capability",
        )
        self.assertEqual(payload.content["accepted_backend_execution_mode"], "schema_compatible_backend_execution")
        self.assertEqual(payload.content["realized_backend_execution_mode"], "schema_compatible_backend_execution")
        self.assertEqual(payload.content["execution_behavior_branch"], "schema_primary_execution_branch")
        self.assertFalse(payload.artifacts["backend_metadata"]["backend_graph_primary_capable"])
        self.assertFalse(payload.artifacts["backend_metadata"]["backend_graph_native_realization_supported"])
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_commit_payload_supplied"])
        self.assertTrue(payload.artifacts["backend_metadata"]["backend_graph_commit_payload_consumed"])
        self.assertFalse(payload.artifacts["backend_metadata"]["backend_graph_native_execution_realized"])
        self.assertEqual(
            payload.artifacts["backend_metadata"]["backend_graph_native_realization_reason"],
            "unsupported_backend_capability",
        )
        self.assertIn("backend_graph_primary_capable=False", summary.notes)
        self.assertIn("backend_graph_native_realization_supported=False", summary.notes)
        self.assertIn("backend_graph_native_execution_realized=False", summary.notes)
        self.assertIn(
            "backend_graph_native_realization_reason=unsupported_backend_capability",
            summary.notes,
        )


if __name__ == "__main__":
    unittest.main()
