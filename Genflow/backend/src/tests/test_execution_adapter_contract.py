import unittest

from app.agent.execution_adapter import ExecutionAdapter
from app.agent.result_executor import ResultExecutor
from app.agent.runtime_models import CommittedPatch, NormalizedSchema, PreviewProbe


class ExecutionAdapterContractTest(unittest.TestCase):
    def test_result_executor_satisfies_execution_adapter_contract(self):
        adapter = ResultExecutor(id_factory=lambda: "adapter-id")
        self.assertIsInstance(adapter, ExecutionAdapter)

    def test_mock_adapter_supports_all_contract_methods(self):
        ids = iter(["initial-1", "preview-1", "commit-1"])
        adapter = ResultExecutor(id_factory=lambda: next(ids))
        schema = NormalizedSchema(
            prompt="cinematic portrait",
            model="sdxl",
            sampler="euler",
            style=["editorial"],
        )
        probe = PreviewProbe(
            probe_id="probe-1",
            summary="Increase color intensity.",
            target_axes=["style"],
            preserve_axes=["composition"],
            preview_execution_spec={"patch_family": "style_shift"},
            source_kind="schema_variation",
        )
        patch = CommittedPatch(
            patch_id="patch-1",
            target_fields=["style"],
            target_axes=["style"],
            preserve_axes=["composition"],
            changes={"style": ["editorial", "vivid"]},
            rationale="Commit the stronger style direction from preview.",
        )

        initial_payload, initial_summary = adapter.produce_initial_result(schema)
        preview_result = adapter.execute_preview_probe(schema, probe)
        committed_payload, committed_summary = adapter.execute_committed_patch(schema, patch)

        self.assertEqual(initial_payload.result_type, "mock_initial_result")
        self.assertIn("initial_generation", initial_summary.changed_axes)
        self.assertEqual(preview_result.probe_id, "probe-1")
        self.assertEqual(preview_result.payload.result_type, "mock_preview_result")
        self.assertEqual(committed_payload.result_type, "mock_committed_result")
        self.assertIn("style", committed_summary.changed_axes)
