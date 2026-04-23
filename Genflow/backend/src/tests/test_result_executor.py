import unittest

from app.agent.result_executor import ResultExecutor
from app.agent.runtime_models import NormalizedSchema, PreviewProbe, ResultPayload, ResultSummary


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


if __name__ == "__main__":
    unittest.main()
