import unittest

from app.agent.result_executor import ResultExecutor
from app.agent.runtime_models import NormalizedSchema, ResultPayload, ResultSummary


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


if __name__ == "__main__":
    unittest.main()
