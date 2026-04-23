import unittest

from app.agent.runtime_models import (
    CommittedPatch,
    NormalizedSchema,
    PreviewResult,
    ResultPayload,
    ResultSummary,
    VerifierResult,
)


class RuntimeModelsTest(unittest.TestCase):
    def test_runtime_models_construct_with_defaults(self):
        schema = NormalizedSchema()
        payload = ResultPayload()
        summary = ResultSummary()
        preview = PreviewResult()
        patch = CommittedPatch()
        verifier = VerifierResult()

        self.assertEqual(schema.prompt, "")
        self.assertEqual(payload.content, {})
        self.assertEqual(summary.notes, [])
        self.assertEqual(preview.comparison_notes, [])
        self.assertEqual(patch.target_fields, [])
        self.assertFalse(verifier.improved)

    def test_default_factories_are_not_shared(self):
        schema_a = NormalizedSchema()
        schema_b = NormalizedSchema()
        schema_a.style.append("cinematic")

        preview_a = PreviewResult()
        preview_b = PreviewResult()
        preview_a.comparison_notes.append("note-a")

        self.assertEqual(schema_b.style, [])
        self.assertEqual(preview_b.comparison_notes, [])


if __name__ == "__main__":
    unittest.main()
