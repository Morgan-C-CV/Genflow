import unittest

from app.agent.runtime_models import CommittedPatch, NormalizedSchema, PreviewProbe
from app.agent.workflow_execution_source_models import (
    WorkflowCommitSource,
    WorkflowExecutionSource,
    WorkflowPreviewSource,
)


class WorkflowExecutionSourceModelsTest(unittest.TestCase):
    def test_execution_source_defaults_are_safe(self):
        source = WorkflowExecutionSource()

        self.assertEqual(source.workflow_id, "")
        self.assertEqual(source.execution_kind, "")
        self.assertEqual(source.schema, NormalizedSchema())
        self.assertEqual(source.selected_reference_ids, [])
        self.assertEqual(source.selected_reference_bundle, {})

    def test_preview_source_wraps_probe(self):
        probe = PreviewProbe(probe_id="p_001", target_axes=["style"])

        source = WorkflowPreviewSource(execution_kind="preview", preview=True, selected_probe=probe)

        self.assertEqual(source.selected_probe.probe_id, "p_001")
        self.assertTrue(source.preview)

    def test_commit_source_wraps_patch(self):
        patch = CommittedPatch(patch_id="cp_001", target_fields=["style"])

        source = WorkflowCommitSource(execution_kind="commit", accepted_patch=patch)

        self.assertEqual(source.accepted_patch.patch_id, "cp_001")
        self.assertEqual(source.accepted_patch.target_fields, ["style"])


if __name__ == "__main__":
    unittest.main()
