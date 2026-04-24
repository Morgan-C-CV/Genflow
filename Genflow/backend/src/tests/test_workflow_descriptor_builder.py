import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, PreviewProbe
from app.agent.workflow_descriptor_builder import build_surrogate_workflow_descriptor
from app.agent.workflow_descriptor_models import (
    SurrogateExecutionDescriptor,
    SurrogateRepairDescriptor,
    SurrogateWorkflowDescriptor,
)


class WorkflowDescriptorBuilderTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.workflow_id = f"workflow-{session.session_id}"
        session.workflow_identity.workflow_kind = "surrogate_workflow"
        session.workflow_metadata = {
            "backend_kind": "mock",
            "workflow_profile": "default",
        }
        session.current_schema.prompt = "a cinematic portrait"
        session.current_schema.negative_prompt = "blurry"
        session.current_schema.model = "sdxl-base"
        session.current_schema.sampler = "DPM++ 2M"
        session.current_schema.style = ["cinematic"]
        session.current_schema.lora = ["portrait-helper"]
        session.current_schema_raw = "schema"
        session.selected_gallery_index = 7
        session.selected_reference_ids = [101, 202]
        session.selected_reference_bundle = {"references": [{"id": 101}, {"id": 202}]}
        session.current_result_id = "result-1"
        return session

    def test_descriptor_defaults_are_safe(self):
        descriptor = SurrogateWorkflowDescriptor()

        self.assertEqual(descriptor.workflow_id, "")
        self.assertEqual(descriptor.schema_style, [])
        self.assertEqual(descriptor.execution, SurrogateExecutionDescriptor())
        self.assertEqual(descriptor.repair, SurrogateRepairDescriptor())
        self.assertEqual(descriptor.metadata, {})

    def test_builder_produces_stable_initial_descriptor(self):
        session = self._make_session()

        descriptor = build_surrogate_workflow_descriptor(session, execution_kind="initial", preview=False)

        self.assertEqual(descriptor.workflow_id, session.workflow_id)
        self.assertEqual(descriptor.schema_prompt, "a cinematic portrait")
        self.assertEqual(descriptor.schema_model, "sdxl-base")
        self.assertEqual(descriptor.selected_gallery_index, 7)
        self.assertEqual(descriptor.selected_reference_ids, [101, 202])
        self.assertEqual(descriptor.execution.execution_kind, "initial")
        self.assertFalse(descriptor.execution.preview)
        self.assertEqual(descriptor.execution.backend_kind, "mock")
        self.assertEqual(descriptor.execution.current_result_id, "result-1")

    def test_builder_includes_repair_probe_patch_and_feedback_information(self):
        session = self._make_session()
        session.latest_feedback = "Keep composition, improve style."
        session.feedback_history = [session.latest_feedback]
        session.dissatisfaction_axes = ["style"]
        session.preserve_constraints = ["composition"]
        session.current_uncertainty_estimate = 0.35
        session.selected_probe = PreviewProbe(
            probe_id="p_001",
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        session.accepted_patch = CommittedPatch(
            patch_id="cp_p_001",
            target_fields=["style", "model"],
        )

        descriptor = build_surrogate_workflow_descriptor(session, execution_kind="commit", preview=False)

        self.assertTrue(descriptor.repair.has_feedback)
        self.assertEqual(descriptor.repair.feedback_count, 1)
        self.assertEqual(descriptor.repair.dissatisfaction_axes, ["style"])
        self.assertEqual(descriptor.repair.preserve_constraints, ["composition"])
        self.assertEqual(descriptor.repair.selected_probe_id, "p_001")
        self.assertEqual(descriptor.repair.probe_target_axes, ["style"])
        self.assertEqual(descriptor.repair.probe_preserve_axes, ["composition"])
        self.assertEqual(descriptor.repair.accepted_patch_id, "cp_p_001")
        self.assertEqual(descriptor.repair.patch_target_fields, ["style", "model"])
        self.assertEqual(descriptor.repair.current_uncertainty_estimate, 0.35)


if __name__ == "__main__":
    unittest.main()
