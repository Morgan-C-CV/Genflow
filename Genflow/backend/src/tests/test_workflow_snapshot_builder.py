import unittest

from app.agent.benchmark_comparison_summary import BenchmarkComparisonSummary
from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, PreviewProbe
from app.agent.workflow_snapshot_builder import (
    SurrogateWorkflowSnapshot,
    build_surrogate_workflow_snapshot,
)


class WorkflowSnapshotBuilderTest(unittest.TestCase):
    def _make_session(self):
        session = AgentMemoryService().create_session("make a portrait")
        session.workflow_id = f"workflow-{session.session_id}"
        session.workflow_identity.workflow_kind = "normalized_schema_surrogate"
        session.workflow_identity.workflow_version = "phase-g-skeleton"
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
        session.current_result_id = "result-1"
        session.benchmark_comparison_summary = BenchmarkComparisonSummary(
            compared_anchor_ids=[101, 202],
            compared_candidate_ids=["benchmark-candidate-101", "benchmark-candidate-202"],
            focus_axes=["style"],
            preserve_axes=["composition"],
            confidence_hint=0.67,
            metadata={"benchmark_source": "refinement_search_bundle"},
        )
        return session

    def test_snapshot_builder_is_stable_for_initial_state(self):
        session = self._make_session()

        first = build_surrogate_workflow_snapshot(session, execution_kind="initial", preview=False)
        second = build_surrogate_workflow_snapshot(session, execution_kind="initial", preview=False)

        self.assertEqual(first, second)
        self.assertEqual(first.workflow_identity.workflow_id, session.workflow_id)
        self.assertEqual(first.workflow_metadata["backend_kind"], "mock")
        self.assertEqual(first.surrogate_payload["schema"]["model"], "sdxl-base")
        self.assertEqual(first.workflow_topology_entry_node_ids, ["reference.bundle", "intent.prompt"])
        self.assertEqual(first.workflow_topology_exit_node_ids, ["result.output"])

    def test_snapshot_builder_includes_repair_probe_patch_feedback_and_scopes(self):
        session = self._make_session()
        session.latest_feedback = "Keep composition, improve style."
        session.feedback_history = [session.latest_feedback]
        session.dissatisfaction_axes = ["style"]
        session.preserve_constraints = ["composition"]
        session.current_uncertainty_estimate = 0.25
        session.selected_probe = PreviewProbe(
            probe_id="p_001",
            target_axes=["style"],
            preserve_axes=["composition"],
        )
        session.accepted_patch = CommittedPatch(
            patch_id="cp_p_001",
            target_fields=["style", "model"],
        )

        snapshot = build_surrogate_workflow_snapshot(session, execution_kind="commit", preview=False)

        self.assertEqual(snapshot.workflow_metadata["selected_probe_id"], "p_001")
        self.assertEqual(snapshot.workflow_metadata["document_region_label"], "repair_region")
        self.assertEqual(snapshot.surrogate_payload["accepted_patch_id"], "cp_p_001")
        self.assertEqual(snapshot.surrogate_payload["latest_feedback"], "Keep composition, improve style.")
        self.assertEqual(snapshot.workflow_topology_hints["accepted_patch_id"], "cp_p_001")
        self.assertEqual(snapshot.workflow_topology_exit_node_ids, ["patch.cp_p_001", "result.output"])
        self.assertEqual(snapshot.editable_scopes[0].node_ids, ["style", "model"])
        self.assertEqual(snapshot.protected_scopes[0].node_ids, ["composition"])
        self.assertEqual(
            snapshot.workflow_metadata["benchmark_comparison"]["benchmark_source"],
            "refinement_search_bundle",
        )
        self.assertEqual(
            snapshot.surrogate_payload["benchmark_comparison"]["compared_candidate_ids"],
            ["benchmark-candidate-101", "benchmark-candidate-202"],
        )

    def test_snapshot_defaults_are_safe(self):
        snapshot = SurrogateWorkflowSnapshot()

        self.assertEqual(snapshot.workflow_identity.workflow_id, "")
        self.assertEqual(snapshot.workflow_metadata, {})
        self.assertEqual(snapshot.surrogate_payload, {})
        self.assertEqual(snapshot.workflow_topology_hints, {})
        self.assertEqual(snapshot.editable_scopes, [])
        self.assertEqual(snapshot.protected_scopes, [])


if __name__ == "__main__":
    unittest.main()
