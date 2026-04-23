import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import ParsedFeedbackEvidence, PreviewProbe
from app.agent.result_executor import ResultExecutor
from app.agent.runtime_models import NormalizedSchema
from app.agent.runtime_service import AgentRuntimeService


VALID_METADATA_JSON = """
{
  "prompt": "a bright cinematic portrait",
  "negative_prompt": "blurry, low quality",
  "cfgscale": "7",
  "steps": "30",
  "sampler": "DPM++ 2M",
  "seed": "1234567890",
  "model": "sdxl-base",
  "clipskip": "2",
  "style": "cinematic, vivid",
  "lora": "portrait-helper, color-boost",
  "full_metadata_string": "prompt: a bright cinematic portrait"
}
"""


class FakeOrchestrationService:
    def __init__(self, memory_service: AgentMemoryService):
        self.memory_service = memory_service

    def start_session(self, user_intent: str):
        session = self.memory_service.create_session(user_intent)
        session.plan = type(
            "FakePlan",
            (),
            {
                "fixed_constraints": {"subject": "portrait"},
                "free_variables": ["style"],
                "locked_axes": ["subject"],
                "unclear_axes": ["style"],
                "next_action": "retrieve_resources",
                "clarification_questions": [],
                "reasoning_summary": "fake plan",
            },
        )()
        return self.memory_service.save_session(session)

    def submit_clarification(self, session_id: str, answers: list[str]):
        session = self.memory_service.get_session(session_id)
        session.clarified_intent = session.clarified_intent + " | " + " | ".join(answers)
        return self.memory_service.save_session(session)

    def generate_candidates(self, session_id: str, refresh: bool = False, per_query_k: int = 2, top_k: int = 12):
        session = self.memory_service.get_session(session_id)
        session.latest_wall = type(
            "FakeWall",
            (),
            {
                "groups": [[7, 8]],
                "flat_indices": [7, 8],
                "query_labels": ["portrait direction"],
            },
        )()
        return self.memory_service.save_session(session)


class FakeSearchService:
    def build_diverse_reference_bundle(self, index: int):
        return {
            "query_index": index,
            "counts": {
                "best": 1,
                "complementary_knn": 2,
                "exploratory": 2,
                "counterexample": 1,
            },
            "references": [
                {"id": 101, "index": index, "role": "best"},
                {"id": 102, "index": index + 1, "role": "complementary_knn"},
                {"id": 103, "index": index + 2, "role": "exploratory"},
            ],
        }

    def generate_image_metadata(
        self,
        reference_bundle: dict,
        user_intent: str,
        previous_output: str = "",
        validation_error: str = "",
    ):
        return VALID_METADATA_JSON


class FakeFeedbackParser:
    def parse(
        self,
        feedback_text: str,
        current_result_summary: str = "",
        current_schema_prompt: str = "",
    ):
        return ParsedFeedbackEvidence(
            dissatisfaction_scope=["style"],
            preserve_constraints=["Keep the composition"],
            requested_changes=["make the style brighter"],
            uncertainty_estimate=0.25,
            raw_feedback=feedback_text,
            parser_notes=["fake_parser"],
        )


class FakeHypothesisBuilder:
    def build(self, current_schema, current_result_summary, feedback_evidence, history=None):
        from app.agent.runtime_models import RepairHypothesis

        return [
            RepairHypothesis(
                hypothesis_id="h_001",
                summary="style mismatch",
                likely_changed_axes=["style"],
                likely_preserved_axes=["composition"],
                likely_patch_family="resource_shift",
                rank=1,
            ),
            RepairHypothesis(
                hypothesis_id="h_002",
                summary="color mismatch",
                likely_changed_axes=["color_palette"],
                likely_preserved_axes=["composition"],
                likely_patch_family="prompt_color_adjustment",
                rank=2,
            ),
        ]


class FakeProbeGenerator:
    def generate(
        self,
        current_schema,
        parsed_feedback,
        repair_hypotheses,
        selected_gallery_index=None,
        selected_reference_ids=None,
    ):
        return [
            PreviewProbe(
                probe_id="p_001",
                summary="preview style shift",
                target_axes=["style"],
                preserve_axes=["composition"],
                preview_execution_spec={"patch_family": "resource_shift", "reference_anchor": selected_gallery_index},
                source_kind="resource_shift",
            ),
            PreviewProbe(
                probe_id="p_002",
                summary="preview color shift",
                target_axes=["color_palette"],
                preserve_axes=["composition"],
                preview_execution_spec={"patch_family": "prompt_color_adjustment", "reference_anchor": selected_gallery_index},
                source_kind="schema_variation",
            ),
        ]


class RuntimeServiceTest(unittest.TestCase):
    def test_runtime_service_initial_commit_path_persists_required_state(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "result-rt-1")
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=orchestration,
            search_service=search,
            result_executor=executor,
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)

        self.assertEqual(session.selected_gallery_index, 7)
        self.assertEqual(session.selected_reference_ids, [101, 102, 103])
        self.assertTrue(session.selected_reference_bundle)
        self.assertIn("Selected gallery anchor bundle", session.current_gallery_anchor_summary)
        self.assertEqual(session.current_schema_raw.strip(), VALID_METADATA_JSON.strip())
        self.assertIsInstance(session.current_schema, NormalizedSchema)
        self.assertEqual(session.current_schema.model, "sdxl-base")
        self.assertEqual(session.current_result_payload.result_id, "result-rt-1")
        self.assertEqual(session.current_result_payload.result_type, "mock_initial_result")
        self.assertIn("references=3", session.current_result_summary.summary_text)

    def test_runtime_service_feedback_and_hypotheses_persist_state(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "result-rt-2")
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=orchestration,
            search_service=search,
            result_executor=executor,
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)

        self.assertEqual(session.feedback_history, ["Keep the composition, but improve style."])
        self.assertEqual(session.latest_feedback, "Keep the composition, but improve style.")
        self.assertEqual(session.preserve_constraints, ["Keep the composition"])
        self.assertEqual(session.dissatisfaction_axes, ["style"])
        self.assertEqual(session.requested_changes, ["make the style brighter"])
        self.assertEqual(session.current_uncertainty_estimate, 0.25)
        self.assertEqual(session.parsed_feedback.raw_feedback, "Keep the composition, but improve style.")
        self.assertEqual(len(session.repair_hypotheses), 2)
        self.assertEqual(session.repair_hypotheses[0].hypothesis_id, "h_001")

    def test_runtime_service_preview_probe_flow_updates_preview_state_only(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "preview-rt-1")
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=orchestration,
            search_service=search,
            result_executor=executor,
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            probe_generator=FakeProbeGenerator(),
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        original_schema_prompt = session.current_schema.prompt
        original_result_id = session.current_result_payload.result_id
        original_summary_text = session.current_result_summary.summary_text

        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        self.assertEqual(len(session.preview_probe_candidates), 2)
        self.assertEqual(session.preview_probe_candidates[0].probe_id, "p_001")

        session = service.preview_probe(session.session_id, "p_001")
        self.assertEqual(len(session.preview_results), 1)
        self.assertEqual(len(session.preview_probe_results), 1)
        self.assertEqual(session.preview_probe_results[0].probe_id, "p_001")

        session = service.select_probe(session.session_id, "p_001")
        self.assertEqual(session.selected_probe.probe_id, "p_001")
        self.assertEqual(session.current_schema.prompt, original_schema_prompt)
        self.assertEqual(session.current_result_payload.result_id, original_result_id)
        self.assertEqual(session.current_result_summary.summary_text, original_summary_text)


if __name__ == "__main__":
    unittest.main()
