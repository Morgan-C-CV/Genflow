import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, ParsedFeedbackEvidence, PreviewProbe, VerifierResult
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
    def __init__(self):
        self.last_refinement_benchmark_set = None

    def generate(
        self,
        current_schema,
        parsed_feedback,
        repair_hypotheses,
        selected_gallery_index=None,
        selected_reference_ids=None,
        refinement_benchmark_set=None,
    ):
        self.last_refinement_benchmark_set = refinement_benchmark_set
        return [
            PreviewProbe(
                probe_id="p_001",
                summary="preview color shift",
                target_axes=["color_palette"],
                preserve_axes=["composition"],
                preview_execution_spec={"patch_family": "resource_shift", "reference_anchor": selected_gallery_index},
                source_kind="resource_shift",
            ),
            PreviewProbe(
                probe_id="p_002",
                summary="preview style shift",
                target_axes=["style"],
                preserve_axes=["composition"],
                preview_execution_spec={"patch_family": "prompt_color_adjustment", "reference_anchor": selected_gallery_index},
                source_kind="schema_variation",
            ),
        ]


class FakePatchPlanner:
    def plan(self, selected_probe, current_schema, parsed_feedback, repair_hypotheses):
        return CommittedPatch(
            patch_id="cp_p_001",
            target_fields=["style", "model"],
            target_axes=["style"],
            preserve_axes=["composition"],
            changes={
                "style": ["cinematic", "vivid"],
                "model": "sdxl-base-patched",
            },
            rationale="apply style-focused committed patch",
        )


class FakeVerifier:
    def __init__(self):
        self.last_benchmark_comparison_summary = None

    def verify(
        self,
        previous_result_summary,
        updated_result_summary,
        selected_probe,
        committed_patch,
        preserve_constraints,
        benchmark_comparison_summary=None,
    ):
        self.last_benchmark_comparison_summary = benchmark_comparison_summary
        return VerifierResult(
            improved=True,
            continue_recommended=False,
            confidence=0.88,
            regression_notes=[],
            summary="verifier accepts current direction",
        )


class FakePboProbeRanker:
    def __init__(self):
        self.last_benchmark_comparison_summary = None
        self.last_refinement_benchmark_set = None

    def __call__(
        self,
        probes,
        parsed_feedback,
        benchmark_comparison_summary=None,
        refinement_benchmark_set=None,
    ):
        self.last_benchmark_comparison_summary = benchmark_comparison_summary
        self.last_refinement_benchmark_set = refinement_benchmark_set
        return list(reversed(probes))


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
            execution_adapter=executor,
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
            execution_adapter=executor,
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
        self.assertEqual(session.refinement_benchmark_set.benchmark_kind, "refinement_local_comparison")
        self.assertEqual(session.refinement_benchmark_set.anchor_ids, [101, 102, 103])
        self.assertTrue(session.refinement_benchmark_summary)
        self.assertIn("focus_axes=style", session.refinement_benchmark_summary)
        self.assertEqual(
            session.benchmark_comparison_summary.compared_candidate_ids,
            ["benchmark-candidate-101", "benchmark-candidate-102", "benchmark-candidate-103"],
        )
        self.assertIn(
            "benchmark_source=refinement_search_bundle",
            session.benchmark_comparison_summary.summary_bullets,
        )

    def test_runtime_service_passes_refinement_benchmark_context_to_probe_generator(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "result-rt-3")
        probe_generator = FakeProbeGenerator()
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=orchestration,
            search_service=search,
            execution_adapter=executor,
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            probe_generator=probe_generator,
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)

        self.assertEqual(
            probe_generator.last_refinement_benchmark_set.benchmark_id,
            session.refinement_benchmark_set.benchmark_id,
        )
        self.assertEqual(
            probe_generator.last_refinement_benchmark_set.anchor_ids,
            [101, 102, 103],
        )

    def test_runtime_service_reranks_generated_probes_with_pbo_ranker(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "result-rt-4")
        probe_generator = FakeProbeGenerator()
        pbo_ranker = FakePboProbeRanker()
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=orchestration,
            search_service=search,
            execution_adapter=executor,
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            probe_generator=probe_generator,
            pbo_probe_ranker=pbo_ranker,
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)

        self.assertEqual(session.preview_probe_candidates[0].probe_id, "p_002")
        self.assertEqual(
            pbo_ranker.last_benchmark_comparison_summary.compared_candidate_ids,
            session.benchmark_comparison_summary.compared_candidate_ids,
        )
        self.assertEqual(
            pbo_ranker.last_refinement_benchmark_set.benchmark_id,
            session.refinement_benchmark_set.benchmark_id,
        )

    def test_runtime_service_preview_probe_flow_updates_preview_state_only(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "preview-rt-1")
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=orchestration,
            search_service=search,
            execution_adapter=executor,
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
        self.assertEqual(session.preview_probe_candidates[0].probe_id, "p_002")
        self.assertIn("pbo_score", session.preview_probe_candidates[0].preview_execution_spec)
        self.assertIn("pbo_rationale", session.preview_probe_candidates[0].preview_execution_spec)

        session = service.preview_probe(session.session_id, "p_002")
        self.assertEqual(len(session.preview_results), 1)
        self.assertEqual(len(session.preview_probe_results), 1)
        self.assertEqual(session.preview_probe_results[0].probe_id, "p_002")

        session = service.select_probe(session.session_id, "p_002")
        self.assertEqual(session.selected_probe.probe_id, "p_002")
        self.assertEqual(session.current_schema.prompt, original_schema_prompt)
        self.assertEqual(session.current_result_payload.result_id, original_result_id)
        self.assertEqual(session.current_result_summary.summary_text, original_summary_text)

    def test_runtime_service_commit_execute_verify_and_should_continue(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        ids = iter(["initial-rt-1", "preview-rt-1", "commit-rt-1"])
        executor = ResultExecutor(id_factory=lambda: next(ids))
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=orchestration,
            search_service=search,
            execution_adapter=executor,
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            probe_generator=FakeProbeGenerator(),
            patch_planner=FakePatchPlanner(),
            verifier=FakeVerifier(),
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        old_result_id = session.current_result_payload.result_id
        old_summary = session.current_result_summary.summary_text

        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.preview_probe(session.session_id, "p_001")
        session = service.select_probe(session.session_id, "p_001")
        session = service.commit_patch(session.session_id)

        self.assertEqual(session.accepted_patch.patch_id, "cp_p_001")
        self.assertEqual(len(session.patch_history), 1)
        self.assertEqual(session.current_schema.model, "sdxl-base-patched")
        self.assertEqual(session.current_schema.style, ["cinematic", "vivid"])
        self.assertNotEqual(session.current_schema_raw.strip(), "")
        self.assertIn('"model": "sdxl-base-patched"', session.current_schema_raw)
        self.assertIn('"style": "cinematic, vivid"', session.current_schema_raw)

        session = service.execute_patch(session.session_id)
        self.assertEqual(session.previous_result_summary.summary_text, old_summary)
        self.assertNotEqual(session.current_result_payload.result_id, old_result_id)
        self.assertEqual(session.current_result_payload.result_type, "mock_committed_result")

        session = service.verify_latest_result(session.session_id)
        self.assertTrue(session.latest_verifier_result.summary)
        self.assertFalse(session.continue_recommended)
        self.assertEqual(session.verifier_confidence, 0.88)
        self.assertEqual(session.stop_reason, "verifier_accepts_current_direction")
        self.assertFalse(service.should_continue(session.session_id))

    def test_runtime_service_passes_benchmark_comparison_summary_to_verifier(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        ids = iter(["initial-rt-1", "commit-rt-1"])
        executor = ResultExecutor(id_factory=lambda: next(ids))
        fake_verifier = FakeVerifier()
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=orchestration,
            search_service=search,
            execution_adapter=executor,
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            probe_generator=FakeProbeGenerator(),
            patch_planner=FakePatchPlanner(),
            verifier=fake_verifier,
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.select_probe(session.session_id, "p_001")
        session = service.commit_patch(session.session_id)
        session = service.execute_patch(session.session_id)
        session = service.verify_latest_result(session.session_id)

        self.assertEqual(
            fake_verifier.last_benchmark_comparison_summary.compared_candidate_ids,
            session.benchmark_comparison_summary.compared_candidate_ids,
        )


if __name__ == "__main__":
    unittest.main()
