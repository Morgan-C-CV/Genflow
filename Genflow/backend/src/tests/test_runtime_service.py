import unittest

from app.agent.memory import AgentMemoryService
from app.agent.runtime_models import CommittedPatch, ParsedFeedbackEvidence, PreviewProbe, VerifierResult, VerifierSignalSummary
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
    def generate(self, selected_probe, current_schema, parsed_feedback, repair_hypotheses):
        return [
            CommittedPatch(
                patch_id="cp_p_001",
                target_fields=["prompt"],
                target_axes=["composition"],
                preserve_axes=["composition"],
                changes={
                    "prompt": f"{current_schema.prompt} | maintain composition",
                },
                rationale="conservative prompt-only patch",
                metadata={"candidate_kind": "conservative"},
            ),
            CommittedPatch(
                patch_id="cp_p_002",
                target_fields=["style", "model"],
                target_axes=["style"],
                preserve_axes=["composition"],
                changes={
                    "style": ["cinematic", "vivid"],
                    "model": "sdxl-base-patched",
                },
                rationale="apply style-focused committed patch",
                metadata={"candidate_kind": "style_shift"},
            ),
        ]


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
            signal_summary=VerifierSignalSummary(
                target_alignment_score=2.0,
                preserve_risk_score=0.2,
                benchmark_support_score=1.1,
                execution_evidence_score=2.0,
                total_score=4.9,
                notes=["strong_benchmark_support"],
                regression_notes=[],
            ),
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


class FakePboPatchRanker:
    def __init__(self):
        self.last_benchmark_comparison_summary = None
        self.last_refinement_benchmark_set = None

    def __call__(
        self,
        patch_candidates,
        parsed_feedback,
        benchmark_comparison_summary=None,
        refinement_benchmark_set=None,
    ):
        self.last_benchmark_comparison_summary = benchmark_comparison_summary
        self.last_refinement_benchmark_set = refinement_benchmark_set
        return list(reversed(patch_candidates))


class FakePboBenchmarkRanker:
    def __init__(self):
        self.last_session_context = None

    def __call__(self, candidates, session_context):
        self.last_session_context = session_context
        return list(reversed(candidates))


class CapturingExecutionAdapter(ResultExecutor):
    def __init__(self, id_factory=None):
        super().__init__(id_factory=id_factory)
        self.last_commit_graph_patch = None
        self.last_commit_execution_mode = ""
        self.last_commit_execution_authority = ""
        self.last_commit_execution_implementation_mode = ""

    def execute_committed_patch(
        self,
        schema,
        patch,
        graph_patch=None,
        commit_execution_mode="",
        commit_execution_authority="",
        commit_execution_implementation_mode="",
    ):
        self.last_commit_graph_patch = graph_patch
        self.last_commit_execution_mode = commit_execution_mode
        self.last_commit_execution_authority = commit_execution_authority
        self.last_commit_execution_implementation_mode = commit_execution_implementation_mode
        return super().execute_committed_patch(
            schema,
            patch,
            graph_patch=graph_patch,
            commit_execution_mode=commit_execution_mode,
            commit_execution_authority=commit_execution_authority,
            commit_execution_implementation_mode=commit_execution_implementation_mode,
        )


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
            ["benchmark-candidate-103", "benchmark-candidate-102", "benchmark-candidate-101"],
        )
        self.assertIn(
            "benchmark_source=refinement_search_bundle",
            session.benchmark_comparison_summary.summary_bullets,
        )

    def test_runtime_service_uses_reranked_benchmark_set_after_repair_hypotheses(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "result-rt-2b")
        pbo_benchmark_ranker = FakePboBenchmarkRanker()
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=orchestration,
            search_service=search,
            execution_adapter=executor,
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            pbo_benchmark_ranker=pbo_benchmark_ranker,
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)

        self.assertEqual(
            [candidate.candidate_id for candidate in session.refinement_benchmark_set.comparison_candidates],
            ["benchmark-candidate-103", "benchmark-candidate-102", "benchmark-candidate-101"],
        )
        self.assertEqual(pbo_benchmark_ranker.last_session_context.session_id, session.session_id)

    def test_runtime_service_exposes_policy_decision_without_rewriting_execution_flow(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "result-rt-2c")
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

        decision = service.get_policy_decision(session.session_id)

        self.assertEqual(decision.next_action, "generate_probes")

    def test_runtime_service_policy_distinguishes_execute_then_verify_after_commit(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        ids = iter(["initial-rt-policy", "commit-rt-policy", "commit-rt-policy-2"])
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
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.preview_selected_probe(session.session_id)
        session = service.commit_patch(session.session_id)

        decision = service.get_policy_decision(session.session_id)
        self.assertEqual(decision.next_action, "execute_patch")
        self.assertEqual(session.current_workflow_graph_patch.patch_id, session.accepted_patch.patch_id)
        self.assertTrue(session.current_workflow_graph_patch.node_patches)
        self.assertEqual(
            session.workflow_metadata["workflow_graph_patch"]["patch_id"],
            session.accepted_patch.patch_id,
        )

        session = service.execute_patch(session.session_id)

        decision = service.get_policy_decision(session.session_id)
        self.assertEqual(decision.next_action, "verify_latest_result")

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
        self.assertGreater(len(session.workflow_graph_patch_candidates), 1)
        self.assertEqual(
            session.workflow_graph_patch_candidates[0].metadata["probe_id"],
            session.selected_probe.probe_id,
        )
        self.assertIn("pbo_score", session.workflow_graph_patch_candidates[0].metadata)
        self.assertIn("pbo_rationale", session.workflow_graph_patch_candidates[0].metadata)

    def test_runtime_service_auto_selects_top_ranked_probe_without_manual_override(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "result-rt-3a")
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
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)

        self.assertEqual(session.preview_probe_candidates[0].probe_id, "p_002")
        self.assertEqual(session.selected_probe.probe_id, "p_002")
        self.assertGreater(len(session.workflow_graph_patch_candidates), 1)
        self.assertIn("pbo_score", session.workflow_graph_patch_candidates[0].metadata)

    def test_runtime_service_manual_probe_selection_overrides_default_probe(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "result-rt-3b")
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
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        self.assertEqual(session.selected_probe.probe_id, "p_002")

        session = service.select_probe(session.session_id, "p_001")

        self.assertEqual(session.selected_probe.probe_id, "p_001")
        self.assertEqual(
            session.workflow_graph_patch_candidates[0].metadata["probe_id"],
            "p_001",
        )
        self.assertIn("pbo_score", session.workflow_graph_patch_candidates[0].metadata)

    def test_runtime_service_preview_selected_probe_uses_default_selected_probe(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "preview-rt-default")
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
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)

        self.assertEqual(session.selected_probe.probe_id, "p_002")

        session = service.preview_selected_probe(session.session_id)

        self.assertEqual(session.preview_probe_results[-1].probe_id, "p_002")

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

    def test_runtime_service_commit_patch_consumes_top_ranked_patch_candidate(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "result-rt-4b")
        pbo_patch_ranker = FakePboPatchRanker()
        service = AgentRuntimeService(
            memory_service=memory,
            orchestration_service=orchestration,
            search_service=search,
            execution_adapter=executor,
            feedback_parser=FakeFeedbackParser(),
            hypothesis_builder=FakeHypothesisBuilder(),
            probe_generator=FakeProbeGenerator(),
            patch_planner=FakePatchPlanner(),
            pbo_patch_ranker=pbo_patch_ranker,
        )

        session = service.start_episode("make a portrait")
        session = service.generate_initial_candidates(session.session_id)
        session = service.select_initial_reference(session.session_id, 7)
        session = service.generate_initial_schema(session.session_id)
        session = service.produce_initial_result(session.session_id)
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.select_probe(session.session_id, "p_002")
        session = service.commit_patch(session.session_id)

        self.assertEqual(session.accepted_patch.patch_id, "cp_p_002")
        self.assertEqual(
            pbo_patch_ranker.last_benchmark_comparison_summary.compared_candidate_ids,
            session.benchmark_comparison_summary.compared_candidate_ids,
        )
        self.assertEqual(
            pbo_patch_ranker.last_refinement_benchmark_set.benchmark_id,
            session.refinement_benchmark_set.benchmark_id,
        )
        self.assertEqual(session.top_schema_patch_candidate.patch_id, "cp_p_002")
        self.assertEqual(
            session.top_workflow_graph_patch_candidate.candidate_id,
            session.workflow_graph_patch_candidates[0].candidate_id,
        )
        self.assertEqual(session.preferred_commit_source, "graph")
        self.assertEqual(
            session.selected_graph_native_patch_candidate.candidate_id,
            session.top_workflow_graph_patch_candidate.candidate_id,
        )
        self.assertTrue(session.selected_workflow_graph_patch.patch_id)
        self.assertEqual(
            session.selected_workflow_graph_patch.metadata["candidate_id"],
            session.selected_graph_native_patch_candidate.candidate_id,
        )
        self.assertEqual(session.current_workflow_graph_patch.patch_id, "cp_p_002")
        self.assertIn("render.model", [patch.node_id for patch in session.current_workflow_graph_patch.node_patches])

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

        session = service.preview_selected_probe(session.session_id)
        self.assertEqual(len(session.preview_results), 1)
        self.assertEqual(len(session.preview_probe_results), 1)
        self.assertEqual(session.preview_probe_results[0].probe_id, "p_002")

        self.assertEqual(session.current_schema.prompt, original_schema_prompt)
        self.assertEqual(session.current_result_payload.result_id, original_result_id)
        self.assertEqual(session.current_result_summary.summary_text, original_summary_text)

    def test_runtime_service_commit_execute_verify_and_should_continue(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        ids = iter(["initial-rt-1", "preview-rt-1", "commit-rt-1"])
        executor = CapturingExecutionAdapter(id_factory=lambda: next(ids))
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
        session = service.preview_selected_probe(session.session_id)
        session = service.commit_patch(session.session_id)

        self.assertEqual(session.accepted_patch.patch_id, "cp_p_002")
        self.assertEqual(len(session.patch_history), 1)
        self.assertEqual(session.top_schema_patch_candidate.patch_id, "cp_p_002")
        self.assertTrue(session.top_workflow_graph_patch_candidate.candidate_id)
        self.assertEqual(session.preferred_commit_source, "graph")
        self.assertEqual(
            session.selected_graph_native_patch_candidate.candidate_id,
            session.top_workflow_graph_patch_candidate.candidate_id,
        )
        self.assertTrue(session.selected_workflow_graph_patch.patch_id)
        self.assertIn("pbo_score", session.accepted_patch.metadata)
        self.assertIn("pbo_rationale", session.accepted_patch.metadata)
        self.assertEqual(session.current_workflow_graph_patch.patch_id, "cp_p_002")
        self.assertTrue(session.current_workflow_graph_patch.edge_patches)
        self.assertTrue(session.current_workflow_graph_patch.region_patches)
        self.assertIn("graph_native_aligned_winner", session.accepted_patch.metadata)
        self.assertEqual(session.accepted_patch.patch_id, session.top_schema_patch_candidate.patch_id)
        self.assertEqual(session.commit_execution_mode, "graph_native_execution_handoff")
        self.assertEqual(session.commit_execution_authority, "graph_authoritative")
        self.assertEqual(session.commit_execution_implementation_mode, "graph_primary_execution")
        self.assertEqual(payload := session.workflow_metadata["patch_winner_comparison"]["commit_execution_implementation_mode"], "graph_primary_execution")
        self.assertEqual(
            session.workflow_metadata["patch_winner_comparison"]["request_primary_plan_kind"],
            "graph_primary",
        )
        self.assertEqual(
            session.accepted_patch.metadata["commit_execution_mode"],
            "graph_native_execution_handoff",
        )
        self.assertEqual(
            session.accepted_patch.metadata["commit_execution_authority"],
            "graph_authoritative",
        )
        self.assertEqual(session.current_schema.model, "sdxl-base-patched")
        self.assertEqual(session.current_schema.style, ["cinematic", "vivid"])
        self.assertNotEqual(session.current_schema_raw.strip(), "")
        self.assertIn('"model": "sdxl-base-patched"', session.current_schema_raw)
        self.assertIn('"style": "cinematic, vivid"', session.current_schema_raw)

        session = service.execute_patch(session.session_id)
        self.assertEqual(session.previous_result_summary.summary_text, old_summary)
        self.assertNotEqual(session.current_result_payload.result_id, old_result_id)
        self.assertEqual(session.current_result_payload.result_type, "mock_committed_result")
        self.assertEqual(executor.last_commit_execution_mode, "graph_native_execution_handoff")
        self.assertEqual(executor.last_commit_execution_authority, "graph_authoritative")
        self.assertEqual(executor.last_commit_execution_implementation_mode, "graph_primary_execution")
        self.assertEqual(executor.last_commit_graph_patch.patch_id, session.selected_workflow_graph_patch.patch_id)
        self.assertEqual(
            session.latest_execution_source_evidence.commit_execution_mode,
            "graph_native_execution_handoff",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.commit_execution_authority,
            "graph_authoritative",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.request_primary_plan_kind,
            "graph_primary",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.commit_execution_implementation_mode,
            "graph_primary_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.request_backend_execution_mode,
            "graph_primary_backend_execution",
        )
        self.assertTrue(
            session.latest_execution_source_evidence.backend_graph_primary_capable,
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_accepted_execution_mode,
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_realized_execution_mode,
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.execution_behavior_branch,
            "graph_primary_execution_branch",
        )
        self.assertEqual(session.latest_execution_source_evidence.preferred_commit_source, "graph")
        self.assertTrue(session.latest_execution_source_evidence.request_graph_native_artifact_input_received)
        self.assertTrue(session.latest_execution_source_evidence.backend_echoed_graph_native_artifact_input_received)
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_commit_execution_authority,
            "graph_authoritative",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.selected_workflow_graph_patch_id,
            session.selected_workflow_graph_patch.patch_id,
        )
        self.assertEqual(
            session.latest_execution_source_evidence.top_schema_patch_id,
            session.top_schema_patch_candidate.patch_id,
        )
        self.assertEqual(
            session.latest_execution_source_evidence.top_graph_patch_candidate_id,
            session.top_workflow_graph_patch_candidate.candidate_id,
        )
        self.assertEqual(
            session.workflow_metadata["execution_source_evidence"]["preferred_commit_source"],
            "graph",
        )
        self.assertEqual(
            session.workflow_metadata["execution_source_evidence"]["commit_execution_mode"],
            "graph_native_execution_handoff",
        )
        self.assertEqual(
            session.workflow_metadata["execution_source_evidence"]["commit_execution_authority"],
            "graph_authoritative",
        )
        self.assertEqual(
            session.workflow_metadata["execution_source_evidence"]["backend_echoed_primary_plan_kind"],
            "graph_primary",
        )
        self.assertEqual(
            session.workflow_metadata["execution_source_evidence"]["backend_echoed_commit_execution_implementation_mode"],
            "graph_primary_execution",
        )
        self.assertTrue(
            session.workflow_metadata["execution_source_evidence"]["backend_echoed_graph_primary_capable"],
        )
        self.assertEqual(
            session.workflow_metadata["execution_source_evidence"]["backend_echoed_backend_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            session.workflow_metadata["execution_source_evidence"]["backend_echoed_accepted_backend_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            session.workflow_metadata["execution_source_evidence"]["backend_echoed_realized_backend_execution_mode"],
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            session.workflow_metadata["execution_source_evidence"]["backend_echoed_execution_behavior_branch"],
            "graph_primary_execution_branch",
        )
        self.assertIn("graph-primary execution branch", session.current_result_summary.summary_text)
        self.assertEqual(
            session.current_result_payload.content["execution_behavior_branch"],
            "graph_primary_execution_branch",
        )

        session = service.verify_latest_result(session.session_id)
        self.assertTrue(session.latest_verifier_result.summary)
        self.assertEqual(
            session.latest_verifier_signal_summary.total_score,
            session.latest_verifier_result.signal_summary.total_score,
        )
        self.assertEqual(
            session.latest_verifier_repair_recommendation.recommended_action,
            "stop",
        )
        self.assertFalse(session.continue_recommended)
        self.assertEqual(session.verifier_confidence, 0.88)
        self.assertEqual(session.stop_reason, "verifier_accepts_current_direction")
        self.assertFalse(service.should_continue(session.session_id))
        self.assertEqual(
            session.workflow_metadata["verifier_signal_summary"]["total_score"],
            session.latest_verifier_signal_summary.total_score,
        )
        self.assertEqual(
            session.workflow_metadata["verifier_repair_recommendation"]["recommended_action"],
            session.latest_verifier_repair_recommendation.recommended_action,
        )

    def test_runtime_service_keeps_graph_supplemental_when_authority_gate_fails(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        ids = iter(["initial-rt-gate-1", "preview-rt-gate-1", "commit-rt-gate-1"])
        executor = CapturingExecutionAdapter(id_factory=lambda: next(ids))
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
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.preview_selected_probe(session.session_id)
        session = service.commit_patch(session.session_id)

        session.selected_workflow_graph_patch.edge_patches = []
        session.selected_workflow_graph_patch.region_patches = []
        session.commit_execution_authority = service._determine_commit_execution_authority(session)
        session.commit_execution_implementation_mode = service._determine_commit_execution_implementation_mode(session)
        session.accepted_patch.metadata["commit_execution_authority"] = session.commit_execution_authority
        session.accepted_patch.metadata["commit_execution_implementation_mode"] = (
            session.commit_execution_implementation_mode
        )
        memory.save_session(session)

        self.assertEqual(session.commit_execution_mode, "graph_native_execution_handoff")
        self.assertEqual(session.commit_execution_authority, "graph_supplemental")
        self.assertEqual(session.commit_execution_implementation_mode, "schema_compatible_execution")

        session = service.execute_patch(session.session_id)

        self.assertEqual(executor.last_commit_execution_mode, "graph_native_execution_handoff")
        self.assertEqual(executor.last_commit_execution_authority, "graph_supplemental")
        self.assertEqual(
            session.latest_execution_source_evidence.commit_execution_authority,
            "graph_supplemental",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.commit_execution_implementation_mode,
            "schema_compatible_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.request_backend_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertTrue(
            session.latest_execution_source_evidence.backend_graph_primary_capable,
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_accepted_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_realized_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertEqual(session.latest_execution_source_evidence.request_primary_plan_kind, "schema_primary")
        self.assertEqual(
            session.latest_execution_source_evidence.execution_behavior_branch,
            "schema_primary_execution_branch",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_commit_execution_authority,
            "graph_supplemental",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_primary_plan_kind,
            "schema_primary",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_commit_execution_implementation_mode,
            "schema_compatible_execution",
        )
        self.assertTrue(
            session.latest_execution_source_evidence.backend_echoed_graph_primary_capable,
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_backend_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_accepted_backend_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_realized_backend_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_execution_behavior_branch,
            "schema_primary_execution_branch",
        )

    def test_runtime_service_persists_backend_mode_downgrade_trace(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        ids = iter(["initial-rt-dg-1", "preview-rt-dg-1", "commit-rt-dg-1"])
        executor = CapturingExecutionAdapter(id_factory=lambda: next(ids))
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
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.preview_selected_probe(session.session_id)
        session = service.commit_patch(session.session_id)

        session.selected_workflow_graph_patch.edge_patches = []
        session.selected_workflow_graph_patch.region_patches = []
        session.commit_execution_mode = "graph_native_execution_handoff"
        session.commit_execution_authority = "graph_authoritative"
        session.commit_execution_implementation_mode = "graph_primary_execution"
        session.accepted_patch.metadata["commit_execution_mode"] = session.commit_execution_mode
        session.accepted_patch.metadata["commit_execution_authority"] = session.commit_execution_authority
        session.accepted_patch.metadata["commit_execution_implementation_mode"] = (
            session.commit_execution_implementation_mode
        )
        memory.save_session(session)

        session = service.execute_patch(session.session_id)

        self.assertEqual(executor.last_commit_execution_mode, "graph_native_execution_handoff")
        self.assertEqual(executor.last_commit_execution_authority, "graph_authoritative")
        self.assertEqual(executor.last_commit_execution_implementation_mode, "graph_primary_execution")
        self.assertEqual(
            session.latest_execution_source_evidence.request_backend_execution_mode,
            "graph_primary_backend_execution",
        )
        self.assertTrue(
            session.latest_execution_source_evidence.backend_graph_primary_capable,
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_accepted_execution_mode,
            "graph_primary_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_realized_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_accepted_backend_execution_mode,
            "graph_primary_backend_execution",
        )
        self.assertTrue(
            session.latest_execution_source_evidence.backend_echoed_graph_primary_capable,
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_realized_backend_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.execution_behavior_branch,
            "schema_primary_execution_branch",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_execution_behavior_branch,
            "schema_primary_execution_branch",
        )

    def test_runtime_service_distinguishes_backend_not_capable_from_downgraded_realization(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        ids = iter(["initial-rt-cap-1", "preview-rt-cap-1", "commit-rt-cap-1"])
        executor = CapturingExecutionAdapter(id_factory=lambda: next(ids))
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
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)
        session = service.preview_selected_probe(session.session_id)
        session = service.commit_patch(session.session_id)

        session.selected_workflow_graph_patch.node_patches = []
        session.selected_workflow_graph_patch.edge_patches = []
        session.selected_workflow_graph_patch.region_patches = []
        session.commit_execution_mode = "graph_native_execution_handoff"
        session.commit_execution_authority = "graph_authoritative"
        session.commit_execution_implementation_mode = "graph_primary_execution"
        session.accepted_patch.metadata["commit_execution_mode"] = session.commit_execution_mode
        session.accepted_patch.metadata["commit_execution_authority"] = session.commit_execution_authority
        session.accepted_patch.metadata["commit_execution_implementation_mode"] = (
            session.commit_execution_implementation_mode
        )
        memory.save_session(session)

        session = service.execute_patch(session.session_id)

        self.assertEqual(
            session.latest_execution_source_evidence.request_backend_execution_mode,
            "graph_primary_backend_execution",
        )
        self.assertFalse(
            session.latest_execution_source_evidence.backend_graph_primary_capable,
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_accepted_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_realized_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertFalse(
            session.latest_execution_source_evidence.backend_echoed_graph_primary_capable,
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_accepted_backend_execution_mode,
            "schema_compatible_backend_execution",
        )
        self.assertEqual(
            session.latest_execution_source_evidence.backend_echoed_realized_backend_execution_mode,
            "schema_compatible_backend_execution",
        )

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
        session = service.commit_patch(session.session_id)
        session = service.execute_patch(session.session_id)
        session = service.verify_latest_result(session.session_id)

        self.assertEqual(
            fake_verifier.last_benchmark_comparison_summary.compared_candidate_ids,
            session.benchmark_comparison_summary.compared_candidate_ids,
        )

    def test_runtime_service_preview_probe_override_remains_available(self):
        memory = AgentMemoryService()
        orchestration = FakeOrchestrationService(memory)
        search = FakeSearchService()
        executor = ResultExecutor(id_factory=lambda: "preview-rt-override")
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
        session = service.submit_feedback(session.session_id, "Keep the composition, but improve style.")
        session = service.build_repair_hypotheses(session.session_id)
        session = service.generate_local_probes(session.session_id)

        session = service.preview_probe(session.session_id, "p_001")

        self.assertEqual(session.preview_probe_results[-1].probe_id, "p_001")


if __name__ == "__main__":
    unittest.main()
