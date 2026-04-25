from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from app.agent.execution_adapter import ExecutionAdapter
from app.agent.feedback_parser import FeedbackParser
from app.agent.memory import AgentMemoryService, AgentSessionState
from app.agent.benchmark_comparison_summary import build_benchmark_comparison_summary
from app.agent.orchestration_policy import PolicyDecision, decide_next_action
from app.agent.patch_planner import PatchPlanner
from app.agent.patch_candidate_generator import PatchCandidateGenerator
from app.agent.pbo_benchmark_ranker import rank_benchmark_candidates
from app.agent.pbo_probe_ranker import rank_probe_candidates
from app.agent.pbo_patch_ranker import rank_patch_candidates
from app.agent.pbo_workflow_graph_patch_ranker import rank_workflow_graph_patch_candidates
from app.agent.probe_generator import PreviewProbeGenerator
from app.agent.refinement_benchmark_retriever import retrieve_refinement_benchmark_set
from app.agent.repair_hypothesis import RepairHypothesisBuilder
from app.agent.schema_utils import parse_and_normalize_metadata, serialize_normalized_schema
from app.agent.verifier import Verifier
from app.agent.verifier_repair_recommendation import build_verifier_repair_recommendation
from app.agent.workflow_graph_patch_candidate_builder import build_workflow_graph_patch_candidates
from app.agent.workflow_graph_patch_builder import (
    build_workflow_graph_patch,
    materialize_workflow_graph_patch_from_candidate,
)
from app.agent.workflow_patch_commit_selector import select_commit_patch_winner
from app.agent.runtime_models import ExecutionSourceEvidenceSummary
from app.agent.workflow_runtime_models import WorkflowExecutionConfig, WorkflowIdentity, WorkflowStateSnapshot
from app.agent.workflow_snapshot_builder import build_surrogate_workflow_snapshot


@dataclass
class PolicyStepResult:
    decision: PolicyDecision
    updated_session: AgentSessionState


@dataclass
class PolicyRunStep:
    action: str
    rationale: list[str]
    session_id: str


@dataclass
class PolicyRunResult:
    steps: list[PolicyRunStep]
    final_session: AgentSessionState
    stopped: bool
    stop_reason: str


class AgentRuntimeService:
    def __init__(
        self,
        memory_service: AgentMemoryService,
        orchestration_service,
        search_service,
        execution_adapter: ExecutionAdapter,
        schema_normalizer: Optional[Callable[[str], object]] = None,
        feedback_parser: Optional[FeedbackParser] = None,
        hypothesis_builder: Optional[RepairHypothesisBuilder] = None,
        probe_generator: Optional[PreviewProbeGenerator] = None,
        pbo_probe_ranker=None,
        pbo_benchmark_ranker=None,
        refinement_benchmark_retriever=None,
        patch_candidate_generator=None,
        pbo_patch_ranker=None,
        pbo_workflow_graph_patch_ranker=None,
        patch_planner: Optional[PatchPlanner] = None,
        verifier: Optional[Verifier] = None,
    ):
        self.memory_service = memory_service
        self.orchestration_service = orchestration_service
        self.search_service = search_service
        self.execution_adapter = execution_adapter
        self.schema_normalizer = schema_normalizer or parse_and_normalize_metadata
        self.feedback_parser = feedback_parser or FeedbackParser()
        self.hypothesis_builder = hypothesis_builder or RepairHypothesisBuilder()
        self.probe_generator = probe_generator or PreviewProbeGenerator()
        self.pbo_probe_ranker = pbo_probe_ranker or rank_probe_candidates
        self.pbo_benchmark_ranker = pbo_benchmark_ranker or rank_benchmark_candidates
        self.refinement_benchmark_retriever = refinement_benchmark_retriever or retrieve_refinement_benchmark_set
        self.patch_candidate_generator = (
            patch_candidate_generator
            or (
                patch_planner
                if patch_planner is not None and (hasattr(patch_planner, "generate") or hasattr(patch_planner, "generate_candidates"))
                else PatchCandidateGenerator(planner=patch_planner)
            )
        )
        self.pbo_patch_ranker = pbo_patch_ranker or rank_patch_candidates
        self.pbo_workflow_graph_patch_ranker = (
            pbo_workflow_graph_patch_ranker or rank_workflow_graph_patch_candidates
        )
        self.patch_planner = patch_planner or PatchPlanner()
        self.verifier = verifier or Verifier()

    def start_episode(self, user_intent: str) -> AgentSessionState:
        return self.orchestration_service.start_session(user_intent)

    def clarify_episode(self, session_id: str, answers: list[str]) -> AgentSessionState:
        return self.orchestration_service.submit_clarification(session_id, answers)

    def generate_initial_candidates(
        self,
        session_id: str,
        refresh: bool = False,
        per_query_k: int = 2,
        top_k: int = 12,
    ) -> AgentSessionState:
        return self.orchestration_service.generate_candidates(
            session_id=session_id,
            refresh=refresh,
            per_query_k=per_query_k,
            top_k=top_k,
        )

    def select_initial_reference(self, session_id: str, gallery_index: int) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        reference_bundle = self.search_service.build_diverse_reference_bundle(gallery_index)
        selected_reference_ids = []
        for item in reference_bundle.get("references", []):
            if "id" in item:
                try:
                    selected_reference_ids.append(int(item["id"]))
                except (TypeError, ValueError):
                    continue
        session.selected_gallery_index = int(gallery_index)
        session.selected_reference_bundle = reference_bundle
        session.selected_reference_ids = selected_reference_ids
        session.current_gallery_anchor_summary = self._build_anchor_summary(reference_bundle)
        self._sync_workflow_state(session, execution_kind="reference_select", preview=False)
        return self.memory_service.save_session(session)

    def generate_initial_schema(self, session_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        if not session.selected_reference_bundle:
            raise ValueError("Reference bundle is missing; select initial reference first.")
        metadata_json = self.search_service.generate_image_metadata(
            reference_bundle=session.selected_reference_bundle,
            user_intent=session.clarified_intent,
        )
        normalized_schema = self.schema_normalizer(metadata_json)
        session.current_schema_raw = metadata_json
        session.current_schema = normalized_schema
        self._sync_workflow_state(session, execution_kind="initial_schema", preview=False)
        return self.memory_service.save_session(session)

    def produce_initial_result(self, session_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        if not session.current_schema_raw:
            raise ValueError("Current schema is missing; generate initial schema first.")
        payload, summary = self.execution_adapter.produce_initial_result(
            schema=session.current_schema,
            reference_bundle=session.selected_reference_bundle,
        )
        session.current_result_id = payload.result_id
        session.current_result_payload = payload
        session.current_result_summary = summary
        session.accepted_results.append(payload)
        self._sync_workflow_state(session, execution_kind="initial", preview=False)
        return self.memory_service.save_session(session)

    def submit_feedback(self, session_id: str, feedback_text: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        evidence = self.feedback_parser.parse(
            feedback_text=feedback_text,
            current_result_summary=session.current_result_summary.summary_text,
            current_schema_prompt=session.current_schema.prompt,
        )
        session.feedback_history.append(feedback_text)
        session.latest_feedback = feedback_text
        session.parsed_feedback = evidence
        session.preserve_constraints = list(evidence.preserve_constraints)
        session.dissatisfaction_axes = list(evidence.dissatisfaction_scope)
        session.requested_changes = list(evidence.requested_changes)
        session.current_uncertainty_estimate = evidence.uncertainty_estimate
        self._sync_workflow_state(session, execution_kind="feedback", preview=False)
        return self.memory_service.save_session(session)

    def build_repair_hypotheses(self, session_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        hypotheses = self.hypothesis_builder.build(
            current_schema=session.current_schema,
            current_result_summary=session.current_result_summary,
            feedback_evidence=session.parsed_feedback,
            history=session.feedback_history,
        )
        session.repair_hypotheses = hypotheses
        benchmark_set = self.refinement_benchmark_retriever(
            session,
            search_service=self.search_service,
            pbo_benchmark_ranker=self.pbo_benchmark_ranker,
        )
        session.refinement_benchmark_set = benchmark_set
        session.refinement_benchmark_summary = " | ".join(benchmark_set.selection_rationale)
        session.benchmark_comparison_summary = build_benchmark_comparison_summary(benchmark_set, session)
        self._sync_workflow_state(session, execution_kind="repair_hypotheses", preview=False)
        return self.memory_service.save_session(session)

    def generate_local_probes(self, session_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        probes = self.probe_generator.generate(
            current_schema=session.current_schema,
            parsed_feedback=session.parsed_feedback,
            repair_hypotheses=session.repair_hypotheses,
            selected_gallery_index=session.selected_gallery_index,
            selected_reference_ids=session.selected_reference_ids,
            refinement_benchmark_set=session.refinement_benchmark_set,
        )
        ranked_probes = self.pbo_probe_ranker(
            probes,
            session.parsed_feedback,
            benchmark_comparison_summary=session.benchmark_comparison_summary,
            refinement_benchmark_set=session.refinement_benchmark_set,
        )
        session.local_probes = ranked_probes
        session.preview_probe_candidates = ranked_probes
        self._select_default_probe(session)
        session.workflow_graph_patch_candidates = self.pbo_workflow_graph_patch_ranker(
            build_workflow_graph_patch_candidates(session),
            session,
        )
        self._sync_workflow_state(session, execution_kind="probe_generation", preview=False)
        return self.memory_service.save_session(session)

    def preview_probe(self, session_id: str, probe_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        probe = next((item for item in session.preview_probe_candidates if item.probe_id == probe_id), None)
        if probe is None:
            raise ValueError(f"Preview probe not found: {probe_id}")

        # Preview must not mutate committed state.
        preview_result = self.execution_adapter.execute_preview_probe(
            schema=session.current_schema,
            probe=probe,
        )
        session.preview_results.append(preview_result)
        session.preview_probe_results.append(preview_result)
        self._sync_workflow_state(session, execution_kind="preview", preview=True)
        return self.memory_service.save_session(session)

    def preview_selected_probe(self, session_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        if not session.selected_probe.probe_id:
            raise ValueError("No selected probe available for preview.")
        return self.preview_probe(session_id, session.selected_probe.probe_id)

    def select_probe(self, session_id: str, probe_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        probe = next((item for item in session.preview_probe_candidates if item.probe_id == probe_id), None)
        if probe is None:
            raise ValueError(f"Preview probe not found: {probe_id}")
        session.selected_probe = probe
        session.workflow_graph_patch_candidates = self.pbo_workflow_graph_patch_ranker(
            build_workflow_graph_patch_candidates(session),
            session,
        )
        self._sync_workflow_state(session, execution_kind="probe_select", preview=False)
        return self.memory_service.save_session(session)

    def commit_patch(self, session_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        if not session.selected_probe.probe_id:
            raise ValueError("No selected probe available for commit.")
        patch_candidates = self._generate_patch_candidates(session)
        ranked_patch_candidates = self.pbo_patch_ranker(
            patch_candidates,
            session.parsed_feedback,
            benchmark_comparison_summary=session.benchmark_comparison_summary,
            refinement_benchmark_set=session.refinement_benchmark_set,
        )
        session.top_schema_patch_candidate = ranked_patch_candidates[0] if ranked_patch_candidates else CommittedPatch()
        session.top_workflow_graph_patch_candidate = (
            session.workflow_graph_patch_candidates[0]
            if session.workflow_graph_patch_candidates
            else type(session.top_workflow_graph_patch_candidate)()
        )
        patch = ranked_patch_candidates[0]
        session.accepted_patch = patch
        self._annotate_patch_winner_alignment(session)
        commit_selection = select_commit_patch_winner(
            schema_patch_winner=session.top_schema_patch_candidate,
            graph_patch_winner=session.top_workflow_graph_patch_candidate,
            session=session,
        )
        session.preferred_commit_source = commit_selection.preferred_commit_source
        session.selected_graph_native_patch_candidate = commit_selection.selected_graph_native_patch_candidate
        session.selected_workflow_graph_patch = (
            materialize_workflow_graph_patch_from_candidate(
                commit_selection.selected_graph_native_patch_candidate,
                session=session,
            )
            if session.preferred_commit_source == "graph"
            else type(session.selected_workflow_graph_patch)()
        )
        session.commit_execution_mode = self._determine_commit_execution_mode(session)
        session.accepted_patch.metadata["commit_selection_rationale"] = list(commit_selection.rationale)
        session.accepted_patch.metadata["preferred_commit_source"] = session.preferred_commit_source
        session.accepted_patch.metadata["commit_execution_mode"] = session.commit_execution_mode
        session.accepted_patch.metadata["top_schema_patch_id"] = session.top_schema_patch_candidate.patch_id
        session.accepted_patch.metadata["top_graph_patch_candidate_id"] = (
            session.top_workflow_graph_patch_candidate.candidate_id
        )
        if session.selected_workflow_graph_patch.patch_id:
            session.accepted_patch.metadata["selected_workflow_graph_patch_id"] = (
                session.selected_workflow_graph_patch.patch_id
            )
        session.patch_history.append(patch)
        session.current_schema = self._apply_patch_to_schema(session.current_schema, patch)
        session.current_schema_raw = serialize_normalized_schema(session.current_schema)
        if not session.workflow_graph_patch_candidates:
            session.workflow_graph_patch_candidates = self.pbo_workflow_graph_patch_ranker(
                build_workflow_graph_patch_candidates(session),
                session,
            )
            session.top_workflow_graph_patch_candidate = (
                session.workflow_graph_patch_candidates[0]
                if session.workflow_graph_patch_candidates
                else type(session.top_workflow_graph_patch_candidate)()
            )
        session.current_workflow_graph_patch = build_workflow_graph_patch(session)
        self._sync_workflow_state(session, execution_kind="commit_plan", preview=False)
        return self.memory_service.save_session(session)

    def execute_patch(self, session_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        if not session.accepted_patch.patch_id:
            raise ValueError("No committed patch available for execution.")
        session.previous_result_summary = session.current_result_summary
        payload, summary = self.execution_adapter.execute_committed_patch(
            schema=session.current_schema,
            patch=session.accepted_patch,
        )
        session.current_result_id = payload.result_id
        session.current_result_payload = payload
        session.current_result_summary = summary
        session.accepted_results.append(payload)
        backend_metadata = dict(payload.artifacts.get("backend_metadata", {}))
        session.latest_execution_source_evidence = ExecutionSourceEvidenceSummary(
            commit_execution_mode=session.commit_execution_mode,
            preferred_commit_source=session.preferred_commit_source,
            selected_workflow_graph_patch_id=session.selected_workflow_graph_patch.patch_id,
            top_schema_patch_id=session.top_schema_patch_candidate.patch_id,
            top_graph_patch_candidate_id=session.top_workflow_graph_patch_candidate.candidate_id,
            request_patch_id=session.accepted_patch.patch_id,
            response_patch_id=str(payload.content.get("patch_id", "")),
            backend_graph_patch_id=str(backend_metadata.get("graph_patch_id", "")),
            backend_echoed_commit_source=str(backend_metadata.get("preferred_commit_source", "")),
            backend_echoed_commit_execution_mode=str(backend_metadata.get("commit_execution_mode", "")),
            comparison_notes=list(summary.notes),
        )
        self._sync_workflow_state(session, execution_kind="commit", preview=False)
        return self.memory_service.save_session(session)

    def verify_latest_result(self, session_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        result = self.verifier.verify(
            previous_result_summary=session.previous_result_summary,
            updated_result_summary=session.current_result_summary,
            selected_probe=session.selected_probe,
            committed_patch=session.accepted_patch,
            preserve_constraints=session.preserve_constraints,
            benchmark_comparison_summary=session.benchmark_comparison_summary,
        )
        session.latest_verifier_result = result
        session.latest_verifier_signal_summary = result.signal_summary
        session.latest_verifier_repair_recommendation = build_verifier_repair_recommendation(
            verifier_signal_summary=result.signal_summary,
            verifier_result=result,
            session=session,
        )
        session.continue_recommended = result.continue_recommended
        session.verifier_confidence = result.confidence
        session.stop_reason = "" if result.continue_recommended else "verifier_accepts_current_direction"
        self._sync_workflow_state(session, execution_kind="verify", preview=False)
        return self.memory_service.save_session(session)

    def should_continue(self, session_id: str) -> bool:
        session = self.memory_service.get_session(session_id)
        return bool(session.continue_recommended)

    def get_policy_decision(self, session_id: str) -> PolicyDecision:
        session = self.memory_service.get_session(session_id)
        return decide_next_action(session)

    def run_next_policy_step(self, session_id: str) -> PolicyStepResult:
        decision = self.get_policy_decision(session_id)
        action = decision.next_action
        if action == "build_hypotheses":
            updated_session = self.build_repair_hypotheses(session_id)
        elif action == "retrieve_benchmarks":
            # Benchmark retrieval currently lives inside build_repair_hypotheses().
            updated_session = self.build_repair_hypotheses(session_id)
        elif action == "generate_probes":
            updated_session = self.generate_local_probes(session_id)
        elif action == "preview_selected_probe":
            updated_session = self.preview_selected_probe(session_id)
        elif action == "commit_selected_patch":
            updated_session = self.commit_patch(session_id)
        elif action == "execute_patch":
            updated_session = self.execute_patch(session_id)
        elif action == "verify_latest_result":
            updated_session = self.verify_latest_result(session_id)
        elif action == "stop":
            updated_session = self.memory_service.get_session(session_id)
        else:
            raise ValueError(f"Unsupported policy action: {action}")
        return PolicyStepResult(decision=decision, updated_session=updated_session)

    def run_policy_steps(self, session_id: str, max_steps: int = 5) -> PolicyRunResult:
        max_steps = max(1, int(max_steps))
        steps: list[PolicyRunStep] = []
        final_session = self.memory_service.get_session(session_id)
        stopped = False
        stop_reason = ""

        for _ in range(max_steps):
            step_result = self.run_next_policy_step(session_id)
            final_session = step_result.updated_session
            steps.append(
                PolicyRunStep(
                    action=step_result.decision.next_action,
                    rationale=list(step_result.decision.rationale),
                    session_id=final_session.session_id,
                )
            )
            if step_result.decision.next_action == "stop":
                stopped = True
                stop_reason = final_session.stop_reason or "policy_stop"
                break

        if not stopped and final_session.stop_reason:
            stop_reason = final_session.stop_reason
        elif not stopped:
            stop_reason = "max_steps_reached"

        return PolicyRunResult(
            steps=steps,
            final_session=final_session,
            stopped=stopped,
            stop_reason=stop_reason,
        )

    @staticmethod
    def _build_anchor_summary(reference_bundle: dict) -> str:
        counts = reference_bundle.get("counts", {})
        references = reference_bundle.get("references", [])
        return (
            f"Selected gallery anchor bundle with {len(references)} references "
            f"(best={counts.get('best', 0)}, complementary_knn={counts.get('complementary_knn', 0)}, "
            f"exploratory={counts.get('exploratory', 0)}, counterexample={counts.get('counterexample', 0)})."
        )

    @staticmethod
    def _annotate_patch_winner_alignment(session: AgentSessionState) -> None:
        schema_winner = session.top_schema_patch_candidate
        graph_winner = session.top_workflow_graph_patch_candidate
        if not schema_winner.patch_id or not graph_winner.candidate_id:
            return
        schema_axes = set(schema_winner.target_axes)
        graph_axes = set(graph_winner.target_axes)
        aligned = bool(schema_axes) and schema_axes == graph_axes
        if aligned:
            session.accepted_patch.metadata["graph_native_aligned_winner"] = True
            session.accepted_patch.metadata["aligned_graph_candidate_id"] = graph_winner.candidate_id

    @staticmethod
    def _determine_commit_execution_mode(session: AgentSessionState) -> str:
        graph_patch = session.selected_workflow_graph_patch
        graph_artifact_complete = bool(graph_patch.patch_id and graph_patch.node_patches)
        if session.preferred_commit_source == "graph" and graph_artifact_complete:
            return "graph_native_execution_handoff"
        return "schema_execution_fallback"

    def _sync_workflow_identity(self, session: AgentSessionState) -> None:
        workflow_id = session.workflow_id or f"workflow-{session.session_id}"
        workflow_kind = "normalized_schema_surrogate"
        identity = WorkflowIdentity(
            workflow_id=workflow_id,
            workflow_kind=workflow_kind,
            workflow_version="phase-g-skeleton",
        )
        session.workflow_id = workflow_id
        session.workflow_identity = identity
        session.workflow_state.identity = identity

    def _sync_workflow_state(
        self,
        session: AgentSessionState,
        execution_kind: str = "",
        preview: bool = False,
    ) -> None:
        self._sync_workflow_identity(session)
        backend_kind, workflow_profile = self._infer_backend_descriptor()
        snapshot = build_surrogate_workflow_snapshot(
            session=session,
            execution_kind=execution_kind,
            preview=preview,
            backend_kind=backend_kind,
            workflow_profile=workflow_profile,
        )
        execution_config = WorkflowExecutionConfig(
            execution_kind=execution_kind,
            preview=preview,
            backend_kind=backend_kind,
            workflow_profile=workflow_profile,
            parameters={
                "selected_gallery_index": session.selected_gallery_index,
                "selected_reference_ids": list(session.selected_reference_ids),
                "selected_probe_id": session.selected_probe.probe_id,
                "accepted_patch_id": session.accepted_patch.patch_id,
                "current_result_id": session.current_result_id,
            },
        )

        session.workflow_id = snapshot.workflow_identity.workflow_id
        session.workflow_identity = snapshot.workflow_identity
        session.editable_scopes = snapshot.editable_scopes
        session.protected_scopes = snapshot.protected_scopes
        session.last_execution_config = execution_config
        session.workflow_metadata = snapshot.workflow_metadata
        session.workflow_graph_placeholder = snapshot.workflow_graph_placeholder
        session.workflow_topology_hints = snapshot.workflow_topology_hints
        session.workflow_topology_entry_node_ids = list(snapshot.workflow_topology_entry_node_ids)
        session.workflow_topology_exit_node_ids = list(snapshot.workflow_topology_exit_node_ids)
        session.workflow_state = WorkflowStateSnapshot(
            identity=snapshot.workflow_identity,
            editable_scopes=snapshot.editable_scopes,
            protected_scopes=snapshot.protected_scopes,
            last_execution_config=execution_config,
            workflow_metadata=snapshot.workflow_metadata,
            surrogate_payload=snapshot.surrogate_payload,
            workflow_graph_placeholder=snapshot.workflow_graph_placeholder,
        )

    def _infer_backend_descriptor(self) -> tuple[str, str]:
        adapter_name = type(self.execution_adapter).__name__.lower()
        if "live" in adapter_name:
            backend_client = getattr(self.execution_adapter, "backend_client", None)
            transport = getattr(backend_client, "transport", None)
            config = getattr(transport, "config", None)
            if config is not None:
                return (
                    getattr(config, "backend_kind", "") or "live_backend",
                    getattr(config, "workflow_profile", "") or "default",
                )
            return ("live_backend", "default")
        return ("mock", "default")

    @staticmethod
    def _select_default_probe(session: AgentSessionState) -> None:
        if not session.preview_probe_candidates:
            return
        if session.selected_probe.probe_id:
            matched_probe = next(
                (probe for probe in session.preview_probe_candidates if probe.probe_id == session.selected_probe.probe_id),
                None,
            )
            if matched_probe is not None:
                session.selected_probe = matched_probe
                return
        session.selected_probe = session.preview_probe_candidates[0]

    def _generate_patch_candidates(self, session: AgentSessionState):
        generator = self.patch_candidate_generator
        kwargs = {
            "selected_probe": session.selected_probe,
            "current_schema": session.current_schema,
            "parsed_feedback": session.parsed_feedback,
            "repair_hypotheses": session.repair_hypotheses,
        }
        if hasattr(generator, "generate"):
            return generator.generate(**kwargs)
        if hasattr(generator, "generate_candidates"):
            return generator.generate_candidates(**kwargs)
        if hasattr(generator, "plan"):
            return [generator.plan(**kwargs)]
        raise TypeError("Patch candidate generator must provide generate(), generate_candidates(), or plan().")

    @staticmethod
    def _apply_patch_to_schema(current_schema, patch):
        updated = type(current_schema)(
            prompt=current_schema.prompt,
            negative_prompt=current_schema.negative_prompt,
            cfgscale=current_schema.cfgscale,
            steps=current_schema.steps,
            sampler=current_schema.sampler,
            seed=current_schema.seed,
            model=current_schema.model,
            clipskip=current_schema.clipskip,
            style=list(current_schema.style),
            lora=list(current_schema.lora),
            full_metadata_string=current_schema.full_metadata_string,
            raw_fields=dict(current_schema.raw_fields),
        )
        for field, value in patch.changes.items():
            if hasattr(updated, field):
                setattr(updated, field, value)
        updated.raw_fields.update(
            {
                "prompt": updated.prompt,
                "negative_prompt": updated.negative_prompt,
                "cfgscale": updated.cfgscale,
                "steps": updated.steps,
                "sampler": updated.sampler,
                "seed": updated.seed,
                "model": updated.model,
                "clipskip": updated.clipskip,
                "style": ", ".join(updated.style) if updated.style else "none",
                "lora": ", ".join(updated.lora) if updated.lora else "none",
                "full_metadata_string": updated.full_metadata_string,
            }
        )
        return updated
