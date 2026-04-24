from __future__ import annotations

from typing import Callable, Optional

from app.agent.execution_adapter import ExecutionAdapter
from app.agent.feedback_parser import FeedbackParser
from app.agent.memory import AgentMemoryService, AgentSessionState
from app.agent.patch_planner import PatchPlanner
from app.agent.probe_generator import PreviewProbeGenerator
from app.agent.repair_hypothesis import RepairHypothesisBuilder
from app.agent.schema_utils import parse_and_normalize_metadata, serialize_normalized_schema
from app.agent.verifier import Verifier
from app.agent.workflow_runtime_models import WorkflowExecutionConfig, WorkflowIdentity, WorkflowStateSnapshot
from app.agent.workflow_snapshot_builder import build_surrogate_workflow_snapshot


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
        )
        session.local_probes = probes
        session.preview_probe_candidates = probes
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

    def select_probe(self, session_id: str, probe_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        probe = next((item for item in session.preview_probe_candidates if item.probe_id == probe_id), None)
        if probe is None:
            raise ValueError(f"Preview probe not found: {probe_id}")
        session.selected_probe = probe
        self._sync_workflow_state(session, execution_kind="probe_select", preview=False)
        return self.memory_service.save_session(session)

    def commit_patch(self, session_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        if not session.selected_probe.probe_id:
            raise ValueError("No selected probe available for commit.")
        patch = self.patch_planner.plan(
            selected_probe=session.selected_probe,
            current_schema=session.current_schema,
            parsed_feedback=session.parsed_feedback,
            repair_hypotheses=session.repair_hypotheses,
        )
        session.accepted_patch = patch
        session.patch_history.append(patch)
        session.current_schema = self._apply_patch_to_schema(session.current_schema, patch)
        session.current_schema_raw = serialize_normalized_schema(session.current_schema)
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
        )
        session.latest_verifier_result = result
        session.continue_recommended = result.continue_recommended
        session.verifier_confidence = result.confidence
        session.stop_reason = "" if result.continue_recommended else "verifier_accepts_current_direction"
        return self.memory_service.save_session(session)

    def should_continue(self, session_id: str) -> bool:
        session = self.memory_service.get_session(session_id)
        return bool(session.continue_recommended)

    @staticmethod
    def _build_anchor_summary(reference_bundle: dict) -> str:
        counts = reference_bundle.get("counts", {})
        references = reference_bundle.get("references", [])
        return (
            f"Selected gallery anchor bundle with {len(references)} references "
            f"(best={counts.get('best', 0)}, complementary_knn={counts.get('complementary_knn', 0)}, "
            f"exploratory={counts.get('exploratory', 0)}, counterexample={counts.get('counterexample', 0)})."
        )

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
