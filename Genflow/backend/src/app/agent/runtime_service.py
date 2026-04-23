from __future__ import annotations

from typing import Callable, Optional

from app.agent.memory import AgentMemoryService, AgentSessionState
from app.agent.result_executor import ResultExecutor
from app.agent.schema_utils import parse_and_normalize_metadata


class AgentRuntimeService:
    def __init__(
        self,
        memory_service: AgentMemoryService,
        orchestration_service,
        search_service,
        result_executor: ResultExecutor,
        schema_normalizer: Optional[Callable[[str], object]] = None,
    ):
        self.memory_service = memory_service
        self.orchestration_service = orchestration_service
        self.search_service = search_service
        self.result_executor = result_executor
        self.schema_normalizer = schema_normalizer or parse_and_normalize_metadata

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
        return self.memory_service.save_session(session)

    def produce_initial_result(self, session_id: str) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        if not session.current_schema_raw:
            raise ValueError("Current schema is missing; generate initial schema first.")
        payload, summary = self.result_executor.produce_initial_result(
            schema=session.current_schema,
            reference_bundle=session.selected_reference_bundle,
        )
        session.current_result_id = payload.result_id
        session.current_result_payload = payload
        session.current_result_summary = summary
        session.accepted_results.append(payload)
        return self.memory_service.save_session(session)

    @staticmethod
    def _build_anchor_summary(reference_bundle: dict) -> str:
        counts = reference_bundle.get("counts", {})
        references = reference_bundle.get("references", [])
        return (
            f"Selected gallery anchor bundle with {len(references)} references "
            f"(best={counts.get('best', 0)}, complementary_knn={counts.get('complementary_knn', 0)}, "
            f"exploratory={counts.get('exploratory', 0)}, counterexample={counts.get('counterexample', 0)})."
        )
