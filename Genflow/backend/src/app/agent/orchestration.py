from dataclasses import replace
from typing import List

from app.agent.memory import AgentMemoryService, AgentSessionState
from app.agent.tools import AgentToolsService


class AgentOrchestrationService:
    def __init__(self, tools_service: AgentToolsService, memory_service: AgentMemoryService):
        self.tools_service = tools_service
        self.memory_service = memory_service

    def start_session(self, user_intent: str) -> AgentSessionState:
        session = self.memory_service.create_session(user_intent)
        plan = self.tools_service.analyze_intent(
            user_intent=session.clarified_intent,
            clarification_closed=False,
        )
        session.plan = plan
        return self.memory_service.save_session(session)

    def submit_clarification(self, session_id: str, answers: List[str]) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        answers = [str(a).strip() for a in answers]
        clarification_closed = session.clarification_closed

        if not answers:
            clarification_closed = True

        for answer in answers:
            if answer and not self._is_unknown_response(answer):
                session.clarified_intent += f" | {answer}"
                continue
            clarification_closed = True
            break

        session.clarification_closed = clarification_closed
        session.clarification_rounds += 1

        if clarification_closed:
            fallback_text = "用户表示不清楚或不想继续补充，请Agent自主推断。"
            if fallback_text not in session.clarified_intent:
                session.clarified_intent += f" | {fallback_text}"

        plan = self.tools_service.analyze_intent(
            user_intent=session.clarified_intent,
            clarification_closed=session.clarification_closed,
        )
        if session.clarification_closed and plan.next_action == "ask_user":
            plan = replace(plan, next_action="retrieve_resources", clarification_questions=[])
        session.plan = plan
        return self.memory_service.save_session(session)

    def generate_candidates(
        self,
        session_id: str,
        refresh: bool = False,
        per_query_k: int = 2,
        top_k: int = 12,
    ) -> AgentSessionState:
        session = self.memory_service.get_session(session_id)
        if session.plan is None:
            session.plan = self.tools_service.analyze_intent(
                user_intent=session.clarified_intent,
                clarification_closed=session.clarification_closed,
            )

        if refresh and session.latest_expansions:
            session.previous_expansions.extend(session.latest_expansions)

        expansions = self.tools_service.build_axis_expansions(
            user_intent=session.clarified_intent,
            plan=session.plan,
            recommendation=None,
            previous_expansions=session.previous_expansions,
            force_refresh=refresh,
        )
        wall = self.tools_service.build_candidate_wall(
            expansions=expansions,
            per_query_k=per_query_k,
            top_k=top_k,
        )
        session.resource_recommendation = self.tools_service.creative_agent.summarize_expansion_resources(expansions)
        session.latest_expansions = expansions
        session.latest_wall = wall
        return self.memory_service.save_session(session)

    def describe_latest_wall(self, session_id: str) -> str:
        session = self.memory_service.get_session(session_id)
        if session.latest_wall is None:
            return ""
        return self.tools_service.describe_wall(session.latest_wall)

    @staticmethod
    def _is_unknown_response(text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return True
        patterns = [
            "不知道",
            "不清楚",
            "随便",
            "你决定",
            "都可以",
            "无所谓",
            "no idea",
            "anything",
            "whatever",
            "don't know",
            "dont know",
        ]
        return any(p in normalized for p in patterns)
