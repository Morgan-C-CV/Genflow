from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional
from uuid import uuid4

from app.agents.creative_agent import (
    CandidateWall,
    CreativeIntentPlan,
    ExpandedQuery,
    ResourceRecommendation,
)


@dataclass
class AgentSessionState:
    session_id: str
    original_intent: str
    clarified_intent: str
    clarification_closed: bool = False
    clarification_rounds: int = 0
    plan: Optional[CreativeIntentPlan] = None
    resource_recommendation: Optional[ResourceRecommendation] = None
    previous_expansions: List[ExpandedQuery] = field(default_factory=list)
    previous_wall_indices: List[int] = field(default_factory=list)
    latest_expansions: List[ExpandedQuery] = field(default_factory=list)
    latest_wall: Optional[CandidateWall] = None


class AgentMemoryService:
    def __init__(self):
        self._lock = RLock()
        self._sessions: Dict[str, AgentSessionState] = {}

    def create_session(self, user_intent: str) -> AgentSessionState:
        session = AgentSessionState(
            session_id=str(uuid4()),
            original_intent=user_intent.strip(),
            clarified_intent=user_intent.strip(),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> AgentSessionState:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session

    def save_session(self, session: AgentSessionState) -> AgentSessionState:
        with self._lock:
            self._sessions[session.session_id] = session
        return session
