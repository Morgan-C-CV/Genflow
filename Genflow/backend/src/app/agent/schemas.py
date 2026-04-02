from typing import Dict, List

from pydantic import BaseModel, Field


class AgentSessionView(BaseModel):
    session_id: str
    original_intent: str
    clarified_intent: str
    clarification_closed: bool
    clarification_rounds: int


class AgentPlanView(BaseModel):
    fixed_constraints: Dict[str, str]
    free_variables: List[str]
    locked_axes: List[str]
    unclear_axes: List[str]
    next_action: str
    clarification_questions: List[str]
    reasoning_summary: str


class AgentResourceRecommendationView(BaseModel):
    checkpoint: str
    sampler: str
    loras: List[str]
    reasoning_summary: str


class AgentExpansionView(BaseModel):
    label: str
    prompt: str
    axis_focus: List[str]
    checkpoint: str
    sampler: str
    loras: List[str]


class AgentCandidateWallView(BaseModel):
    groups: List[List[int]]
    flat_indices: List[int]
    query_labels: List[str]
    description: str


class AgentStartRequest(BaseModel):
    user_intent: str = Field(..., min_length=1)


class AgentClarifyRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    answers: List[str] = Field(default_factory=list)


class AgentGenerateCandidatesRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    refresh: bool = False
    per_query_k: int = Field(default=2, ge=1, le=4)
    top_k: int = Field(default=12, ge=4, le=32)


class AgentStartResponse(BaseModel):
    session: AgentSessionView
    plan: AgentPlanView


class AgentClarifyResponse(BaseModel):
    session: AgentSessionView
    plan: AgentPlanView


class AgentCandidatesResponse(BaseModel):
    session: AgentSessionView
    plan: AgentPlanView
    recommendation: AgentResourceRecommendationView
    expansions: List[AgentExpansionView]
    wall: AgentCandidateWallView
