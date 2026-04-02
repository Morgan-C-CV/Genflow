from fastapi import APIRouter, Depends, HTTPException

from app.agent.schemas import (
    AgentCandidatesResponse,
    AgentCandidateWallView,
    AgentClarifyRequest,
    AgentClarifyResponse,
    AgentExpansionView,
    AgentGenerateCandidatesRequest,
    AgentPlanView,
    AgentResourceRecommendationView,
    AgentSessionView,
    AgentStartRequest,
    AgentStartResponse,
)
from app.agent.memory import AgentMemoryService, AgentSessionState
from app.agent.orchestration import AgentOrchestrationService
from app.agent.tools import AgentToolsService
from app.agents.creative_agent import CreativeAgent
from app.repositories.search_repository import SearchRepository

router = APIRouter()

_search_repo = None
_agent_service = None


def get_agent_service():
    global _search_repo, _agent_service
    if _search_repo is None:
        _search_repo = SearchRepository()
    if _agent_service is None:
        tools = AgentToolsService(creative_agent=CreativeAgent(), search_repo=_search_repo)
        memory = AgentMemoryService()
        _agent_service = AgentOrchestrationService(
            tools_service=tools,
            memory_service=memory,
        )
    return _agent_service


def _to_session_view(session: AgentSessionState) -> AgentSessionView:
    return AgentSessionView(
        session_id=session.session_id,
        original_intent=session.original_intent,
        clarified_intent=session.clarified_intent,
        clarification_closed=session.clarification_closed,
        clarification_rounds=session.clarification_rounds,
    )


def _to_plan_view(session: AgentSessionState) -> AgentPlanView:
    if session.plan is None:
        raise ValueError("Plan is missing in session state.")
    return AgentPlanView(
        fixed_constraints=session.plan.fixed_constraints,
        free_variables=session.plan.free_variables,
        locked_axes=session.plan.locked_axes,
        unclear_axes=session.plan.unclear_axes,
        next_action=session.plan.next_action,
        clarification_questions=session.plan.clarification_questions,
        reasoning_summary=session.plan.reasoning_summary,
    )


@router.post("/start", response_model=AgentStartResponse)
def start_agent_session(
    request: AgentStartRequest,
    service: AgentOrchestrationService = Depends(get_agent_service),
):
    try:
        session = service.start_session(request.user_intent)
        return AgentStartResponse(
            session=_to_session_view(session),
            plan=_to_plan_view(session),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clarify", response_model=AgentClarifyResponse)
def clarify_agent_session(
    request: AgentClarifyRequest,
    service: AgentOrchestrationService = Depends(get_agent_service),
):
    try:
        session = service.submit_clarification(
            session_id=request.session_id,
            answers=request.answers,
        )
        return AgentClarifyResponse(
            session=_to_session_view(session),
            plan=_to_plan_view(session),
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates", response_model=AgentCandidatesResponse)
def generate_agent_candidates(
    request: AgentGenerateCandidatesRequest,
    service: AgentOrchestrationService = Depends(get_agent_service),
):
    try:
        session = service.generate_candidates(
            session_id=request.session_id,
            refresh=request.refresh,
            per_query_k=request.per_query_k,
            top_k=request.top_k,
        )
        if session.resource_recommendation is None or session.latest_wall is None:
            raise ValueError("Candidate generation did not produce recommendation or wall.")
        recommendation = AgentResourceRecommendationView(
            checkpoint=session.resource_recommendation.checkpoint,
            sampler=session.resource_recommendation.sampler,
            loras=session.resource_recommendation.loras,
            reasoning_summary=session.resource_recommendation.reasoning_summary,
        )
        expansions = [
            AgentExpansionView(
                label=item.label,
                prompt=item.prompt,
                axis_focus=item.axis_focus,
                checkpoint=item.checkpoint or "",
                sampler=item.sampler or "",
                loras=item.loras,
            )
            for item in session.latest_expansions
        ]
        wall = AgentCandidateWallView(
            groups=session.latest_wall.groups,
            flat_indices=session.latest_wall.flat_indices,
            query_labels=session.latest_wall.query_labels,
            description=service.describe_latest_wall(session.session_id),
        )
        return AgentCandidatesResponse(
            session=_to_session_view(session),
            plan=_to_plan_view(session),
            recommendation=recommendation,
            expansions=expansions,
            wall=wall,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
