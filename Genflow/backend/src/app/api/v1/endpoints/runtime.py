"""Interactive Genflow runtime API.

Exposes the canonical agent pipeline (start -> clarify -> candidates -> select ->
schema -> result) and the ComfyUI hand-off that turns the generated schema into a
runnable ComfyUI workflow.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.agent.memory import AgentMemoryService, AgentSessionState
from app.agent.runtime_schemas import (
    ComfyUIStatusResponse,
    RuntimeCandidatesRequest,
    RuntimeCandidatesResponse,
    RuntimeCandidateView,
    RuntimeClarifyRequest,
    RuntimeExpansionView,
    RuntimePlanResponse,
    RuntimePlanView,
    RuntimePushResponse,
    RuntimeRecommendationView,
    RuntimeRefineFeedbackRequest,
    RuntimeRefineResponse,
    RuntimeRefineRoundRequest,
    RuntimeRefinementView,
    RuntimeRefineStartRequest,
    RuntimeResultResponse,
    RuntimeSchemaResponse,
    RuntimeSelectRequest,
    RuntimeSelectResponse,
    RuntimeSessionView,
    RuntimeShowcaseRequest,
    RuntimeShowcaseResponse,
    RuntimeStartRequest,
    RuntimeStartResponse,
    RuntimeWallView,
    RuntimeWorkflowRequest,
    RuntimeWorkflowResponse,
)
from app.core.config import settings
from app.core.llm_client import active_provider
from app.modules import gallery_catalog
from app.repositories.comfyui_repository import ComfyUIError
from app.services.comfyui_service import ComfyUIService

router = APIRouter()

_runtime_service = None
_comfy_service: Optional[ComfyUIService] = None


# ---------------------------------------------------------------------------
# service wiring
# ---------------------------------------------------------------------------


def get_runtime_service():
    """Build the stateful runtime service once per process.

    Mirrors ``run_agent_demo.build_runtime_service`` without importing the CLI
    script, so the API layer only depends on ``app.*``.
    """
    global _runtime_service
    if _runtime_service is not None:
        return _runtime_service

    from app.agent.feedback_parser import FeedbackParser
    from app.agent.patch_planner import PatchPlanner
    from app.agent.probe_generator import PreviewProbeGenerator
    from app.agent.repair_hypothesis import RepairHypothesisBuilder
    from app.agent.result_executor import ResultExecutor
    from app.agent.runtime_service import AgentRuntimeService
    from app.agent.orchestration import AgentOrchestrationService
    from app.agent.tools import AgentToolsService
    from app.agent.verifier import Verifier
    from app.agents.creative_agent import CreativeAgent
    from app.repositories.llm_repository import LLMRepository
    from app.repositories.search_repository import SearchRepository
    from app.services.search_service import SearchService

    search_repo = SearchRepository()
    memory = AgentMemoryService()
    orchestration = AgentOrchestrationService(
        tools_service=AgentToolsService(creative_agent=CreativeAgent(), search_repo=search_repo),
        memory_service=memory,
    )
    _runtime_service = AgentRuntimeService(
        memory_service=memory,
        orchestration_service=orchestration,
        search_service=SearchService(search_repo=search_repo, llm_repo=LLMRepository()),
        execution_adapter=ResultExecutor(),
        feedback_parser=FeedbackParser(),
        hypothesis_builder=RepairHypothesisBuilder(),
        probe_generator=PreviewProbeGenerator(),
        patch_planner=PatchPlanner(),
        verifier=Verifier(),
    )
    return _runtime_service


def get_comfy_service() -> ComfyUIService:
    global _comfy_service
    if _comfy_service is None:
        _comfy_service = ComfyUIService()
    return _comfy_service


def _service_dependencies_ready() -> Optional[str]:
    """Return a human-readable reason when the agent cannot run at all."""
    if active_provider() == "deepseek":
        if not settings.DEEPSEEK_API_KEY.strip():
            return "DEEPSEEK_API_KEY is not configured; the Genflow planner cannot run."
        return None
    if not settings.GOOGLE_API_KEY.strip():
        return "GOOGLE_API_KEY is not configured; the Genflow planner cannot run."
    return None


# ---------------------------------------------------------------------------
# serialisation helpers
# ---------------------------------------------------------------------------


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _session_view(session: AgentSessionState) -> RuntimeSessionView:
    return RuntimeSessionView(
        session_id=session.session_id,
        original_intent=session.original_intent,
        clarified_intent=session.clarified_intent,
        clarification_closed=session.clarification_closed,
        clarification_rounds=session.clarification_rounds,
        selected_gallery_index=session.selected_gallery_index,
        next_action=session.plan.next_action if session.plan else "",
    )


def _plan_view(session: AgentSessionState) -> RuntimePlanView:
    plan = session.plan
    if plan is None:
        return RuntimePlanView(user_intent=session.clarified_intent)
    return RuntimePlanView(
        user_intent=getattr(plan, "user_intent", session.clarified_intent),
        fixed_constraints=dict(getattr(plan, "fixed_constraints", {}) or {}),
        free_variables=list(getattr(plan, "free_variables", []) or []),
        locked_axes=list(getattr(plan, "locked_axes", []) or []),
        unclear_axes=list(getattr(plan, "unclear_axes", []) or []),
        next_action=getattr(plan, "next_action", "") or "",
        clarification_questions=list(getattr(plan, "clarification_questions", []) or []),
        reasoning_summary=getattr(plan, "reasoning_summary", "") or "",
    )


def _gallery_row(service, gallery_index: int) -> Dict[str, Any]:
    df = service.search_service.search_repo.get_all_data()
    if gallery_index < 0 or gallery_index >= len(df):
        raise KeyError(f"Gallery index out of range: {gallery_index}")
    row = df.iloc[gallery_index].to_dict()
    cleaned: Dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                value = value.item()
            except (ValueError, AttributeError):
                value = str(value)
        cleaned[str(key)] = value
    return cleaned


def _image_url_for(gallery_index: int) -> str:
    # The dedicated gallery route serves a cached thumbnail without touching the
    # embedding stack, so grids render fast even before the search service is warm.
    return gallery_catalog.image_url(gallery_index)


def _candidate_view(service, gallery_index: int, slot: int, group_index: int, group_label: str) -> RuntimeCandidateView:
    row = _gallery_row(service, gallery_index)
    width = row.get("width")
    height = row.get("height")

    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return RuntimeCandidateView(
        slot=slot,
        gallery_index=gallery_index,
        id=str(row.get("id", "") or ""),
        image_url=_image_url_for(gallery_index),
        local_path=str(row.get("local_path", "") or ""),
        prompt=str(row.get("prompt", "") or ""),
        negative_prompt=str(row.get("negative_prompt", "") or ""),
        model=str(row.get("model", "") or ""),
        sampler=str(row.get("sampler", "") or ""),
        steps=str(row.get("steps", "") or ""),
        cfgscale=str(row.get("cfgscale", "") or ""),
        seed=str(row.get("seed", "") or ""),
        width=_as_int(width),
        height=_as_int(height),
        group_index=group_index,
        group_label=group_label,
        distance=float(row["distance"]) if isinstance(row.get("distance"), (int, float)) else None,
    )


def _wall_view(service, session: AgentSessionState) -> RuntimeWallView:
    wall = session.latest_wall
    if wall is None:
        return RuntimeWallView()

    candidates: List[RuntimeCandidateView] = []
    slot = 0
    for group_index, group in enumerate(wall.groups):
        label = wall.query_labels[group_index] if group_index < len(wall.query_labels) else ""
        for gallery_index in group:
            slot += 1
            try:
                candidates.append(_candidate_view(service, int(gallery_index), slot, group_index + 1, label))
            except KeyError:
                continue

    try:
        description = service.orchestration_service.describe_latest_wall(session.session_id)
    except Exception:  # noqa: BLE001 - description is decorative
        description = ""

    return RuntimeWallView(
        groups=[list(group) for group in wall.groups],
        query_labels=list(wall.query_labels),
        candidates=candidates,
        description=description or "",
    )


def _recommendation_view(session: AgentSessionState) -> RuntimeRecommendationView:
    rec = session.resource_recommendation
    if rec is None:
        return RuntimeRecommendationView()
    return RuntimeRecommendationView(
        checkpoint=getattr(rec, "checkpoint", "") or "",
        sampler=getattr(rec, "sampler", "") or "",
        loras=list(getattr(rec, "loras", []) or []),
        reasoning_summary=getattr(rec, "reasoning_summary", "") or "",
    )


def _get_session(service, session_id: str) -> AgentSessionState:
    try:
        return service.memory_service.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _require_schema(session: AgentSessionState) -> None:
    if not session.current_schema_raw:
        raise HTTPException(
            status_code=409,
            detail="No schema yet. Select a candidate and generate the schema first.",
        )


# ---------------------------------------------------------------------------
# pipeline endpoints
# ---------------------------------------------------------------------------


@router.post("/episodes", response_model=RuntimeStartResponse)
def start_episode(request: RuntimeStartRequest):
    reason = _service_dependencies_ready()
    if reason:
        raise HTTPException(status_code=503, detail=reason)
    service = get_runtime_service()
    try:
        session = service.start_episode(request.user_intent)
    except Exception as exc:  # noqa: BLE001 - surface upstream planner failures
        raise HTTPException(status_code=502, detail=f"Planner failed: {exc}") from exc
    return RuntimeStartResponse(session=_session_view(session), plan=_plan_view(session))


@router.post("/episodes/{session_id}/clarify", response_model=RuntimePlanResponse)
def clarify_episode(session_id: str, request: RuntimeClarifyRequest):
    service = get_runtime_service()
    _get_session(service, session_id)
    try:
        session = service.clarify_episode(session_id, request.answers)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Clarification failed: {exc}") from exc
    return RuntimePlanResponse(session=_session_view(session), plan=_plan_view(session))


@router.post("/episodes/{session_id}/candidates", response_model=RuntimeCandidatesResponse)
def generate_candidates(session_id: str, request: RuntimeCandidatesRequest):
    service = get_runtime_service()
    _get_session(service, session_id)
    try:
        session = service.generate_initial_candidates(
            session_id=session_id,
            refresh=request.refresh,
            per_query_k=request.per_query_k,
            top_k=request.top_k,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Candidate generation failed: {exc}") from exc

    expansions = [
        RuntimeExpansionView(
            label=getattr(item, "label", "") or "",
            prompt=getattr(item, "prompt", "") or "",
            axis_focus=list(getattr(item, "axis_focus", []) or []),
            target_cluster_id=getattr(item, "target_cluster_id", None),
            checkpoint=getattr(item, "checkpoint", "") or "",
            sampler=getattr(item, "sampler", "") or "",
            loras=list(getattr(item, "loras", []) or []),
        )
        for item in session.latest_expansions
    ]
    return RuntimeCandidatesResponse(
        session=_session_view(session),
        plan=_plan_view(session),
        wall=_wall_view(service, session),
        expansions=expansions,
        recommendation=_recommendation_view(session),
    )


@router.post("/episodes/{session_id}/select", response_model=RuntimeSelectResponse)
def select_reference(session_id: str, request: RuntimeSelectRequest):
    service = get_runtime_service()
    _get_session(service, session_id)
    try:
        session = service.select_initial_reference(session_id, request.gallery_index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RuntimeSelectResponse(
        session=_session_view(session),
        anchor_summary=session.current_gallery_anchor_summary,
        references=list(session.selected_reference_bundle.get("references", []) or []),
        selected_reference_ids=list(session.selected_reference_ids),
    )


@router.post("/episodes/{session_id}/schema", response_model=RuntimeSchemaResponse)
def generate_schema(session_id: str):
    service = get_runtime_service()
    session = _get_session(service, session_id)
    if not session.selected_reference_bundle:
        raise HTTPException(status_code=409, detail="Select a candidate before generating the schema.")
    try:
        session = service.generate_initial_schema(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Schema generation failed: {exc}") from exc
    return RuntimeSchemaResponse(
        session=_session_view(session),
        normalized=_to_serializable(session.current_schema),
        raw_metadata=session.current_schema_raw,
    )


@router.post("/episodes/{session_id}/result", response_model=RuntimeResultResponse)
def produce_result(session_id: str):
    service = get_runtime_service()
    session = _get_session(service, session_id)
    _require_schema(session)
    try:
        session = service.produce_initial_result(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Result production failed: {exc}") from exc
    return RuntimeResultResponse(
        session=_session_view(session),
        payload=_to_serializable(session.current_result_payload),
        summary=_to_serializable(session.current_result_summary),
    )


@router.get("/episodes/{session_id}", response_model=RuntimePlanResponse)
def get_episode(session_id: str):
    service = get_runtime_service()
    session = _get_session(service, session_id)
    return RuntimePlanResponse(session=_session_view(session), plan=_plan_view(session))


# ---------------------------------------------------------------------------
# PBO refinement loop
# ---------------------------------------------------------------------------


def _refinement_view(service, session: AgentSessionState) -> RuntimeRefinementView:
    pending = [
        _candidate_view(service, int(index), slot, 0, "")
        for slot, index in enumerate(session.pbo_current_candidates, start=1)
    ]
    return RuntimeRefinementView(
        active=session.pbo_active,
        finished=session.pbo_finished,
        seed_indices=list(session.pbo_seed_indices),
        round_index=session.pbo_round_index,
        consecutive_skips=session.pbo_consecutive_skips,
        batch_size=session.pbo_batch_size,
        pending_candidates=pending,
        history=[_to_serializable(item) for item in session.pbo_history],
        best_index=session.pbo_best_index,
    )


def _refine_response(service, session: AgentSessionState) -> RuntimeRefineResponse:
    return RuntimeRefineResponse(
        session=_session_view(session),
        refinement=_refinement_view(service, session),
        anchor_summary=session.current_gallery_anchor_summary,
        selected_reference_ids=list(session.selected_reference_ids),
    )


def _refine_guard(service, session_id: str) -> AgentSessionState:
    session = _get_session(service, session_id)
    if session.current_schema_raw:
        raise HTTPException(
            status_code=409,
            detail="A schema has already been generated for this session; start a new episode.",
        )
    return session


@router.post("/episodes/{session_id}/refine/start", response_model=RuntimeRefineResponse)
def start_refinement(session_id: str, request: RuntimeRefineStartRequest):
    service = get_runtime_service()
    _refine_guard(service, session_id)
    try:
        session = service.start_refinement(session_id, request.seed_indices)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _refine_response(service, session)


@router.post("/episodes/{session_id}/refine/round", response_model=RuntimeRefineResponse)
def run_refinement_round(session_id: str, request: RuntimeRefineRoundRequest):
    service = get_runtime_service()
    _refine_guard(service, session_id)
    try:
        session = service.run_refinement_round(session_id, batch_size=request.batch_size)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _refine_response(service, session)


@router.post("/episodes/{session_id}/refine/feedback", response_model=RuntimeRefineResponse)
def submit_refinement_feedback(session_id: str, request: RuntimeRefineFeedbackRequest):
    service = get_runtime_service()
    _refine_guard(service, session_id)
    try:
        session = service.submit_refinement_feedback(
            session_id,
            best_slot=request.best_slot,
            worst_slot=request.worst_slot,
            skip=request.skip,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _refine_response(service, session)


@router.post("/episodes/{session_id}/refine/finish", response_model=RuntimeRefineResponse)
def finish_refinement(session_id: str):
    service = get_runtime_service()
    _refine_guard(service, session_id)
    try:
        session = service.finish_refinement(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _refine_response(service, session)


@router.get("/episodes/{session_id}/refine", response_model=RuntimeRefineResponse)
def get_refinement(session_id: str):
    service = get_runtime_service()
    session = _get_session(service, session_id)
    return _refine_response(service, session)


# ---------------------------------------------------------------------------
# ComfyUI hand-off
# ---------------------------------------------------------------------------


def _workflow_kwargs(request: RuntimeWorkflowRequest) -> Dict[str, Any]:
    return {
        "width": request.width,
        "height": request.height,
        "batch_size": request.batch_size,
        "seed": request.seed,
        "filename_prefix": request.filename_prefix or "Genflow",
        "checkpoint_override": request.checkpoint_override,
    }


@router.get("/comfyui/status", response_model=ComfyUIStatusResponse)
def comfyui_status():
    payload = get_comfy_service().status()
    payload.setdefault("error", "")
    return ComfyUIStatusResponse(**payload)


@router.post("/episodes/{session_id}/workflow", response_model=RuntimeWorkflowResponse)
def build_workflow(session_id: str, request: RuntimeWorkflowRequest):
    service = get_runtime_service()
    session = _get_session(service, session_id)
    _require_schema(session)
    try:
        built = get_comfy_service().build_workflow(session.current_schema, **_workflow_kwargs(request))
    except ComfyUIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return RuntimeWorkflowResponse(session=_session_view(session), **built)


@router.post("/episodes/{session_id}/workflow/push", response_model=RuntimePushResponse)
def push_workflow(session_id: str, request: RuntimeWorkflowRequest):
    service = get_runtime_service()
    session = _get_session(service, session_id)
    _require_schema(session)
    payload = get_comfy_service().push_workflow(session.current_schema, **_workflow_kwargs(request))
    return RuntimePushResponse(session=_session_view(session), **payload)


@router.get("/workflow/result/{prompt_id}")
def workflow_result(prompt_id: str):
    return get_comfy_service().prompt_result(prompt_id)


# ---------------------------------------------------------------------------
# gallery images
# ---------------------------------------------------------------------------


def _detect_image_media_type(path: Path) -> str:
    """Kept for backwards compatibility; delegates to the shared catalog."""
    return gallery_catalog.media_type(path)


@router.get("/gallery/image/{gallery_index}")
def gallery_image(gallery_index: int):
    """Serve a gallery thumbnail without initialising the embedding stack."""
    try:
        path = gallery_catalog.image_path(gallery_index)
    except gallery_catalog.GalleryImageNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, media_type=gallery_catalog.media_type(path))


# ---------------------------------------------------------------------------
# refine showcase
# ---------------------------------------------------------------------------


@router.post("/showcase/episode", response_model=RuntimeShowcaseResponse)
def start_showcase_episode(request: RuntimeShowcaseRequest):
    """Fabricate a session over chosen gallery images.

    Lets the preference-search UI be exercised against the real PBO loop without
    running the planner or the retrieval pipeline.
    """
    service = get_runtime_service()
    try:
        session = service.start_showcase_session(
            gallery_indices=request.gallery_indices,
            label=request.label,
            size=request.size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RuntimeShowcaseResponse(
        session=_session_view(session),
        wall=_wall_view(service, session),
    )
