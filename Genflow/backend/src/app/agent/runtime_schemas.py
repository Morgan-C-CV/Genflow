"""Request/response views for the interactive runtime API.

Deeply nested agent artifacts (result payloads, reference bundles, generated
workflow documents) are exposed as free-form mappings because their shape is
owned by the agent dataclasses and the ComfyUI graph builder respectively.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RuntimePlanView(BaseModel):
    user_intent: str = ""
    fixed_constraints: Dict[str, str] = Field(default_factory=dict)
    free_variables: List[str] = Field(default_factory=list)
    locked_axes: List[str] = Field(default_factory=list)
    unclear_axes: List[str] = Field(default_factory=list)
    next_action: str = ""
    clarification_questions: List[str] = Field(default_factory=list)
    reasoning_summary: str = ""


class RuntimeSessionView(BaseModel):
    session_id: str
    original_intent: str
    clarified_intent: str
    clarification_closed: bool = False
    clarification_rounds: int = 0
    selected_gallery_index: Optional[int] = None
    next_action: str = ""


class RuntimeCandidateView(BaseModel):
    slot: int
    gallery_index: int
    id: str = ""
    image_url: str = ""
    local_path: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    model: str = ""
    sampler: str = ""
    steps: str = ""
    cfgscale: str = ""
    seed: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    group_index: int = 0
    group_label: str = ""
    distance: Optional[float] = None


class RuntimeWallView(BaseModel):
    groups: List[List[int]] = Field(default_factory=list)
    query_labels: List[str] = Field(default_factory=list)
    candidates: List[RuntimeCandidateView] = Field(default_factory=list)
    description: str = ""


class RuntimeExpansionView(BaseModel):
    label: str = ""
    prompt: str = ""
    axis_focus: List[str] = Field(default_factory=list)
    target_cluster_id: Optional[int] = None
    checkpoint: str = ""
    sampler: str = ""
    loras: List[str] = Field(default_factory=list)


class RuntimeRecommendationView(BaseModel):
    checkpoint: str = ""
    sampler: str = ""
    loras: List[str] = Field(default_factory=list)
    reasoning_summary: str = ""


class RuntimeStartRequest(BaseModel):
    user_intent: str = Field(..., min_length=1)


class RuntimeClarifyRequest(BaseModel):
    answers: List[str] = Field(default_factory=list)


class RuntimeCandidatesRequest(BaseModel):
    refresh: bool = False
    per_query_k: int = Field(default=2, ge=1, le=4)
    top_k: int = Field(default=12, ge=4, le=32)


class RuntimeSelectRequest(BaseModel):
    gallery_index: int = Field(..., ge=0)


class RuntimeWorkflowRequest(BaseModel):
    width: int = Field(default=1024, ge=64, le=8192)
    height: int = Field(default=1024, ge=64, le=8192)
    batch_size: int = Field(default=1, ge=1, le=16)
    seed: Optional[int] = Field(default=None, ge=0)
    filename_prefix: str = "Genflow"


class RuntimeStartResponse(BaseModel):
    session: RuntimeSessionView
    plan: RuntimePlanView


class RuntimePlanResponse(BaseModel):
    session: RuntimeSessionView
    plan: RuntimePlanView


class RuntimeCandidatesResponse(BaseModel):
    session: RuntimeSessionView
    plan: RuntimePlanView
    wall: RuntimeWallView
    expansions: List[RuntimeExpansionView] = Field(default_factory=list)
    recommendation: RuntimeRecommendationView


class RuntimeSelectResponse(BaseModel):
    session: RuntimeSessionView
    anchor_summary: str = ""
    references: List[Dict[str, Any]] = Field(default_factory=list)
    selected_reference_ids: List[int] = Field(default_factory=list)


class RuntimeSchemaResponse(BaseModel):
    session: RuntimeSessionView
    normalized: Dict[str, Any] = Field(default_factory=dict)
    raw_metadata: str = ""


class RuntimeResultResponse(BaseModel):
    session: RuntimeSessionView
    payload: Dict[str, Any] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)


class RuntimeWorkflowResponse(BaseModel):
    session: RuntimeSessionView
    title: str = "Genflow Workflow"
    api_graph: Dict[str, Any] = Field(default_factory=dict)
    ui_workflow: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    checkpoint: str = ""
    checkpoint_resolved: bool = False
    applied_loras: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_loras: List[str] = Field(default_factory=list)
    controls: Dict[str, Any] = Field(default_factory=dict)
    available_checkpoints: List[str] = Field(default_factory=list)


class RuntimePushResponse(RuntimeWorkflowResponse):
    pushed: bool = False
    prompt_id: str = ""
    queue_number: Optional[int] = None
    error: str = ""
    node_errors: Dict[str, Any] = Field(default_factory=dict)


class ComfyUIStatusResponse(BaseModel):
    reachable: bool = False
    base_url: str = ""
    comfyui_version: str = ""
    device: str = ""
    checkpoints: List[str] = Field(default_factory=list)
    loras: List[str] = Field(default_factory=list)
    samplers: List[str] = Field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# PBO refinement loop
# ---------------------------------------------------------------------------


class RuntimeRefinementView(BaseModel):
    active: bool = False
    finished: bool = False
    seed_indices: List[int] = Field(default_factory=list)
    round_index: int = 0
    consecutive_skips: int = 0
    batch_size: int = 6
    pending_candidates: List[RuntimeCandidateView] = Field(default_factory=list)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    best_index: Optional[int] = None


class RuntimeRefineStartRequest(BaseModel):
    seed_indices: List[int] = Field(..., min_length=1)


class RuntimeRefineRoundRequest(BaseModel):
    batch_size: int = Field(default=6, ge=2, le=12)


class RuntimeRefineFeedbackRequest(BaseModel):
    best_slot: Optional[int] = Field(default=None, ge=1)
    worst_slot: Optional[int] = Field(default=None, ge=1)
    skip: bool = False


class RuntimeRefineResponse(BaseModel):
    session: RuntimeSessionView
    refinement: RuntimeRefinementView
    anchor_summary: str = ""
    selected_reference_ids: List[int] = Field(default_factory=list)
