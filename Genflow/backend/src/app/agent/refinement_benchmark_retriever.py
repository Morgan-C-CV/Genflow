from __future__ import annotations

"""Refinement-time local benchmark retrieval.

This module builds a small comparison set for refinement steps after feedback
exists. It is intentionally separate from the cold-start reference bundle so
repair-time comparisons can evolve without reusing the original bundle shape as
the benchmark data model.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from app.agent.pbo_benchmark_ranker import rank_benchmark_candidates

if TYPE_CHECKING:
    from app.agent.memory import AgentSessionState


@dataclass
class RefinementBenchmarkCandidate:
    candidate_id: str = ""
    reference_id: int | None = None
    source_index: int | None = None
    source_role: str = ""
    selection_rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementBenchmarkSet:
    benchmark_id: str = ""
    benchmark_kind: str = ""
    benchmark_source: str = ""
    anchor_ids: List[int] = field(default_factory=list)
    anchor_summary: str = ""
    comparison_candidates: List[RefinementBenchmarkCandidate] = field(default_factory=list)
    selection_rationale: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def retrieve_refinement_benchmark_set(
    session: AgentSessionState,
    search_service,
    limit: int = 3,
    pbo_benchmark_ranker=rank_benchmark_candidates,
) -> RefinementBenchmarkSet:
    limit = max(1, int(limit))
    benchmark_source, candidate_pool = build_benchmark_candidate_pool(session, search_service=search_service)
    ranked_candidates = pbo_benchmark_ranker(candidate_pool, session)
    anchor_ids = list(session.selected_reference_ids)
    if not anchor_ids:
        anchor_ids = _extract_reference_ids([{"id": candidate.reference_id} for candidate in ranked_candidates[:1]])

    rationale_parts = _build_selection_rationale(session, ranked_candidates)
    return RefinementBenchmarkSet(
        benchmark_id=f"refinement-benchmark-{session.session_id}",
        benchmark_kind="refinement_local_comparison",
        benchmark_source=benchmark_source,
        anchor_ids=anchor_ids,
        anchor_summary=session.current_gallery_anchor_summary
        or f"Refinement anchor for gallery index {session.selected_gallery_index}.",
        comparison_candidates=ranked_candidates[:limit],
        selection_rationale=rationale_parts,
        metadata={
            "selected_gallery_index": session.selected_gallery_index,
            "selected_probe_id": session.selected_probe.probe_id,
            "accepted_patch_id": session.accepted_patch.patch_id,
            "feedback_present": bool(session.latest_feedback),
            "dissatisfaction_axes": list(session.dissatisfaction_axes),
            "preserve_constraints": list(session.preserve_constraints),
            "current_uncertainty_estimate": session.current_uncertainty_estimate,
            "candidate_limit": limit,
            "candidate_pool_size": len(candidate_pool),
        },
    )


def build_benchmark_candidate_pool(
    session: AgentSessionState,
    search_service,
) -> tuple[str, list[RefinementBenchmarkCandidate]]:
    local_bundle = {}
    if session.selected_gallery_index is not None and search_service is not None:
        local_bundle = search_service.build_diverse_reference_bundle(session.selected_gallery_index) or {}

    bundle_references = list(local_bundle.get("references", []))
    if not bundle_references:
        bundle_references = list(session.selected_reference_bundle.get("references", []))

    anchor_ids = list(session.selected_reference_ids)
    benchmark_source = "refinement_search_bundle" if local_bundle else "selected_reference_context"
    candidate_pool: list[RefinementBenchmarkCandidate] = []
    for item in bundle_references:
        reference_id = _coerce_reference_id(item)
        if reference_id is None:
            continue
        anchor_overlap = reference_id in anchor_ids
        novelty_score = 0.0 if anchor_overlap else 1.0
        coverage_score = _compute_coverage_score(
            session.selected_gallery_index,
            _coerce_int(item.get("index")),
        )
        candidate_pool.append(
            RefinementBenchmarkCandidate(
                candidate_id=f"benchmark-candidate-{reference_id}",
                reference_id=reference_id,
                source_index=_coerce_int(item.get("index")),
                source_role=str(item.get("role", "")),
                selection_rationale=_build_candidate_rationale(session, item),
                metadata={
                    "anchor_overlap": anchor_overlap,
                    "novelty_score": novelty_score,
                    "coverage_score": coverage_score,
                    "role": str(item.get("role", "")),
                    "current_result_id": session.current_result_id,
                    "schema_model": session.current_schema.model,
                    "benchmark_source": benchmark_source,
                },
            )
        )
    return benchmark_source, candidate_pool


def _build_selection_rationale(session: AgentSessionState, ranked_candidates: List[RefinementBenchmarkCandidate]) -> list[str]:
    rationale = []
    if session.selected_gallery_index is not None:
        rationale.append(f"anchor_gallery_index={session.selected_gallery_index}")
    if session.selected_reference_ids:
        rationale.append(f"anchor_reference_count={len(session.selected_reference_ids)}")
    if session.dissatisfaction_axes:
        rationale.append(f"focus_axes={','.join(session.dissatisfaction_axes)}")
    if session.preserve_constraints:
        rationale.append(f"preserve={','.join(session.preserve_constraints)}")
    if session.current_result_summary.summary_text:
        rationale.append("current_result_summary_available=true")
    rationale.append(f"candidate_pool_size={len(ranked_candidates)}")
    if ranked_candidates:
        rationale.append(f"top_candidate={ranked_candidates[0].candidate_id}")
    return rationale


def _build_candidate_rationale(session: AgentSessionState, item: Dict[str, Any]) -> str:
    role = str(item.get("role", "reference"))
    reference_id = _coerce_reference_id(item)
    focus_axes = ",".join(session.dissatisfaction_axes) if session.dissatisfaction_axes else "general_quality"
    preserve = ",".join(session.preserve_constraints) if session.preserve_constraints else "none"
    return (
        f"selected role={role} ref={reference_id} "
        f"for focus={focus_axes} while preserving={preserve}"
    )


def _extract_reference_ids(references: List[Dict[str, Any]]) -> list[int]:
    ids = []
    for item in references:
        reference_id = _coerce_reference_id(item)
        if reference_id is not None:
            ids.append(reference_id)
    return ids


def _coerce_reference_id(item: Dict[str, Any]) -> int | None:
    return _coerce_int(item.get("id"))


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compute_coverage_score(
    selected_gallery_index: int | None,
    source_index: int | None,
) -> float:
    if selected_gallery_index is None or source_index is None:
        return 0.0
    distance = abs(selected_gallery_index - source_index)
    return round(max(0.0, 1.0 - 0.25 * distance), 2)
