from __future__ import annotations

"""Local PBO reranking for refinement benchmark candidates."""

from typing import TYPE_CHECKING, Iterable, List

if TYPE_CHECKING:
    from app.agent.memory import AgentSessionState
    from app.agent.refinement_benchmark_retriever import RefinementBenchmarkCandidate


def rank_benchmark_candidates(
    candidates: List["RefinementBenchmarkCandidate"],
    session_context: "AgentSessionState",
) -> List["RefinementBenchmarkCandidate"]:
    ranked_items = []
    for index, candidate in enumerate(candidates):
        score, rationale = _score_candidate(candidate, session_context=session_context)
        ranked_items.append(
            (
                -score,
                candidate.candidate_id,
                index,
                _clone_candidate_with_pbo_annotations(candidate, score=score, rationale=rationale),
            )
        )
    ranked_items.sort()
    return [item[-1] for item in ranked_items]


def _score_candidate(
    candidate: "RefinementBenchmarkCandidate",
    session_context: "AgentSessionState",
) -> tuple[float, list[str]]:
    score = 0.0
    rationale: list[str] = []

    dissatisfaction_axes = list(session_context.dissatisfaction_axes)
    preserve_axes = _collect_preserve_axes(session_context)
    current_uncertainty = float(session_context.current_uncertainty_estimate or 0.0)
    role = candidate.source_role or str(candidate.metadata.get("role", "reference"))
    benchmark_source = str(candidate.metadata.get("benchmark_source", ""))
    novelty_score = float(candidate.metadata.get("novelty_score", 0.0) or 0.0)
    coverage_score = float(candidate.metadata.get("coverage_score", 0.0) or 0.0)
    anchor_overlap = bool(candidate.metadata.get("anchor_overlap", False))

    role_focus_bonus = _role_focus_bonus(role)
    if dissatisfaction_axes and role_focus_bonus:
        gain = round(role_focus_bonus * len(dissatisfaction_axes), 2)
        score += gain
        rationale.append(f"+{gain:.2f}:dissatisfaction_role={role}")

    if preserve_axes and anchor_overlap:
        gain = 0.75
        score += gain
        rationale.append(f"+{gain:.2f}:preserve_anchor_overlap")

    if preserve_axes and role == "best":
        gain = 0.5
        score += gain
        rationale.append(f"+{gain:.2f}:preserve_best_role")

    if preserve_axes and role == "exploratory":
        penalty = 0.5
        score -= penalty
        rationale.append(f"-{penalty:.2f}:preserve_exploratory_penalty")

    explicit_preserve_bonus = _explicit_preserve_bonus(role=role, preserve_axes=preserve_axes)
    if explicit_preserve_bonus:
        score += explicit_preserve_bonus
        rationale.append(f"+{explicit_preserve_bonus:.2f}:explicit_preserve_match")

    if novelty_score:
        gain = round(2.0 * novelty_score, 2)
        score += gain
        rationale.append(f"+{gain:.2f}:novelty")

    if coverage_score:
        gain = round(1.5 * coverage_score, 2)
        score += gain
        rationale.append(f"+{gain:.2f}:coverage")

    uncertainty_gain = _uncertainty_bonus(
        current_uncertainty=current_uncertainty,
        anchor_overlap=anchor_overlap,
        novelty_score=novelty_score,
    )
    if uncertainty_gain:
        score += uncertainty_gain
        rationale.append(f"+{uncertainty_gain:.2f}:uncertainty_alignment")

    if session_context.current_result_summary.summary_text and not anchor_overlap:
        gain = 0.5
        score += gain
        rationale.append(f"+{gain:.2f}:current_result_context")

    if session_context.current_schema.model and role in {"exploratory", "complementary_knn"}:
        gain = 0.25
        score += gain
        rationale.append(f"+{gain:.2f}:schema_model_context")

    if benchmark_source == "refinement_search_bundle":
        gain = 0.75
        score += gain
        rationale.append(f"+{gain:.2f}:benchmark_source")
    elif benchmark_source:
        gain = 0.25
        score += gain
        rationale.append(f"+{gain:.2f}:benchmark_source")

    return round(score, 2), rationale


def _collect_preserve_axes(session_context: "AgentSessionState") -> set[str]:
    preserve_axes = set(session_context.preserve_constraints)
    lower_constraints = " ".join(session_context.preserve_constraints).lower()
    for axis in session_context.dissatisfaction_axes:
        if axis.lower() in lower_constraints:
            preserve_axes.add(axis)
    return preserve_axes


def _role_focus_bonus(role: str) -> float:
    if role == "exploratory":
        return 2.5
    if role == "complementary_knn":
        return 1.5
    if role == "best":
        return 0.75
    return 1.0


def _uncertainty_bonus(
    current_uncertainty: float,
    anchor_overlap: bool,
    novelty_score: float,
) -> float:
    if current_uncertainty >= 0.5:
        return round(1.5 * max(0.5, novelty_score), 2)
    if current_uncertainty <= 0.1 and anchor_overlap:
        return 0.75
    return 0.0


def _explicit_preserve_bonus(role: str, preserve_axes: set[str]) -> float:
    lower_preserve = {item.lower() for item in preserve_axes}
    if role and role.lower() in lower_preserve:
        return 3.0
    if "anchor" in lower_preserve and role == "best":
        return 1.5
    return 0.0


def _clone_candidate_with_pbo_annotations(
    candidate: "RefinementBenchmarkCandidate",
    score: float,
    rationale: Iterable[str],
) -> "RefinementBenchmarkCandidate":
    from app.agent.refinement_benchmark_retriever import RefinementBenchmarkCandidate

    metadata = dict(candidate.metadata)
    metadata["pbo_score"] = score
    metadata["pbo_rationale"] = list(rationale)
    return RefinementBenchmarkCandidate(
        candidate_id=candidate.candidate_id,
        reference_id=candidate.reference_id,
        source_index=candidate.source_index,
        source_role=candidate.source_role,
        selection_rationale=candidate.selection_rationale,
        metadata=metadata,
    )
