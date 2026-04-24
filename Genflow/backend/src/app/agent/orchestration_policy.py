from __future__ import annotations

"""Deterministic conditional orchestration policy for the local repair loop."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from app.agent.memory import AgentSessionState


@dataclass
class PolicyDecision:
    next_action: str = "stop"
    rationale: List[str] = field(default_factory=list)
    continue_loop: bool = False


def decide_next_action(session: "AgentSessionState") -> PolicyDecision:
    rationale: list[str] = []

    if session.stop_reason:
        rationale.append(f"stop_reason={session.stop_reason}")
        return PolicyDecision(next_action="stop", rationale=rationale, continue_loop=False)

    if session.latest_verifier_result.summary and not session.continue_recommended:
        rationale.append("verifier_recommends_stop")
        return PolicyDecision(next_action="stop", rationale=rationale, continue_loop=False)

    if not session.repair_hypotheses:
        rationale.append("repair_hypotheses_missing")
        return PolicyDecision(next_action="build_hypotheses", rationale=rationale, continue_loop=True)

    if _should_retrieve_benchmarks(session):
        rationale.append("refinement_benchmark_refresh_needed")
        return PolicyDecision(next_action="retrieve_benchmarks", rationale=rationale, continue_loop=True)

    if not session.preview_probe_candidates:
        rationale.append("preview_probe_candidates_missing")
        return PolicyDecision(next_action="generate_probes", rationale=rationale, continue_loop=True)

    if session.selected_probe.probe_id and not _has_preview_for_selected_probe(session):
        rationale.append(f"selected_probe_needs_preview={session.selected_probe.probe_id}")
        return PolicyDecision(next_action="preview_selected_probe", rationale=rationale, continue_loop=True)

    if session.selected_probe.probe_id and not session.accepted_patch.patch_id:
        rationale.append(f"selected_probe_ready_for_commit={session.selected_probe.probe_id}")
        return PolicyDecision(next_action="commit_selected_patch", rationale=rationale, continue_loop=True)

    if session.accepted_patch.patch_id and not session.latest_verifier_result.summary:
        rationale.append(f"accepted_patch_needs_verification={session.accepted_patch.patch_id}")
        return PolicyDecision(next_action="verify_latest_result", rationale=rationale, continue_loop=True)

    if session.continue_recommended:
        rationale.append("verifier_requests_continue")
        return PolicyDecision(next_action="generate_probes", rationale=rationale, continue_loop=True)

    rationale.append("no_further_action_available")
    return PolicyDecision(next_action="stop", rationale=rationale, continue_loop=False)


def _should_retrieve_benchmarks(session: "AgentSessionState") -> bool:
    has_benchmark_candidates = bool(session.refinement_benchmark_set.comparison_candidates)
    uncertainty_high = float(session.current_uncertainty_estimate or 0.0) >= 0.4
    return not has_benchmark_candidates and uncertainty_high


def _has_preview_for_selected_probe(session: "AgentSessionState") -> bool:
    selected_probe_id = session.selected_probe.probe_id
    if not selected_probe_id:
        return False
    return any(result.probe_id == selected_probe_id for result in session.preview_probe_results)
