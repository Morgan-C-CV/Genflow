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

    if session.stop_reason and session.stop_reason != "verifier_accepts_current_direction":
        rationale.append(f"stop_reason={session.stop_reason}")
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

    if session.accepted_patch.patch_id and not _has_executed_accepted_patch(session):
        rationale.append(f"accepted_patch_needs_execution={session.accepted_patch.patch_id}")
        return PolicyDecision(next_action="execute_patch", rationale=rationale, continue_loop=True)

    if session.accepted_patch.patch_id and _has_executed_accepted_patch(session) and not session.latest_verifier_result.summary:
        rationale.append(f"accepted_patch_needs_verification={session.accepted_patch.patch_id}")
        return PolicyDecision(next_action="verify_latest_result", rationale=rationale, continue_loop=True)

    verifier_decision = _decide_post_verifier_action(session)
    if verifier_decision is not None:
        return verifier_decision

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


def _has_executed_accepted_patch(session: "AgentSessionState") -> bool:
    if not session.accepted_patch.patch_id:
        return False
    if session.previous_result_summary.summary_text:
        return True
    return False


def _decide_post_verifier_action(session: "AgentSessionState") -> PolicyDecision | None:
    if not session.latest_verifier_result.summary:
        return None

    recommendation_decision = _decide_from_verifier_recommendation(session)
    if recommendation_decision is not None:
        return recommendation_decision

    return _decide_from_verifier_signals(session)


def _decide_from_verifier_recommendation(session: "AgentSessionState") -> PolicyDecision | None:
    recommendation = session.latest_verifier_repair_recommendation.recommended_action
    if not recommendation:
        return None

    rationale = [f"verifier_repair_recommendation={recommendation}"]
    if recommendation == "probe_more":
        return PolicyDecision(next_action="generate_probes", rationale=rationale, continue_loop=True)
    if recommendation == "refresh_benchmarks":
        return PolicyDecision(next_action="retrieve_benchmarks", rationale=rationale, continue_loop=True)
    if recommendation == "reduce_preserve_risk":
        return PolicyDecision(next_action="generate_probes", rationale=rationale, continue_loop=True)
    if recommendation == "continue_current_direction":
        rationale.append("continue_current_direction")
        return PolicyDecision(next_action="generate_probes", rationale=rationale, continue_loop=True)
    if recommendation == "stop":
        return PolicyDecision(next_action="stop", rationale=rationale, continue_loop=False)
    return None


def _decide_from_verifier_signals(session: "AgentSessionState") -> PolicyDecision:
    rationale: list[str] = []
    signal_summary = session.latest_verifier_signal_summary
    preserve_risk = float(signal_summary.preserve_risk_score or 0.0)
    execution_evidence = float(signal_summary.execution_evidence_score or 0.0)
    benchmark_support = float(signal_summary.benchmark_support_score or 0.0)
    uncertainty = float(session.current_uncertainty_estimate or 0.0)

    if preserve_risk >= 2.0:
        rationale.extend(["high_preserve_risk", "verifier_requests_direction_change"])
        return PolicyDecision(next_action="generate_probes", rationale=rationale, continue_loop=True)

    if execution_evidence < 1.0:
        rationale.append("weak_execution_evidence")
        if benchmark_support < 0.5 and uncertainty >= 0.4:
            rationale.append("low_benchmark_support")
            return PolicyDecision(next_action="retrieve_benchmarks", rationale=rationale, continue_loop=True)
        rationale.append("continue_probing")
        return PolicyDecision(next_action="generate_probes", rationale=rationale, continue_loop=True)

    if not session.continue_recommended:
        rationale.append("verifier_recommends_stop")
        if benchmark_support >= 1.0:
            rationale.append("strong_benchmark_support")
        if preserve_risk < 1.0:
            rationale.append("low_preserve_risk")
        return PolicyDecision(next_action="stop", rationale=rationale, continue_loop=False)

    rationale.append("verifier_requests_continue")
    if benchmark_support < 0.5 and uncertainty >= 0.4:
        rationale.append("low_benchmark_support")
        return PolicyDecision(next_action="retrieve_benchmarks", rationale=rationale, continue_loop=True)
    return PolicyDecision(next_action="generate_probes", rationale=rationale, continue_loop=True)
