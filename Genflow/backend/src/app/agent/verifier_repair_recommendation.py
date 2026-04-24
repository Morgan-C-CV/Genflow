from __future__ import annotations

"""Deterministic repair recommendation layer on top of verifier signals."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from app.agent.runtime_models import VerifierResult, VerifierSignalSummary

if TYPE_CHECKING:
    from app.agent.memory import AgentSessionState


@dataclass
class VerifierRepairRecommendation:
    recommended_action: str = ""
    rationale: List[str] = field(default_factory=list)
    priority: str = ""
    supporting_signals: List[str] = field(default_factory=list)


def build_verifier_repair_recommendation(
    verifier_signal_summary: VerifierSignalSummary,
    verifier_result: VerifierResult,
    session: "AgentSessionState" | None = None,
) -> VerifierRepairRecommendation:
    preserve_risk = float(verifier_signal_summary.preserve_risk_score or 0.0)
    execution_evidence = float(verifier_signal_summary.execution_evidence_score or 0.0)
    benchmark_support = float(verifier_signal_summary.benchmark_support_score or 0.0)
    rationale: list[str] = []
    supporting_signals: list[str] = []

    if preserve_risk >= 2.0:
        rationale.append("preserve risk is materially elevated")
        supporting_signals.append("high_preserve_risk")
        return VerifierRepairRecommendation(
            recommended_action="reduce_preserve_risk",
            rationale=rationale,
            priority="high",
            supporting_signals=supporting_signals,
        )

    if execution_evidence < 1.0:
        rationale.append("execution evidence is weak for the committed direction")
        supporting_signals.append("weak_execution_evidence")
        if benchmark_support < 0.5 and _session_uncertainty(session) >= 0.4:
            rationale.append("benchmark support is weak under high uncertainty")
            supporting_signals.append("low_benchmark_support")
            return VerifierRepairRecommendation(
                recommended_action="refresh_benchmarks",
                rationale=rationale,
                priority="high",
                supporting_signals=supporting_signals,
            )
        return VerifierRepairRecommendation(
            recommended_action="probe_more",
            rationale=rationale,
            priority="medium",
            supporting_signals=supporting_signals,
        )

    if not verifier_result.continue_recommended:
        rationale.append("verifier accepts the current direction")
        supporting_signals.append("verifier_accepts_current_direction")
        if benchmark_support >= 1.0:
            rationale.append("benchmark support is strong")
            supporting_signals.append("strong_benchmark_support")
        return VerifierRepairRecommendation(
            recommended_action="stop",
            rationale=rationale,
            priority="low",
            supporting_signals=supporting_signals,
        )

    rationale.append("verifier suggests continuing current direction")
    supporting_signals.append("continue_current_direction")
    return VerifierRepairRecommendation(
        recommended_action="continue_current_direction",
        rationale=rationale,
        priority="medium",
        supporting_signals=supporting_signals,
    )


def _session_uncertainty(session: "AgentSessionState" | None) -> float:
    if session is None:
        return 0.0
    return float(session.current_uncertainty_estimate or 0.0)
