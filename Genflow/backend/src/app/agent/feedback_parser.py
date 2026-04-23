from __future__ import annotations

import re
from typing import Dict, List, Optional

from app.agent.runtime_models import ParsedFeedbackEvidence


AXIS_KEYWORDS: Dict[str, List[str]] = {
    "style": ["style", "stylish", "风格", "画风"],
    "composition": ["composition", "layout", "构图", "布局"],
    "color_palette": ["color", "colour", "palette", "颜色", "色彩", "色调"],
    "lighting_vibe": ["lighting", "light", "mood", "vibe", "氛围", "光", "打光"],
    "background_setting": ["background", "scene", "setting", "背景", "场景", "环境"],
    "subject": ["subject", "character", "face", "identity", "主体", "人物", "脸"],
}

PRESERVE_PREFIXES = [
    "keep ",
    "preserve ",
    "maintain ",
    "don't change ",
    "do not change ",
    "保持",
    "保留",
    "不要改",
]

CHANGE_PREFIXES = [
    "make it ",
    "make the ",
    "make ",
    "more ",
    "less ",
    "change ",
    "turn it ",
    "让它",
    "更",
    "改得",
]

AMBIGUOUS_MARKERS = [
    "better",
    "different",
    "improve",
    "not right",
    "不好",
    "不太对",
    "改一下",
]


class FeedbackParser:
    def __init__(self, axis_keywords: Optional[Dict[str, List[str]]] = None):
        self.axis_keywords = axis_keywords or AXIS_KEYWORDS

    def parse(
        self,
        feedback_text: str,
        current_result_summary: str = "",
        current_schema_prompt: str = "",
    ) -> ParsedFeedbackEvidence:
        text = (feedback_text or "").strip()
        lowered = text.lower()

        dissatisfaction_scope = self._extract_axes(text, lowered)
        preserve_constraints = self._extract_preserve_constraints(text)
        requested_changes = self._extract_requested_changes(text)
        uncertainty_estimate = self._estimate_uncertainty(
            text=text,
            dissatisfaction_scope=dissatisfaction_scope,
            requested_changes=requested_changes,
        )

        parser_notes = []
        if current_result_summary:
            parser_notes.append("current_result_summary_available")
        if current_schema_prompt:
            parser_notes.append("current_schema_prompt_available")
        if not dissatisfaction_scope:
            parser_notes.append("no_explicit_axis_detected")
        if not requested_changes:
            parser_notes.append("no_explicit_change_phrase_detected")

        return ParsedFeedbackEvidence(
            dissatisfaction_scope=dissatisfaction_scope,
            preserve_constraints=preserve_constraints,
            requested_changes=requested_changes,
            uncertainty_estimate=uncertainty_estimate,
            raw_feedback=text,
            parser_notes=parser_notes,
        )

    def _extract_axes(self, text: str, lowered: str) -> List[str]:
        axes = []
        for axis, keywords in self.axis_keywords.items():
            if any(keyword in lowered or keyword in text for keyword in keywords):
                axes.append(axis)
        return axes

    @staticmethod
    def _extract_preserve_constraints(text: str) -> List[str]:
        constraints: List[str] = []
        for raw_line in re.split(r"[,.，。;；\n]+", text):
            clause = raw_line.strip()
            if not clause:
                continue
            lower_clause = clause.lower()
            if any(lower_clause.startswith(prefix) for prefix in PRESERVE_PREFIXES[:5]) or any(
                clause.startswith(prefix) for prefix in PRESERVE_PREFIXES[5:]
            ):
                constraints.append(clause)
                continue
            if "okay" in lower_clause or "is okay" in lower_clause or "可以" in clause or "还行" in clause:
                constraints.append(clause)
        return constraints

    @staticmethod
    def _extract_requested_changes(text: str) -> List[str]:
        changes: List[str] = []
        for raw_line in re.split(r"[,.，。;；\n]+", text):
            clause = raw_line.strip()
            if not clause:
                continue
            lower_clause = clause.lower()
            if any(marker in lower_clause for marker in CHANGE_PREFIXES[:7]) or any(
                marker in clause for marker in CHANGE_PREFIXES[7:]
            ):
                changes.append(clause)
        return changes

    @staticmethod
    def _estimate_uncertainty(
        text: str,
        dissatisfaction_scope: List[str],
        requested_changes: List[str],
    ) -> float:
        if not text:
            return 1.0
        score = 0.2
        if not dissatisfaction_scope:
            score += 0.35
        if not requested_changes:
            score += 0.25
        lowered = text.lower()
        if any(marker in lowered or marker in text for marker in AMBIGUOUS_MARKERS):
            score += 0.15
        if len(text) < 18:
            score += 0.1
        return min(1.0, round(score, 2))
