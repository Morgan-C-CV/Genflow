"""Read refinement feedback into modification axes, with an LLM.

The rule-based parser in ``feedback_parser.py`` matches keywords, so feedback
like "it looks flat and washed out" contributes nothing at all. This module puts
an interpretation model on the primary path: it reads the sentence, decides which
of the six axes the user is dissatisfied with, records what must be preserved,
and writes one short retrieval query per axis. Those queries are what the gallery
retriever later projects into the PBO space.

The keyword parser stays as the fallback: it is deterministic, needs no network,
and keeps the unit tests runnable offline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.agent.feedback_parser import FeedbackParser
from app.agent.runtime_models import NormalizedSchema, ParsedFeedbackEvidence
from app.core.config import settings
from app.core.llm_client import build_llm_model


AXES = (
    "subject",
    "style",
    "composition",
    "lighting_vibe",
    "background_setting",
    "color_palette",
)

MAX_AXES = 4

# Used to build a retrieval query when only the axis name is known.
AXIS_QUERY_HINTS: Dict[str, str] = {
    "subject": "clear, well defined single subject",
    "style": "distinct cohesive visual style",
    "composition": "strong composition and deliberate framing",
    "lighting_vibe": "dramatic cinematic lighting and mood",
    "background_setting": "rich detailed background setting",
    "color_palette": "coherent, vivid colour palette",
}

_SYSTEM_INSTRUCTION = """\
You are the refinement analyst inside an image-generation agent.

A user has seen a generated result and written feedback about it. Read that
feedback and report, as strict JSON:

- "dissatisfaction_axes": the axes the user is unhappy with. Choose ONLY from
  ["subject", "style", "composition", "lighting_vibe", "background_setting",
  "color_palette"]. Judge meaning, not keywords: "flat", "washed out", "muddy"
  and "lifeless" are all lighting_vibe or color_palette complaints even though
  they name no axis. Return at most 4, most important first. Never return an
  empty list unless the feedback is genuinely unreadable.
- "preserve_constraints": short phrases naming what the user wants kept. Include
  anything phrased as "keep", "don't change", "still", "the same", or explicitly
  praised. Use the user's own words.
- "requested_changes": short imperative phrases naming what should change.
- "axis_queries": one object mapping each axis in "dissatisfaction_axes" to a
  SHORT English image-retrieval query (8-16 words) describing what the user wants
  along that axis. These are embedded and compared against a gallery of image
  prompts, so write them the way a prompt would read - visual nouns and
  adjectives, no instructions, no sentences. Example for lighting_vibe:
  "dramatic high-contrast rim lighting, deep shadows, cinematic mood".
- "uncertainty_estimate": 0.0-1.0, how ambiguous the feedback is.

Respond with JSON only.
"""


class RefineIntentInterpreter:
    def __init__(
        self,
        fallback_parser: Optional[FeedbackParser] = None,
        llm_model: Any = None,
        enabled: bool = True,
    ):
        self.fallback_parser = fallback_parser or FeedbackParser()
        self.enabled = enabled
        self._agent = llm_model
        if self._agent is None and enabled:
            try:
                self._agent = build_llm_model(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0.2,
                    timeout=settings.REFINE_COMPOSITION_TIMEOUT,
                )
            except Exception:
                self._agent = None

    # -- public API -----------------------------------------------------

    def interpret(
        self,
        *,
        feedback_text: str,
        current_schema: Optional[NormalizedSchema] = None,
        current_result_summary: str = "",
    ) -> ParsedFeedbackEvidence:
        if self._agent is not None:
            try:
                evidence = self._interpret_with_llm(feedback_text, current_schema)
                if evidence.dissatisfaction_scope:
                    return evidence
            except Exception as exc:
                fallback = self._interpret_with_rules(
                    feedback_text=feedback_text,
                    current_schema=current_schema,
                    current_result_summary=current_result_summary,
                )
                fallback.parser_notes.append(f"llm_failed:{type(exc).__name__}")
                return fallback

        return self._interpret_with_rules(
            feedback_text=feedback_text,
            current_schema=current_schema,
            current_result_summary=current_result_summary,
        )

    # -- LLM path -------------------------------------------------------

    def _interpret_with_llm(
        self,
        feedback_text: str,
        current_schema: Optional[NormalizedSchema],
    ) -> ParsedFeedbackEvidence:
        response = self._agent.generate_content(
            self._build_user_message(feedback_text, current_schema)
        )
        raw = getattr(response, "text", None)
        if not raw:
            raise RuntimeError("Interpretation model returned no text.")
        payload = json.loads(_extract_json(raw))

        axes = [axis for axis in payload.get("dissatisfaction_axes", []) if axis in AXES][:MAX_AXES]
        queries = {
            axis: str(text).strip()
            for axis, text in dict(payload.get("axis_queries", {})).items()
            if axis in AXES and str(text).strip()
        }
        for axis in axes:
            queries.setdefault(axis, self._default_query(axis, feedback_text))

        uncertainty = payload.get("uncertainty_estimate", 0.3)
        try:
            uncertainty = min(1.0, max(0.0, float(uncertainty)))
        except (TypeError, ValueError):
            uncertainty = 0.3

        return ParsedFeedbackEvidence(
            dissatisfaction_scope=axes,
            preserve_constraints=_as_text_list(payload.get("preserve_constraints")),
            requested_changes=_as_text_list(payload.get("requested_changes")),
            uncertainty_estimate=round(uncertainty, 2),
            raw_feedback=feedback_text.strip(),
            parser_notes=["interpreted_by_llm"],
            axis_queries=queries,
            interpreted_by="llm",
        )

    @staticmethod
    def _build_user_message(
        feedback_text: str,
        current_schema: Optional[NormalizedSchema],
    ) -> str:
        lines = [f"User feedback:\n{feedback_text.strip()}"]
        if current_schema is not None:
            lines.append(
                "The result they are reacting to:\n"
                f"- prompt: {current_schema.prompt}\n"
                f"- negative prompt: {current_schema.negative_prompt}\n"
                f"- model: {current_schema.model}\n"
                f"- sampler: {current_schema.sampler}"
            )
        return "\n\n".join(lines)

    # -- rule path ------------------------------------------------------

    def _interpret_with_rules(
        self,
        *,
        feedback_text: str,
        current_schema: Optional[NormalizedSchema],
        current_result_summary: str,
    ) -> ParsedFeedbackEvidence:
        evidence = self.fallback_parser.parse(
            feedback_text=feedback_text,
            current_result_summary=current_result_summary,
            current_schema_prompt=current_schema.prompt if current_schema else "",
        )
        axes = list(evidence.dissatisfaction_scope) or ["style", "color_palette"]
        evidence.dissatisfaction_scope = axes
        evidence.axis_queries = {
            axis: self._default_query(axis, feedback_text) for axis in axes
        }
        evidence.interpreted_by = "rules"
        evidence.parser_notes.append("interpreted_by_rules")
        return evidence

    @staticmethod
    def _default_query(axis: str, feedback_text: str) -> str:
        hint = AXIS_QUERY_HINTS.get(axis, "refined detail")
        cleaned = " ".join((feedback_text or "").split())
        return f"{hint}, {cleaned}" if cleaned else hint


def _as_text_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()
