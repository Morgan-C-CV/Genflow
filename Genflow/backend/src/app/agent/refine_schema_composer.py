"""Compose one workflow schema from the references the user picked.

Once the user has chosen a gallery reference for each modification axis, this
asks the model to unify them into a single normalized schema. That schema is the
proposal the preview shows; committing it is what actually changes the session's
committed state.

If the model is unavailable the composer returns ``None`` and the caller falls
back to the deterministic patch planner.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from app.agent.axis_reference_retriever import AxisReference
from app.agent.runtime_models import NormalizedSchema, ParsedFeedbackEvidence
from app.core.config import settings
from app.core.llm_client import build_llm_model


_LORA_TAG = re.compile(r"<lora:([^:>]+)(?::([\d.]+))?>", re.IGNORECASE)

_SYSTEM_INSTRUCTION = """\
You are a world-class Stable Diffusion prompt engineer.

A user gave feedback on a generated image and picked, for each axis they were
unhappy with, one reference image from a gallery that moves in the direction they
asked for. Unify those references with the image they were reacting to and return
ONE new, production-ready metadata object.

Rules:
1. Keep what the user asked to preserve. Only move the axes they complained about;
   leave every other axis as the current schema has it.
2. Write the positive prompt in attention order:
   subject -> style/medium -> lighting -> colour palette -> composition ->
   quality boosters. Use `(term:weight)` sparingly, weights within [0.5, 1.5].
3. A reference in the "far" band is a stronger move than one in the "near" band.
   Let nearer references keep more of the current image's character.
4. Embed LoRAs as `<lora:Name:Weight>` inside the prompt string. Only use LoRAs
   that appear in the reference records.
5. The negative prompt must cover structural artifacts, quality problems and
   style contradictions with what the user asked for.
6. Parameters: inherit the current schema's sampler, steps, cfg and clip skip
   unless a reference clearly demands otherwise. CFG stays in 2-8, steps in 20-40.
   Pick the checkpoint that best matches the requested direction. Give a new
   plausible 10-digit seed.
7. Never copy subject nouns or character identities from a reference that does
   not fit the user's intent.

Return strict JSON with exactly these keys:
{"prompt": str, "negative_prompt": str, "cfgscale": str, "steps": str,
 "sampler": str, "seed": str, "model": str, "clipskip": str,
 "style": [str], "lora": [str]}
"""


class RefineSchemaComposer:
    def __init__(self, llm_model: Any = None, enabled: bool = True):
        self.enabled = enabled
        self.last_error = ""
        self._agent = llm_model
        if self._agent is None and enabled:
            try:
                self._agent = build_llm_model(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0.6,
                    timeout=settings.REFINE_COMPOSITION_TIMEOUT,
                )
            except Exception:
                self._agent = None

    def compose(
        self,
        *,
        current_schema: NormalizedSchema,
        evidence: ParsedFeedbackEvidence,
        selected_references: Sequence[AxisReference],
    ) -> Optional[NormalizedSchema]:
        self.last_error = ""
        if self._agent is None:
            self.last_error = "composer_unavailable"
            return None
        if not selected_references:
            self.last_error = "no_selected_references"
            return None
        try:
            response = self._agent.generate_content(
                self._build_user_message(current_schema, evidence, selected_references)
            )
            raw = getattr(response, "text", None)
            if not raw:
                self.last_error = "empty_response"
                return None
            payload = json.loads(_extract_json(raw))
        except Exception as exc:
            # Never raise into the loop, but keep the reason so a silent fallback
            # to the rule path can be diagnosed.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        schema = self._to_schema(payload, current_schema)
        if schema is None:
            self.last_error = "unusable_payload"
        return schema

    # -- prompt ---------------------------------------------------------

    @staticmethod
    def _build_user_message(
        current_schema: NormalizedSchema,
        evidence: ParsedFeedbackEvidence,
        selected_references: Sequence[AxisReference],
    ) -> str:
        sections: List[str] = []

        sections.append(
            "The image the user is reacting to (current schema):\n"
            + json.dumps(
                {
                    "prompt": current_schema.prompt,
                    "negative_prompt": current_schema.negative_prompt,
                    "cfgscale": current_schema.cfgscale,
                    "steps": current_schema.steps,
                    "sampler": current_schema.sampler,
                    "seed": current_schema.seed,
                    "model": current_schema.model,
                    "clipskip": current_schema.clipskip,
                    "style": list(current_schema.style),
                    "lora": list(current_schema.lora),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        sections.append(f"Their feedback:\n{evidence.raw_feedback}")

        if evidence.dissatisfaction_scope:
            sections.append(
                "Axes they are unhappy with: " + ", ".join(evidence.dissatisfaction_scope)
            )
        if evidence.preserve_constraints:
            sections.append(
                "They asked to preserve:\n- " + "\n- ".join(evidence.preserve_constraints)
            )
        if evidence.requested_changes:
            sections.append(
                "They asked to change:\n- " + "\n- ".join(evidence.requested_changes)
            )

        lines: List[str] = []
        for reference in selected_references:
            lines.append(
                json.dumps(
                    {
                        "axis": reference.axis,
                        "band": reference.band,
                        "prompt": reference.prompt,
                        "model": reference.model,
                        "sampler": reference.sampler,
                        "cfgscale": reference.cfgscale,
                        "steps": reference.steps,
                        "clipskip": reference.clipskip,
                        "loras": reference.loras,
                    },
                    ensure_ascii=False,
                )
            )
        sections.append(
            "References the user selected, one per axis:\n" + "\n".join(lines)
        )

        return "\n\n".join(sections)

    # -- validation -----------------------------------------------------

    def _to_schema(self, payload: Dict[str, Any], current: NormalizedSchema) -> Optional[NormalizedSchema]:
        if not isinstance(payload, dict):
            return None
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            return None

        loras = _as_text_list(payload.get("lora")) or [
            match.group(1) for match in _LORA_TAG.finditer(prompt)
        ]

        return NormalizedSchema(
            prompt=prompt,
            negative_prompt=str(payload.get("negative_prompt", current.negative_prompt)).strip(),
            cfgscale=_number_string(payload.get("cfgscale"), current.cfgscale),
            steps=_number_string(payload.get("steps"), current.steps),
            sampler=str(payload.get("sampler") or current.sampler).upper().strip(),
            seed=str(payload.get("seed") or current.seed).strip(),
            model=str(payload.get("model") or current.model).strip(),
            clipskip=_number_string(payload.get("clipskip"), current.clipskip),
            style=_as_text_list(payload.get("style")) or list(current.style),
            lora=loras,
            full_metadata_string=json.dumps(payload, ensure_ascii=False),
            raw_fields={key: str(value) for key, value in payload.items() if not isinstance(value, (list, dict))},
        )


def _as_text_list(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",")]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _number_string(value: Any, default: str) -> str:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()
