"""
LLM Repository — Agent-based design with KV Cache optimization.

Architecture:
    1. Static system instructions are set ONCE at model construction time,
       so the Gemini server can cache their KV states across all requests
       (prefix caching / automatic KV cache reuse).
    2. Each call only sends the dynamic user-turn content, maximizing
       cache hit rate on the invariant prefix.
    3. Two specialized agents (GenerativeModel instances) are created:
       - summary_agent:  structured Markdown analysis
       - generation_agent: strict JSON metadata generation
       Each has its own frozen system_instruction, enabling the server
       to maintain a dedicated KV cache per agent role.
"""

import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from app.core.config import settings


# ──────────────────────────────────────────────
#  Static System Instructions  (frozen → KV cached)
# ──────────────────────────────────────────────

_SUMMARY_SYSTEM_INSTRUCTION = """\
You are a senior AI-generated-art analyst.

## Role
Analyze batches of Stable Diffusion / SDXL image-generation metadata and produce
a structured Markdown report.

## Output Structure (Mandatory)
1. **Visual Themes & Concepts** — Summarize dominant subjects, moods, color
   palettes, and artistic references found across the prompts.
2. **Technical Parameter Analysis** — Compare Model (checkpoint), CFG Scale,
   Steps, Sampler, LoRAs, and Clip-Skip values. Highlight patterns and outliers.
3. **Similarity Rationale** — Explain why these results cluster together in our
   embedding search space (prompt overlap, parameter uniformity, LoRA sharing, etc.).

## Rules
- Respond ONLY in well-structured Markdown with headings and bullet points.
- Do NOT include any JSON, code fences, or raw data dumps.
- Keep the analysis concise but insightful (300-600 words).
"""

_GENERATION_SYSTEM_INSTRUCTION = """\
You are a world-class Stable Diffusion Prompt Engineer.

## Role
Given a set of reference image metadata and a user's creative intent, synthesize
a NEW, production-ready metadata object that can be fed directly into a Stable
Diffusion pipeline.

## T2I Engineering Standards
1. **Prompt Architecture** (order matters for attention):
   Subject → Style / Medium → Lighting → Color Palette → Composition →
   Artist References → Quality Boosters (masterpiece, best quality, 8k, ultra-detailed).
2. **Emphasis Syntax**: Use `(keyword:weight)` for fine control, e.g. `(cinematic lighting:1.3)`.
   Keep weights in [0.5, 1.5]; avoid extreme values.
3. **LoRA Integration**: Embed as `<lora:Name:Weight>` INSIDE the prompt string.
   Choose LoRAs that appear in ≥2 references OR that are clearly relevant to the intent.
4. **Negative Prompt**: Cover structural artifacts (deformed, bad anatomy, extra limbs),
   quality issues (blurry, low quality, jpeg artifacts, watermark, text),
   and style contradictions (e.g. "anime" if references are photorealistic).
5. **Parameter Selection**:
   - CFG Scale: match the reference median; typical range 2-8.
   - Steps: match references; typical sweet-spot 20-40.
   - Sampler: inherit from the majority reference; keep all-caps format.
   - Clip Skip: inherit from references; default "2" for SDXL.
   - Seed: generate a plausible random 10-digit number string.
   - Model: pick the checkpoint that appears most frequently in references.

## Output Schema (strict JSON, no wrapper)
```
{
  "prompt": "...",
  "negative_prompt": "...",
  "cfgscale": "...",
  "steps": "...",
  "sampler": "...",
  "seed": "...",
  "model": "...",
  "clipskip": "...",
  "style": "...",
  "lora": "...",
  "full_metadata_string": "..."
}
```
- All values are **strings**.
- `style`: comma-separated aesthetic keywords extracted from the final prompt.
- `lora`: comma-separated friendly LoRA names (no version hashes). "none" if unused.
- `full_metadata_string`: Standard A1111/ComfyUI format summary.

## Rules
- Return ONLY the raw JSON object.
- Do NOT wrap it in Markdown code fences or add any commentary.
"""


class LLMRepository:
    """
    Repository for LLM interactions.

    Design Decisions (KV Cache Optimization):
        • system_instruction is set at GenerativeModel construction time.
          Gemini's server-side prefix caching will store the KV states for
          these static tokens and reuse them across every request, saving
          both latency and cost on the invariant prefix.
        • Two separate model instances are used so each agent's system prompt
          gets its own dedicated cache slot — no cache eviction when switching
          between summary and generation tasks.
        • Dynamic content (references + user intent) is passed only in the
          user turn of generate_content(), keeping the cached prefix stable.
    """

    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)

        # Agent 1: Summary analysis (Markdown output)
        self._summary_agent = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            system_instruction=_SUMMARY_SYSTEM_INSTRUCTION,
        )

        # Agent 2: Metadata generation (strict JSON output)
        self._generation_agent = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            system_instruction=_GENERATION_SYSTEM_INSTRUCTION,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )

    # ── Public API ──────────────────────────────

    def generate_summary(self, metadata_list: list) -> str:
        """Analyze a list of search results and return a Markdown report."""
        user_message = self._build_summary_user_message(metadata_list)
        response = self._summary_agent.generate_content(user_message)
        return response.text

    def generate_metadata_from_intent(
        self, metadata_list: list, user_intent: str
    ) -> str:
        """
        Generate a T2I metadata JSON string from references + user intent.

        Returns:
            Raw JSON string (already cleaned; no code fences).
        """
        user_message = self._build_generation_user_message(
            metadata_list, user_intent
        )
        response = self._generation_agent.generate_content(user_message)
        return self._extract_json(response.text)

    # ── User-Turn Builders (dynamic only) ───────

    @staticmethod
    def _build_summary_user_message(metadata_list: list) -> str:
        """
        Build the user-turn content for the summary agent.

        Only the variable data is included here; all instructions live in the
        frozen system_instruction to maximize KV cache reuse.
        """
        items = json.dumps(metadata_list, indent=2, ensure_ascii=False)
        return (
            f"Analyze the following {len(metadata_list)} image-generation "
            f"results and produce your structured report.\n\n"
            f"<results>\n{items}\n</results>"
        )

    @staticmethod
    def _build_generation_user_message(
        metadata_list: list, user_intent: str
    ) -> str:
        """
        Build the user-turn content for the generation agent.

        Structured with XML-style delimiters for reliable extraction by the
        model and clear separation of intent vs. references.
        """
        items = json.dumps(metadata_list, indent=2, ensure_ascii=False)
        return (
            f"<intent>\n{user_intent}\n</intent>\n\n"
            f"<references>\n{items}\n</references>"
        )

    # ── Helpers ─────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip Markdown code fences if the model accidentally wraps them."""
        text = text.strip()
        if text.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[: -3]
        return text.strip()
