from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import google.generativeai as genai
import numpy as np
from google.generativeai.types import GenerationConfig

from app.core.config import settings
from app.agents.prompts import EXPANSION_SYSTEM_INSTRUCTION, PLANNER_SYSTEM_INSTRUCTION


AXES = [
    "subject",
    "style",
    "composition",
    "lighting_vibe",
    "background_setting",
    "color_palette",
]


@dataclass
class ResourceContext:
    raw_markdown: str
    checkpoints: List[str] = field(default_factory=list)
    loras: List[str] = field(default_factory=list)
    samplers: List[str] = field(default_factory=list)
    vaes: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)

    def to_context_block(self) -> str:
        def block(title: str, values: List[str]) -> str:
            if not values:
                return f"## {title}\n- none"
            return "\n".join([f"## {title}"] + [f"- {value}" for value in values])

        sections = [
            block("Checkpoints", self.checkpoints),
            block("LoRAs", self.loras),
            block("Samplers", self.samplers),
            block("VAE / Auxiliary", self.vaes),
            block("Retrieval Hints", self.hints),
        ]
        return "\n\n".join(sections)


@dataclass
class ExpandedQuery:
    label: str
    prompt: str
    axis_focus: List[str] = field(default_factory=list)
    checkpoint: Optional[str] = None
    sampler: Optional[str] = None
    loras: List[str] = field(default_factory=list)


@dataclass
class CreativeIntentPlan:
    user_intent: str
    fixed_constraints: Dict[str, str]
    free_variables: List[str]
    locked_axes: List[str]
    unclear_axes: List[str]
    next_action: str
    clarification_questions: List[str]
    reasoning_summary: str


@dataclass
class ResourceRecommendation:
    checkpoint: str
    sampler: str
    loras: List[str]
    reasoning_summary: str


@dataclass
class CandidateWall:
    groups: List[List[int]]
    flat_indices: List[int]
    query_labels: List[str]


class CreativeAgent:
    def __init__(self, resources_path: Optional[str] = None, model_name: Optional[str] = None):
        api_key = settings.GOOGLE_API_KEY.strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for CreativeAgent.")

        genai.configure(api_key=api_key)

        self.model_name = model_name or settings.GEMINI_MODEL
        self._planner_model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=PLANNER_SYSTEM_INSTRUCTION,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        self._expander_model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=EXPANSION_SYSTEM_INSTRUCTION,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )

        default_path = Path(__file__).with_name("resources.md")
        self.resources_path = Path(resources_path) if resources_path else default_path
        self._resource_cache: Optional[ResourceContext] = None

    def load_resources(self) -> ResourceContext:
        if self._resource_cache is not None:
            return self._resource_cache

        raw = self.resources_path.read_text(encoding="utf-8")
        self._resource_cache = ResourceContext(
            raw_markdown=raw,
            checkpoints=self._extract_bullets(raw, "## Checkpoints"),
            loras=self._extract_bullets(raw, "## LoRAs"),
            samplers=self._extract_bullets(raw, "## Samplers"),
            vaes=self._extract_bullets(raw, "## VAE / Auxiliary"),
            hints=self._extract_bullets(raw, "## Retrieval Hints"),
        )
        return self._resource_cache

    def analyze_intent(
        self, user_intent: str, clarification_closed: bool = False
    ) -> CreativeIntentPlan:
        resources = self.load_resources()
        payload = {
            "user_intent": user_intent.strip(),
            "clarification_closed": clarification_closed,
            "axes": AXES,
            "resource_inventory": resources.to_context_block(),
        }
        response = self._planner_model.generate_content(self._build_json_payload(payload))
        data = self._parse_json(response.text, "analyze_intent")
        return self._coerce_plan(data, user_intent.strip(), clarification_closed=clarification_closed)

    def recommend_resources(
        self, plan: CreativeIntentPlan, resources: Optional[ResourceContext] = None
    ) -> ResourceRecommendation:
        resources = resources or self.load_resources()
        payload = {
            "user_intent": plan.user_intent,
            "fixed_constraints": plan.fixed_constraints,
            "locked_axes": plan.locked_axes,
            "free_variables": plan.free_variables,
            "resource_inventory": resources.to_context_block(),
        }
        response = self._expander_model.generate_content(self._build_json_payload(payload))
        data = self._parse_json(response.text, "recommend_resources")
        return self._coerce_resource_recommendation(data, resources)

    def build_clarification_prompt(self, plan: CreativeIntentPlan) -> str:
        if not plan.clarification_questions:
            return "当前意图已经足够明确，可以继续检索。"
        locked = ", ".join(
            f"{axis}={value}" for axis, value in plan.fixed_constraints.items()
        ) or "无"
        questions = "\n".join(f"- {question}" for question in plan.clarification_questions[:3])
        return (
            "我已经锁定的内容如下：\n"
            f"{locked}\n\n"
            "还需要你补充下面几个未指定轴：\n"
            f"{questions}"
        )

    def build_axis_expansions(
        self,
        user_intent: str,
        plan: CreativeIntentPlan,
        resources: ResourceContext,
        recommendation: Optional[ResourceRecommendation] = None,
        previous_expansions: Optional[List[ExpandedQuery]] = None,
    ) -> List[ExpandedQuery]:
        recommendation = recommendation or self.recommend_resources(plan, resources)
        payload = {
            "user_intent": user_intent.strip(),
            "fixed_constraints": plan.fixed_constraints,
            "free_variables": plan.free_variables,
            "unclear_axes": plan.unclear_axes,
            "recommendation": {
                "recommended_checkpoint": recommendation.checkpoint,
                "recommended_sampler": recommendation.sampler,
                "recommended_loras": recommendation.loras,
            },
            "resource_inventory": resources.to_context_block(),
            "target_candidate_count": 8,
        }
        if previous_expansions:
            payload["previous_expansions"] = [
                {"label": e.label, "prompt": e.prompt} for e in previous_expansions
            ]

        response = self._expander_model.generate_content(self._build_json_payload(payload))
        data = self._parse_json(response.text, "build_axis_expansions")
        expansions = self._coerce_expansions(data, recommendation)
        if len(expansions) != 8:
            raise ValueError(f"Expected 8 expansions, got {len(expansions)}")
        return expansions

    def describe_wall(self, wall: CandidateWall) -> str:
        lines = ["16 图发散矩阵的 8 个轴向分组："]
        for i, label in enumerate(wall.query_labels, start=1):
            lines.append(f"- Group {i}: {label}")
        return "\n".join(lines)

    def build_candidate_wall(
        self,
        search_engine: Any,
        expansions: Sequence[ExpandedQuery],
        per_query_k: int = 4,
        top_k: int = 12,
    ) -> CandidateWall:
        groups: List[List[int]] = []
        flat_indices: List[int] = []
        seen_indices: set[int] = set()
        query_labels: List[str] = []

        for expansion in expansions:
            query_labels.append(expansion.label)
            candidates = self._rank_gallery_candidates(search_engine, expansion, top_k=top_k)
            group: List[int] = []
            for candidate in candidates:
                idx = candidate["index"]
                if idx in seen_indices:
                    continue
                group.append(idx)
                seen_indices.add(idx)
                flat_indices.append(idx)
                if len(group) == per_query_k:
                    break
            if len(group) < per_query_k:
                for idx in self._fallback_pool(search_engine, seen_indices):
                    if idx in seen_indices:
                        continue
                    group.append(idx)
                    seen_indices.add(idx)
                    flat_indices.append(idx)
                    if len(group) == per_query_k:
                        break
            groups.append(group)

        return CandidateWall(groups=groups, flat_indices=flat_indices, query_labels=query_labels)

    def build_training_labels(
        self,
        pbo_space: np.ndarray,
        selected_indices: Sequence[int],
        wall: CandidateWall,
        selected_score: float = 1.0,
        unselected_score: float = 0.5,
    ) -> Tuple[List[np.ndarray], List[float]]:
        x_train: List[np.ndarray] = []
        y_train: List[float] = []
        selected = set(selected_indices)
        for idx in wall.flat_indices:
            x_train.append(np.asarray(pbo_space[idx]))
            y_train.append(selected_score if idx in selected else unselected_score)
        return x_train, y_train

    def _rank_gallery_candidates(
        self, search_engine: Any, expansion: ExpandedQuery, top_k: int = 12
    ) -> List[Dict[str, Any]]:
        if hasattr(search_engine, "transform_query_to_pbo"):
            try:
                query_vector = search_engine.transform_query_to_pbo(
                    prompt=expansion.prompt,
                    sampler=expansion.sampler,
                    model=expansion.checkpoint,
                )
                return search_engine.search_top_k(
                    query_vector=np.asarray(query_vector).reshape(1, -1),
                    top_k=top_k,
                )
            except Exception:
                pass

        return self._rank_gallery_candidates_text(search_engine, expansion, top_k=top_k)

    def _rank_gallery_candidates_text(
        self, search_engine: Any, expansion: ExpandedQuery, top_k: int = 12
    ) -> List[Dict[str, Any]]:
        df = search_engine.df
        corpus = self._gallery_corpus(df)
        query_counter, query_norm = self._tokenize_to_counter(expansion.prompt)
        scored: List[Tuple[int, float]] = []
        for idx, doc in enumerate(corpus):
            doc_counter, doc_norm = self._tokenize_to_counter(doc)
            score = self._cosine_from_counters(query_counter, query_norm, doc_counter, doc_norm)
            scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        score_map = {idx: score for idx, score in scored}
        results: List[Dict[str, Any]] = []
        for idx, _ in scored[:top_k]:
            row = df.iloc[int(idx)].to_dict()
            row["index"] = int(idx)
            row["distance"] = round(float(1.0 - score_map[int(idx)]), 4)
            results.append(row)
        return results

    def _fallback_pool(self, search_engine: Any, seen_indices: set[int]) -> List[int]:
        if hasattr(search_engine, "df"):
            return [int(i) for i in range(len(search_engine.df)) if i not in seen_indices]
        return []

    @staticmethod
    def _build_json_payload(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_json(text: str, context: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            if text.endswith("```"):
                text = text[:-3]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: model returned invalid JSON: {text[:400]}") from exc

    @staticmethod
    def _coerce_plan(
        data: Dict[str, Any],
        user_intent: str,
        clarification_closed: bool = False,
    ) -> CreativeIntentPlan:
        fixed_constraints = {
            str(k): str(v)
            for k, v in dict(data.get("fixed_constraints", {})).items()
            if str(k) in AXES and str(v).strip()
        }
        locked_axes = [axis for axis in data.get("locked_axes", []) if axis in AXES]
        free_variables = [axis for axis in data.get("free_variables", []) if axis in AXES]
        unclear_axes = [axis for axis in data.get("unclear_axes", []) if axis in AXES]
        clarification_questions = [str(q).strip() for q in data.get("clarification_questions", []) if str(q).strip()]
        next_action = str(data.get("next_action", "ask_user")).strip()
        if next_action not in {"ask_user", "retrieve_resources"}:
            next_action = "ask_user"
        reasoning_summary = str(data.get("reasoning_summary", "")).strip()
        if not reasoning_summary:
            reasoning_summary = "LLM completed intent analysis."
        if clarification_closed:
            next_action = "retrieve_resources"
            clarification_questions = []
        return CreativeIntentPlan(
            user_intent=user_intent,
            fixed_constraints=fixed_constraints,
            free_variables=free_variables,
            locked_axes=locked_axes,
            unclear_axes=unclear_axes,
            next_action=next_action,
            clarification_questions=clarification_questions,
            reasoning_summary=reasoning_summary,
        )

    @staticmethod
    def _coerce_resource_recommendation(
        data: Dict[str, Any], resources: ResourceContext
    ) -> ResourceRecommendation:
        checkpoint = str(data.get("recommended_checkpoint", "")).strip()
        sampler = str(data.get("recommended_sampler", "")).strip()
        loras = [str(item).strip() for item in data.get("recommended_loras", []) if str(item).strip()]
        reasoning_summary = str(data.get("reasoning_summary", "")).strip() or "LLM completed resource selection."

        if checkpoint and resources.checkpoints and checkpoint not in resources.checkpoints:
            checkpoint = resources.checkpoints[0]
        if sampler and resources.samplers and sampler not in resources.samplers:
            sampler = resources.samplers[0]
        if not loras and resources.loras:
            loras = resources.loras[:2]

        return ResourceRecommendation(
            checkpoint=checkpoint or (resources.checkpoints[0] if resources.checkpoints else "UNKNOWN"),
            sampler=sampler or (resources.samplers[0] if resources.samplers else "UNKNOWN"),
            loras=list(dict.fromkeys(loras)),
            reasoning_summary=reasoning_summary,
        )

    @staticmethod
    def _coerce_expansions(
        data: Dict[str, Any], recommendation: ResourceRecommendation
    ) -> List[ExpandedQuery]:
        items = data.get("expansions", [])
        expansions: List[ExpandedQuery] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            axis_focus = [axis for axis in item.get("axis_focus", []) if axis in AXES]
            loras = [str(v).strip() for v in item.get("loras", []) if str(v).strip()]
            expansions.append(
                ExpandedQuery(
                    label=str(item.get("label", "Expansion")).strip(),
                    prompt=str(item.get("prompt", "")).strip(),
                    axis_focus=axis_focus,
                    checkpoint=str(item.get("checkpoint", recommendation.checkpoint)).strip() or recommendation.checkpoint,
                    sampler=str(item.get("sampler", recommendation.sampler)).strip() or recommendation.sampler,
                    loras=list(dict.fromkeys(loras or recommendation.loras[:2])),
                )
            )
        return expansions

    @staticmethod
    def _tokenize_to_counter(text: str) -> Tuple[Counter, float]:
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_+-]+", text.lower())
        counter = Counter(tokens)
        norm = math.sqrt(sum(v * v for v in counter.values())) or 1.0
        return counter, norm

    @staticmethod
    def _cosine_from_counters(
        left: Counter,
        left_norm: float,
        right: Counter,
        right_norm: float,
    ) -> float:
        if not left or not right:
            return 0.0
        overlap = set(left.keys()) & set(right.keys())
        if not overlap:
            return 0.0
        dot = sum(left[token] * right[token] for token in overlap)
        return dot / (left_norm * right_norm)

    @staticmethod
    def _gallery_corpus(df: Any) -> List[str]:
        corpus: List[str] = []
        for _, row in df.iterrows():
            parts = [
                str(row.get("prompt", "")),
                str(row.get("negative_prompt", "")),
                str(row.get("style", "")),
                str(row.get("lora", "")),
                str(row.get("sampler", "")),
                str(row.get("model", "")),
            ]
            corpus.append(" ".join(part for part in parts if part))
        return corpus

    @staticmethod
    def _extract_bullets(markdown: str, heading: str) -> List[str]:
        lines = markdown.splitlines()
        start = None
        for i, line in enumerate(lines):
            if line.strip() == heading:
                start = i + 1
                break
        if start is None:
            return []

        bullets: List[str] = []
        for line in lines[start:]:
            stripped = line.strip()
            if stripped.startswith("## "):
                break
            if stripped.startswith("- "):
                bullets.append(stripped[2:].strip())
        return bullets
