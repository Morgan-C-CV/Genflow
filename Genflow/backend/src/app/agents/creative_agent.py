from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


AXES = [
    "subject",
    "style",
    "composition",
    "lighting_vibe",
    "background_setting",
    "color_palette",
]

STYLE_VARIANTS = [
    {
        "label": "Watercolor children's book style",
        "prompt": "watercolor children's book illustration, soft washes, warm paper texture, gentle contours",
    },
    {
        "label": "Impasto oil painting classic",
        "prompt": "impasto oil painting, classic fine-art surface, visible brush texture, rich pigment density",
    },
    {
        "label": "Anime 2D flat color",
        "prompt": "anime 2D, flat color, crisp linework, expressive silhouette, clean cel shading",
    },
    {
        "label": "Vector art minimalist",
        "prompt": "vector art, minimalist, flat shapes, clean geometry, limited palette, graphic clarity",
    },
]

AXIS_QUESTION_BANK = {
    "subject": "你希望画面的核心主体是谁或是什么？",
    "style": "你更想要偏 `插画/绘本`、`写实/电影感`、`二次元`，还是 `扁平化/矢量`？",
    "composition": "画面更偏 `特写`、`半身`、`全身`、`广角`，还是 `俯视/仰视`？",
    "lighting_vibe": "你希望氛围更偏 `温暖`、`冷峻`、`梦幻`、`赛博霓虹`，还是 `电影感`？",
    "background_setting": "背景更想要 `纯色`、`室内`、`室外自然`、`城市`，还是 `抽象环境`？",
    "color_palette": "颜色更想要 `高饱和`、`低饱和`、`单色系`，还是有明确主色？",
}

STYLE_KEYWORDS = {
    "watercolor": ["水彩", "watercolor", "绘本", "children's book", "storybook"],
    "oil_painting": ["油画", "oil painting", "impasto", "classic painting"],
    "anime": ["二次元", "anime", "日漫", "manga", "flat color", "flat-colour"],
    "vector": ["矢量", "vector", "minimalist", "扁平化", "graphic design"],
    "photoreal": ["写实", "photoreal", "realistic", "cinematic", "film still"],
    "illustration": ["插画", "illustration", "storybook", "editorial"],
}

COMPOSITION_KEYWORDS = {
    "close-up": ["特写", "close-up", "extreme close-up", "portrait", "headshot"],
    "medium shot": ["半身", "medium shot", "bust shot", "waist up"],
    "full body": ["全身", "full body", "full shot"],
    "wide shot": ["广角", "wide shot", "establishing shot"],
    "low angle": ["仰视", "low angle"],
    "high angle": ["俯视", "high angle", "bird's-eye"],
}

LIGHTING_KEYWORDS = {
    "warm": ["温暖", "warm", "golden hour", "sunset", "soft light"],
    "cold": ["冷", "cold", "moonlight", "blue light", "night"],
    "neon": ["赛博", "neon", "cyberpunk"],
    "cinematic": ["电影感", "cinematic", "film still", "dramatic"],
    "dreamy": ["梦幻", "dreamy", "ethereal", "volumetric"],
}

BACKGROUND_KEYWORDS = {
    "plain background": ["纯色", "plain background", "solid background"],
    "indoor": ["室内", "indoor", "library", "room", "studio"],
    "outdoor": ["室外", "outdoor", "nature", "forest", "city", "street"],
    "fantasy": ["废墟", "ruins", "castle", "fantasy", "magical", "enchanted"],
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    normalized = _normalize(text)
    return any(keyword.lower() in normalized for keyword in keywords)


def _first_match(text: str, mapping: Dict[str, Sequence[str]]) -> Optional[str]:
    for label, keywords in mapping.items():
        if _contains_any(text, keywords):
            return label
    return None


@dataclass
class ResourceContext:
    raw_markdown: str
    checkpoints: List[str] = field(default_factory=list)
    loras: List[str] = field(default_factory=list)
    samplers: List[str] = field(default_factory=list)
    vaes: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)

    def recommended_checkpoint(self, analysis: "CreativeIntentPlan") -> str:
        subject = analysis.fixed_constraints.get("subject", "").lower()
        style = analysis.fixed_constraints.get("style", "").lower()
        if any(token in subject for token in ["cat", "猫", "animal", "furry", "kemono"]):
            return self._match_or_default(["NoobAI", "Pony"], fallback=self.checkpoints)
        if any(token in style for token in ["anime", "二次元", "manga"]):
            return self._match_or_default(["Pony", "incursiosMemeDiffusion_v27PDXL"], fallback=self.checkpoints)
        if any(token in style for token in ["watercolor", "illustration", "绘本"]):
            return self._match_or_default(["lilithsDesire_v10", "Juggernaut_XL_-_Ragnarok_by_RunDiffusion"], fallback=self.checkpoints)
        if any(token in style for token in ["vector", "minimalist", "graphic"]):
            return self._match_or_default(["Kody", "Juggernaut_XL_-_Ragnarok_by_RunDiffusion"], fallback=self.checkpoints)
        return self._match_or_default(["Juggernaut_XL_-_Ragnarok_by_RunDiffusion", "CyberRealistic_CyberIllustrious_-_v7-0"], fallback=self.checkpoints)

    def recommended_loras(self, analysis: "CreativeIntentPlan") -> List[str]:
        style = analysis.fixed_constraints.get("style", "").lower()
        result: List[str] = []
        if any(token in style for token in ["illustration", "watercolor", "绘本", "story"]):
            result.extend(["add-detail-xl", "zy_Detailed_Backgrounds_v1"])
        if any(token in style for token in ["anime", "二次元", "flat color"]):
            result.extend(["Pony_DetailV2.0", "great_lighting"])
        if any(token in style for token in ["photoreal", "realistic", "cinematic", "写实"]):
            result.extend(["perfection style", "great_lighting"])
        if not result:
            result.extend(self.loras[:2])
        return list(dict.fromkeys(result))

    def recommended_sampler(self, analysis: "CreativeIntentPlan") -> str:
        style = analysis.fixed_constraints.get("style", "").lower()
        if any(token in style for token in ["vector", "minimalist", "graphic"]):
            return self._match_or_default(["Euler a", "DPM++ 2M Karras"], fallback=self.samplers)
        if any(token in style for token in ["anime", "flat color", "二次元"]):
            return self._match_or_default(["Euler a", "DPM++ 2M Karras"], fallback=self.samplers)
        return self._match_or_default(["DPM++ 2M Karras", "DPM++ 3M SDE Exponential"], fallback=self.samplers)

    @staticmethod
    def _match_or_default(options: Sequence[str], fallback: Sequence[str]) -> str:
        for option in options:
            if option in fallback:
                return option
        return fallback[0] if fallback else options[0]


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
class CandidateWall:
    groups: List[List[int]]
    flat_indices: List[int]
    query_labels: List[str]


class CreativeAgent:
    def __init__(self, resources_path: Optional[str] = None):
        default_path = Path(__file__).with_name("resources.md")
        self.resources_path = Path(resources_path) if resources_path else default_path
        self._resource_cache: Optional[ResourceContext] = None
        self._corpus_cache_key = None
        self._corpus_cache: List[Tuple[Counter, float]] = []

    def load_resources(self) -> ResourceContext:
        if self._resource_cache is not None:
            return self._resource_cache

        raw = self.resources_path.read_text(encoding="utf-8")
        checkpoints = self._extract_bullets(raw, "## Checkpoints")
        loras = self._extract_bullets(raw, "## LoRAs")
        samplers = self._extract_bullets(raw, "## Samplers")
        vaes = self._extract_bullets(raw, "## VAE / Auxiliary")
        hints = self._extract_bullets(raw, "## Retrieval Hints")

        self._resource_cache = ResourceContext(
            raw_markdown=raw,
            checkpoints=checkpoints,
            loras=loras,
            samplers=samplers,
            vaes=vaes,
            hints=hints,
        )
        return self._resource_cache

    def analyze_intent(self, user_intent: str) -> CreativeIntentPlan:
        text = user_intent.strip()
        fixed_constraints: Dict[str, str] = {}

        subject = self._extract_subject(text)
        if subject:
            fixed_constraints["subject"] = subject

        style = _first_match(text, STYLE_KEYWORDS)
        if style:
            fixed_constraints["style"] = style

        composition = _first_match(text, COMPOSITION_KEYWORDS)
        if composition:
            fixed_constraints["composition"] = composition

        lighting = _first_match(text, LIGHTING_KEYWORDS)
        if lighting:
            fixed_constraints["lighting_vibe"] = lighting

        background = _first_match(text, BACKGROUND_KEYWORDS)
        if background:
            fixed_constraints["background_setting"] = background

        if any(token in _normalize(text) for token in ["红", "蓝", "绿", "黄", "black", "white", "pastel", "neon"]):
            fixed_constraints["color_palette"] = self._extract_color_palette(text)

        locked_axes = [axis for axis in AXES if axis in fixed_constraints]
        free_variables = [axis for axis in AXES if axis not in fixed_constraints]

        unclear_axes = free_variables[:3]
        clarification_questions = [AXIS_QUESTION_BANK[axis] for axis in unclear_axes]

        if "subject" not in fixed_constraints or len(locked_axes) < 2:
            next_action = "ask_user"
            reasoning_summary = (
                f"已锁定轴: {', '.join(locked_axes) if locked_axes else '无'}; "
                f"优先补充: {', '.join(unclear_axes[:2]) if unclear_axes else '无'}"
            )
        else:
            next_action = "retrieve_resources"
            reasoning_summary = (
                f"已锁定轴: {', '.join(locked_axes)}; "
                f"待发散轴: {', '.join(unclear_axes[:3]) if unclear_axes else '无'}"
            )

        return CreativeIntentPlan(
            user_intent=text,
            fixed_constraints=fixed_constraints,
            free_variables=free_variables,
            locked_axes=locked_axes,
            unclear_axes=unclear_axes,
            next_action=next_action,
            clarification_questions=clarification_questions,
            reasoning_summary=reasoning_summary,
        )

    def build_clarification_prompt(self, plan: CreativeIntentPlan) -> str:
        if not plan.clarification_questions:
            return "当前意图已经足够明确，可以继续检索。"
        items = "\n".join(f"- {question}" for question in plan.clarification_questions[:3])
        locked = ", ".join(
            f"{axis}={value}" for axis, value in plan.fixed_constraints.items()
        ) or "无"
        return (
            "我已经锁定的内容如下：\n"
            f"{locked}\n\n"
            "还需要你补充下面几个未指定轴：\n"
            f"{items}"
        )

    def build_axis_expansions(
        self,
        user_intent: str,
        plan: CreativeIntentPlan,
        resources: ResourceContext,
    ) -> List[ExpandedQuery]:
        checkpoint = resources.recommended_checkpoint(plan)
        sampler = resources.recommended_sampler(plan)
        loras = resources.recommended_loras(plan)

        subject = plan.fixed_constraints.get("subject", user_intent)
        background = plan.fixed_constraints.get("background_setting")
        lighting = plan.fixed_constraints.get("lighting_vibe")
        composition = plan.fixed_constraints.get("composition")
        color_palette = plan.fixed_constraints.get("color_palette")

        variants: List[ExpandedQuery] = []
        for variant in STYLE_VARIANTS:
            prompt_parts = [
                f"subject: {subject}",
                variant["prompt"],
            ]
            if composition:
                prompt_parts.append(f"composition: {composition}")
            else:
                prompt_parts.append("composition: intentional framing, readable silhouette")
            if lighting:
                prompt_parts.append(f"lighting: {lighting}")
            else:
                prompt_parts.append("lighting: strong readable mood contrast")
            if background:
                prompt_parts.append(f"background: {background}")
            else:
                prompt_parts.append("background: expressive environment, not random")
            if color_palette:
                prompt_parts.append(f"palette: {color_palette}")
            else:
                prompt_parts.append("palette: variant-specific palette")

            prompt_parts.extend(self._free_axis_expansions(plan.free_variables))
            prompt = ", ".join(dict.fromkeys(prompt_parts))
            variants.append(
                ExpandedQuery(
                    label=variant["label"],
                    prompt=prompt,
                    axis_focus=plan.free_variables[:3],
                    checkpoint=checkpoint,
                    sampler=sampler,
                    loras=loras[:2],
                )
            )

        return variants

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
                fallback_candidates = self._fallback_pool(search_engine, seen_indices)
                for idx in fallback_candidates:
                    if idx in seen_indices:
                        continue
                    group.append(idx)
                    seen_indices.add(idx)
                    flat_indices.append(idx)
                    if len(group) == per_query_k:
                        break
            groups.append(group)

        return CandidateWall(groups=groups, flat_indices=flat_indices, query_labels=query_labels)

    def describe_wall(self, wall: CandidateWall) -> str:
        lines = ["16 图发散矩阵的 4 个轴向分组："]
        for i, label in enumerate(wall.query_labels, start=1):
            lines.append(f"- Group {i}: {label}")
        return "\n".join(lines)

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
                results = search_engine.search_top_k(
                    query_vector=np.asarray(query_vector).reshape(1, -1),
                    top_k=top_k,
                )
                return results
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
        order = [idx for idx, _ in scored[:top_k]]
        results: List[Dict[str, Any]] = []
        for idx in order:
            row = df.iloc[int(idx)].to_dict()
            row["index"] = int(idx)
            score = score_map[int(idx)]
            row["distance"] = round(float(1.0 - score), 4)
            results.append(row)
        return results

    def _fallback_pool(self, search_engine: Any, seen_indices: set[int]) -> List[int]:
        if hasattr(search_engine, "df"):
            return [int(i) for i in range(len(search_engine.df)) if i not in seen_indices]
        return []

    def _free_axis_expansions(self, free_axes: Sequence[str]) -> List[str]:
        expansions: List[str] = []
        for axis in free_axes[:3]:
            if axis == "style":
                expansions.append("style variation: deliberate, high-variance finish")
            elif axis == "composition":
                expansions.append("composition variation: one readable focal point, no randomness")
            elif axis == "lighting_vibe":
                expansions.append("lighting variation: distinct contrast and atmosphere")
            elif axis == "background_setting":
                expansions.append("background variation: materially different setting")
            elif axis == "color_palette":
                expansions.append("palette variation: orthogonal color direction")
        return expansions

    @staticmethod
    def _extract_subject(text: str) -> Optional[str]:
        patterns = [
            r"^(?:我想|想要|我要|请帮我|帮我)?生成(?:一张|一幅|一个)?([^，。；、\n]+)",
            r"关于([^，。；、\n]+)",
            r"生成一张([^，。；、\n]+)",
            r"画([^，。；、\n]+)",
            r"做一张([^，。；、\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip()
                candidate = re.sub(r"[的]?插画|的?图|的?作品|的?海报|的?海报设计|的?海报", "", candidate).strip()
                candidate = re.sub(r"^(?:一张|一幅|一个)?", "", candidate).strip()
                if candidate:
                    return candidate
        return None

    @staticmethod
    def _extract_color_palette(text: str) -> str:
        normalized = _normalize(text)
        if "红" in normalized or "red" in normalized:
            return "red-dominant"
        if "蓝" in normalized or "blue" in normalized:
            return "blue-dominant"
        if "绿" in normalized or "green" in normalized:
            return "green-dominant"
        if "黄" in normalized or "yellow" in normalized:
            return "yellow-dominant"
        if "黑" in normalized or "black" in normalized:
            return "monochrome-dark"
        if "white" in normalized or "白" in normalized:
            return "high-key-neutral"
        if "pastel" in normalized:
            return "pastel"
        if "neon" in normalized:
            return "neon"
        return "unspecified"

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
