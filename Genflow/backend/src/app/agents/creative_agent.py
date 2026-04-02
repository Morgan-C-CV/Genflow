from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.core.config import settings
from app.agents.prompts import EXPANSION_SYSTEM_INSTRUCTION, PLANNER_SYSTEM_INSTRUCTION
from app.core.genai_client import GenAIModel


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
    target_cluster_id: Optional[int] = None


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

        from google import genai
        self._client = genai.Client(api_key=api_key)

        self.model_name = model_name or settings.GEMINI_MODEL
        self._planner_model = GenAIModel(
            client=self._client,
            model_name=self.model_name,
            system_instruction=PLANNER_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.2,
        )
        self._expander_model = GenAIModel(
            client=self._client,
            model_name=self.model_name,
            system_instruction=EXPANSION_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.7,
        )

        default_path = Path(__file__).with_name("resources.md")
        self.resources_path = Path(resources_path) if resources_path else default_path
        self._resource_cache: Optional[ResourceContext] = None
        self._latest_cluster_labels: Optional[np.ndarray] = None
        self._latest_cluster_count: Optional[int] = None

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
        force_refresh: bool = False,
        search_engine: Optional[Any] = None,
    ) -> List[ExpandedQuery]:
        gallery_awareness = self._build_gallery_awareness(
            search_engine=search_engine,
            user_intent=user_intent,
            plan=plan,
            previous_expansions=previous_expansions,
            force_refresh=force_refresh,
        )
        payload = {
            "user_intent": user_intent.strip(),
            "fixed_constraints": plan.fixed_constraints,
            "free_variables": plan.free_variables,
            "unclear_axes": plan.unclear_axes,
            "resource_assignment_mode": "per_candidate",
            "refresh_mode": force_refresh,
            "diversity_constraints": {
                "target_candidate_count": 8,
                "prefer_distinct_resource_signatures": True,
                "prefer_distinct_checkpoints": True,
                "minimum_unique_checkpoints": min(3, max(1, len(resources.checkpoints))),
            },
            "resource_inventory": resources.to_context_block(),
        }
        if gallery_awareness:
            payload["gallery_awareness"] = gallery_awareness
        if recommendation is not None:
            payload["resource_hint"] = {
                "checkpoint": recommendation.checkpoint,
                "sampler": recommendation.sampler,
                "loras": recommendation.loras,
            }
        if previous_expansions:
            payload["previous_expansions"] = [
                {
                    "label": e.label,
                    "prompt": e.prompt,
                    "checkpoint": e.checkpoint,
                    "sampler": e.sampler,
                    "loras": e.loras,
                }
                for e in previous_expansions
            ]

        response = self._expander_model.generate_content(self._build_json_payload(payload))
        data = self._parse_json(response.text, "build_axis_expansions")
        expansions = self._coerce_expansions(data, resources)
        expansions = self._bind_selected_clusters(expansions, gallery_awareness)
        expansions = self._rebalance_expansions(
            expansions=expansions,
            resources=resources,
            previous_expansions=previous_expansions if force_refresh else None,
        )
        if len(expansions) != 8:
            raise ValueError(f"Expected 8 expansions, got {len(expansions)}")
        return expansions

    def _build_gallery_awareness(
        self,
        search_engine: Optional[Any],
        user_intent: str,
        plan: CreativeIntentPlan,
        previous_expansions: Optional[List[ExpandedQuery]] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        if search_engine is None or not hasattr(search_engine, "df") or not hasattr(search_engine, "pbo_space"):
            return {}
        df = search_engine.df
        vectors = np.asarray(search_engine.pbo_space) if search_engine.pbo_space is not None else None
        if vectors is None or len(vectors) < 2 or df is None or len(df) < 2:
            return {}

        target_k = self._select_cluster_k(vectors, min_k=12, max_k=16)
        kmeans = KMeans(n_clusters=target_k, n_init=12, random_state=42)
        labels = kmeans.fit_predict(vectors)
        self._latest_cluster_labels = labels.astype(int)
        self._latest_cluster_count = int(target_k)

        profile_rows: List[Dict[str, Any]] = []
        intent_text = " ".join(
            [
                user_intent.strip(),
                " ".join(f"{k}:{v}" for k, v in plan.fixed_constraints.items()),
                " ".join(plan.free_variables),
                " ".join(plan.unclear_axes),
            ]
        ).strip()
        intent_counter, intent_norm = self._tokenize_to_counter(intent_text)
        previous_checkpoints = {
            (e.checkpoint or "").strip().lower()
            for e in (previous_expansions or [])
            if (e.checkpoint or "").strip()
        }

        for cluster_id in range(target_k):
            member_indices = np.where(labels == cluster_id)[0].tolist()
            if not member_indices:
                continue
            cluster_df = df.iloc[member_indices]
            centroid = vectors[member_indices].mean(axis=0)
            centroid_norm = float(np.linalg.norm(centroid) or 1.0)
            dominant_model = self._mode_text(cluster_df, "model")
            dominant_sampler = self._mode_text(cluster_df, "sampler")
            dominant_loras = self._top_terms(cluster_df, "loras", top_n=3, split_words=False)
            signature_terms = self._cluster_signature_terms(cluster_df, top_n=6)
            profile_text = " ".join(
                [dominant_model, dominant_sampler, " ".join(dominant_loras), " ".join(signature_terms)]
            ).strip()
            profile_counter, profile_norm = self._tokenize_to_counter(profile_text)
            relevance = self._cosine_from_counters(
                intent_counter,
                intent_norm,
                profile_counter,
                profile_norm,
            )
            profile_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "size": int(len(member_indices)),
                    "dominant_model": dominant_model,
                    "dominant_sampler": dominant_sampler,
                    "dominant_loras": dominant_loras,
                    "signature_terms": signature_terms,
                    "relevance": float(relevance),
                    "_centroid": centroid,
                    "_centroid_norm": centroid_norm,
                    "_refresh_penalty": (
                        0.25
                        if force_refresh and dominant_model.strip().lower() in previous_checkpoints
                        else 0.0
                    ),
                }
            )

        selected = self._select_clusters_for_expansion(profile_rows, pick_count=8)
        if not selected:
            return {}

        selected_payload = []
        for rank, row in enumerate(selected, start=1):
            selected_payload.append(
                {
                    "rank": rank,
                    "cluster_id": row["cluster_id"],
                    "size": row["size"],
                    "dominant_model": row["dominant_model"],
                    "dominant_sampler": row["dominant_sampler"],
                    "dominant_loras": row["dominant_loras"],
                    "signature_terms": row["signature_terms"],
                    "relevance": round(float(row["relevance"]), 4),
                }
            )

        return {
            "cluster_count": int(len(profile_rows)),
            "selected_cluster_count": int(len(selected_payload)),
            "selection_policy": "mmr_relevance_diversity",
            "selected_clusters": selected_payload,
        }

    @staticmethod
    def _bind_selected_clusters(
        expansions: List[ExpandedQuery],
        gallery_awareness: Dict[str, Any],
    ) -> List[ExpandedQuery]:
        selected = gallery_awareness.get("selected_clusters", []) if gallery_awareness else []
        selected_ids = [
            int(item.get("cluster_id"))
            for item in selected
            if isinstance(item, dict) and str(item.get("cluster_id", "")).isdigit()
        ]
        if not selected_ids:
            return expansions
        for i, item in enumerate(expansions):
            if item.target_cluster_id not in selected_ids:
                item.target_cluster_id = selected_ids[i % len(selected_ids)]
        return expansions

    @staticmethod
    def _select_cluster_k(vectors: np.ndarray, min_k: int = 12, max_k: int = 16) -> int:
        n_samples = int(len(vectors))
        upper = max(2, min(max_k, n_samples - 1))
        lower = min(min_k, upper)
        candidates = list(range(lower, upper + 1))
        if len(candidates) == 1:
            return candidates[0]

        if n_samples > 1200:
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(n_samples, size=1200, replace=False)
            sample_vectors = vectors[sample_indices]
        else:
            sample_vectors = vectors

        best_k = candidates[0]
        best_score = -1.0
        for k in candidates:
            if len(sample_vectors) <= k:
                continue
            model = KMeans(n_clusters=k, n_init=8, random_state=42)
            labels = model.fit_predict(sample_vectors)
            if len(set(labels)) < 2:
                continue
            try:
                score = float(silhouette_score(sample_vectors, labels, metric="euclidean"))
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best_k = k
        return int(best_k)

    @staticmethod
    def _mode_text(df: Any, column: str) -> str:
        if column not in df.columns or df.empty:
            return "UNKNOWN"
        series = df[column].fillna("").astype(str)
        series = series[series.str.strip() != ""]
        if series.empty:
            return "UNKNOWN"
        try:
            return str(series.mode().iloc[0]).strip() or "UNKNOWN"
        except Exception:
            return str(series.iloc[0]).strip() or "UNKNOWN"

    def _cluster_signature_terms(self, df: Any, top_n: int = 6) -> List[str]:
        counter = Counter()
        for _, row in df.head(min(len(df), 220)).iterrows():
            text = " ".join(
                [
                    str(row.get("prompt", "")),
                    str(row.get("style", "")),
                    str(row.get("negative_prompt", "")),
                ]
            )
            term_counter, _ = self._tokenize_to_counter(text)
            counter.update(term_counter)
        blocked = {
            "best", "quality", "masterpiece", "high", "ultra", "detailed", "detail", "8k",
            "prompt", "style", "lighting", "background", "color", "negative",
        }
        terms = [t for t, _ in counter.most_common(top_n * 4) if t not in blocked and len(t) > 1]
        return terms[:top_n]

    @staticmethod
    def _top_terms(df: Any, column: str, top_n: int = 3, split_words: bool = True) -> List[str]:
        if column not in df.columns or df.empty:
            return []
        counter = Counter()
        for value in df[column].fillna("").astype(str).tolist():
            text = value.strip()
            if not text:
                continue
            if split_words:
                for token in re.split(r"[,\s]+", text):
                    token = token.strip()
                    if token:
                        counter[token] += 1
            else:
                counter[text] += 1
        return [term for term, _ in counter.most_common(top_n)]

    @staticmethod
    def _centroid_cosine(a: np.ndarray, a_norm: float, b: np.ndarray, b_norm: float) -> float:
        return float(np.dot(a, b) / ((a_norm or 1.0) * (b_norm or 1.0)))

    def _select_clusters_for_expansion(
        self,
        profile_rows: List[Dict[str, Any]],
        pick_count: int = 8,
    ) -> List[Dict[str, Any]]:
        if not profile_rows:
            return []
        total_size = max(1, sum(int(row["size"]) for row in profile_rows))
        selected: List[Dict[str, Any]] = []
        remaining = list(profile_rows)
        while remaining and len(selected) < pick_count:
            best_row = None
            best_score = -10.0
            for row in remaining:
                size_bonus = float(row["size"]) / float(total_size)
                base = 0.72 * float(row["relevance"]) + 0.18 * size_bonus - float(row["_refresh_penalty"])
                if not selected:
                    score = base
                else:
                    similarity = max(
                        self._centroid_cosine(
                            row["_centroid"],
                            row["_centroid_norm"],
                            s["_centroid"],
                            s["_centroid_norm"],
                        )
                        for s in selected
                    )
                    score = base - 0.22 * max(0.0, similarity)
                if score > best_score:
                    best_score = score
                    best_row = row
            if best_row is None:
                break
            selected.append(best_row)
            remaining = [row for row in remaining if row["cluster_id"] != best_row["cluster_id"]]
        return selected

    @staticmethod
    def summarize_expansion_resources(expansions: Sequence[ExpandedQuery]) -> ResourceRecommendation:
        if not expansions:
            return ResourceRecommendation(
                checkpoint="UNKNOWN",
                sampler="UNKNOWN",
                loras=[],
                reasoning_summary="No expansion resources available.",
            )
        checkpoint_counts = Counter((item.checkpoint or "").strip() for item in expansions if (item.checkpoint or "").strip())
        sampler_counts = Counter((item.sampler or "").strip() for item in expansions if (item.sampler or "").strip())
        lora_counts = Counter(
            lora.strip() for item in expansions for lora in item.loras if lora.strip()
        )
        checkpoint = checkpoint_counts.most_common(1)[0][0] if checkpoint_counts else "UNKNOWN"
        sampler = sampler_counts.most_common(1)[0][0] if sampler_counts else "UNKNOWN"
        loras = [name for name, _ in lora_counts.most_common(3)]
        reasoning_summary = (
            f"Per-candidate resource assignment enabled; "
            f"{len(checkpoint_counts)} checkpoints used across {len(expansions)} candidates."
        )
        return ResourceRecommendation(
            checkpoint=checkpoint,
            sampler=sampler,
            loras=loras,
            reasoning_summary=reasoning_summary,
        )

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
        avoid_indices: Optional[Sequence[int]] = None,
    ) -> CandidateWall:
        groups: List[List[int]] = []
        flat_indices: List[int] = []
        seen_indices: set[int] = set()
        blocked_indices: set[int] = {int(i) for i in (avoid_indices or [])}
        query_labels: List[str] = []

        for expansion in expansions:
            query_labels.append(expansion.label)
            candidates = self._rank_gallery_candidates(search_engine, expansion, top_k=top_k)
            group: List[int] = []
            for candidate in candidates:
                idx = candidate["index"]
                if idx in seen_indices or idx in blocked_indices:
                    continue
                group.append(idx)
                seen_indices.add(idx)
                flat_indices.append(idx)
                if len(group) == per_query_k:
                    break
            if len(group) < per_query_k:
                for idx in self._fallback_pool(
                    search_engine,
                    seen_indices,
                    target_cluster_id=expansion.target_cluster_id,
                ):
                    if idx in seen_indices or idx in blocked_indices:
                        continue
                    group.append(idx)
                    seen_indices.add(idx)
                    flat_indices.append(idx)
                    if len(group) == per_query_k:
                        break
            if len(group) < per_query_k:
                for idx in self._fallback_pool(
                    search_engine,
                    seen_indices,
                    target_cluster_id=None,
                ):
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
                results = search_engine.search_top_k(
                    query_vector=np.asarray(query_vector).reshape(1, -1),
                    top_k=max(top_k * 4, top_k),
                )
                return self._filter_candidates_by_cluster(
                    search_engine=search_engine,
                    candidates=results,
                    target_cluster_id=expansion.target_cluster_id,
                    top_k=top_k,
                )
            except Exception:
                pass

        results = self._rank_gallery_candidates_text(search_engine, expansion, top_k=max(top_k * 4, top_k))
        return self._filter_candidates_by_cluster(
            search_engine=search_engine,
            candidates=results,
            target_cluster_id=expansion.target_cluster_id,
            top_k=top_k,
        )

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

    def _fallback_pool(
        self,
        search_engine: Any,
        seen_indices: set[int],
        target_cluster_id: Optional[int] = None,
    ) -> List[int]:
        if hasattr(search_engine, "df"):
            if target_cluster_id is not None:
                labels = self._resolve_cluster_labels(search_engine)
                if labels is not None and len(labels) == len(search_engine.df):
                    cluster_members = [
                        int(i)
                        for i, c in enumerate(labels)
                        if int(c) == int(target_cluster_id) and i not in seen_indices
                    ]
                    if cluster_members:
                        return cluster_members
            return [int(i) for i in range(len(search_engine.df)) if i not in seen_indices]
        return []

    def _filter_candidates_by_cluster(
        self,
        search_engine: Any,
        candidates: List[Dict[str, Any]],
        target_cluster_id: Optional[int],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if target_cluster_id is None:
            return candidates[:top_k]
        labels = self._resolve_cluster_labels(search_engine)
        if labels is None:
            return candidates[:top_k]
        in_cluster = []
        out_cluster = []
        for item in candidates:
            idx = int(item.get("index", -1))
            if 0 <= idx < len(labels) and int(labels[idx]) == int(target_cluster_id):
                in_cluster.append(item)
            else:
                out_cluster.append(item)
        if in_cluster:
            return in_cluster[:top_k]
        return out_cluster[:top_k]

    def _resolve_cluster_labels(self, search_engine: Any) -> Optional[np.ndarray]:
        if not hasattr(search_engine, "pbo_space") or search_engine.pbo_space is None:
            return None
        vectors = np.asarray(search_engine.pbo_space)
        if len(vectors) < 2:
            return None
        if (
            self._latest_cluster_labels is not None
            and len(self._latest_cluster_labels) == len(vectors)
            and self._latest_cluster_count is not None
        ):
            return self._latest_cluster_labels
        cluster_k = self._select_cluster_k(vectors, min_k=12, max_k=16)
        labels = KMeans(n_clusters=cluster_k, n_init=10, random_state=42).fit_predict(vectors).astype(int)
        self._latest_cluster_labels = labels
        self._latest_cluster_count = int(cluster_k)
        return labels

    @staticmethod
    def _build_json_payload(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_json(text: str, context: str) -> Dict[str, Any]:
        raw_text = text.strip()
        for candidate in CreativeAgent._json_candidates(raw_text):
            try:
                parsed = json.loads(candidate)
                return CreativeAgent._coerce_json_payload(parsed, context)
            except json.JSONDecodeError:
                pass
            parsed = CreativeAgent._decode_first_json(candidate)
            if parsed is not None:
                return CreativeAgent._coerce_json_payload(parsed, context)
        raise ValueError(f"{context}: model returned invalid JSON: {raw_text[:400]}")

    @staticmethod
    def _json_candidates(text: str) -> List[str]:
        candidates: List[str] = []
        if text:
            candidates.append(text)
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
            block = match.group(1).strip()
            if block:
                candidates.append(block)
        return candidates

    @staticmethod
    def _decode_first_json(text: str) -> Optional[Any]:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\{\[]", text):
            start = match.start()
            try:
                obj, _ = decoder.raw_decode(text, idx=start)
                return obj
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _coerce_json_payload(parsed: Any, context: str) -> Dict[str, Any]:
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and context == "build_axis_expansions":
            return {"expansions": parsed}
        raise ValueError(f"{context}: model returned non-object JSON: {str(parsed)[:200]}")

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
        data: Dict[str, Any], resources: ResourceContext
    ) -> List[ExpandedQuery]:
        items = data.get("expansions", [])
        expansions: List[ExpandedQuery] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            axis_focus = [axis for axis in item.get("axis_focus", []) if axis in AXES]
            checkpoint = str(item.get("checkpoint", "")).strip()
            sampler = str(item.get("sampler", "")).strip()
            loras = [str(v).strip() for v in item.get("loras", []) if str(v).strip()]
            raw_cluster_id = item.get("cluster_id")
            target_cluster_id = None
            if isinstance(raw_cluster_id, int):
                target_cluster_id = int(raw_cluster_id)
            elif isinstance(raw_cluster_id, str) and raw_cluster_id.strip().isdigit():
                target_cluster_id = int(raw_cluster_id.strip())
            if resources.checkpoints and checkpoint not in resources.checkpoints:
                checkpoint = resources.checkpoints[i % len(resources.checkpoints)]
            if resources.samplers and sampler not in resources.samplers:
                sampler = resources.samplers[i % len(resources.samplers)]
            if resources.loras:
                loras = [l for l in loras if l in resources.loras]
            loras = list(dict.fromkeys(loras))
            expansions.append(
                ExpandedQuery(
                    label=str(item.get("label", "Expansion")).strip(),
                    prompt=str(item.get("prompt", "")).strip(),
                    axis_focus=axis_focus,
                    checkpoint=checkpoint or (resources.checkpoints[i % len(resources.checkpoints)] if resources.checkpoints else "UNKNOWN"),
                    sampler=sampler or (resources.samplers[i % len(resources.samplers)] if resources.samplers else "UNKNOWN"),
                    loras=loras,
                    target_cluster_id=target_cluster_id,
                )
            )
        return expansions

    @staticmethod
    def _rebalance_expansions(
        expansions: List[ExpandedQuery],
        resources: ResourceContext,
        previous_expansions: Optional[List[ExpandedQuery]] = None,
    ) -> List[ExpandedQuery]:
        if not expansions:
            return expansions
        available_checkpoints = resources.checkpoints or [item.checkpoint or "UNKNOWN" for item in expansions]
        target_unique = min(max(1, len(available_checkpoints)), min(4, len(expansions)))
        used_checkpoints = {item.checkpoint for item in expansions if item.checkpoint}
        if len(used_checkpoints) < target_unique:
            pool = [cp for cp in available_checkpoints if cp not in used_checkpoints]
            cursor = 0
            for i, item in enumerate(expansions):
                if len(used_checkpoints) >= target_unique or cursor >= len(pool):
                    break
                if i < len(expansions) and item.checkpoint in used_checkpoints and sum(1 for e in expansions if e.checkpoint == item.checkpoint) > 1:
                    item.checkpoint = pool[cursor]
                    used_checkpoints.add(item.checkpoint)
                    cursor += 1
        if previous_expansions:
            previous_signatures = {
                (
                    (e.checkpoint or "").strip().lower(),
                    (e.sampler or "").strip().lower(),
                    tuple(sorted((l or "").strip().lower() for l in e.loras if (l or "").strip())),
                )
                for e in previous_expansions
            }
            replacement_checkpoints = iter(available_checkpoints)
            for item in expansions:
                signature = (
                    (item.checkpoint or "").strip().lower(),
                    (item.sampler or "").strip().lower(),
                    tuple(sorted((l or "").strip().lower() for l in item.loras if (l or "").strip())),
                )
                if signature not in previous_signatures:
                    continue
                for candidate in replacement_checkpoints:
                    if candidate and candidate.lower() != (item.checkpoint or "").strip().lower():
                        item.checkpoint = candidate
                        break
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
            if line.startswith("- "):
                value = line[2:].strip()
                value = re.sub(r"^`(.+)`$", r"\1", value)
                if value:
                    bullets.append(value)
        return list(dict.fromkeys(bullets))
