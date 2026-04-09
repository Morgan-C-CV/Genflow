from __future__ import annotations

from typing import Dict, List

import numpy as np

from app.modules.embedding_v4 import ImageEmbeddingSearch
from app.core.config import settings


class SearchRepository:
    def __init__(self):
        self.search_engine = ImageEmbeddingSearch(
            metadata_path=settings.METADATA_PATH,
            gallery_dir=settings.GALLERY_DIR
        )

    def search_by_index(self, index: int, top_k: int = 5):
        return self.search_engine.search_top_k(query_index=index, top_k=top_k)

    def search_diverse_references(
        self,
        query_index: int,
        complementary_k: int = 2,
        exploratory_k: int = 2,
        include_counterexample: bool = True,
        near_pool_size: int = 24,
        far_pool_size: int = 24,
    ) -> Dict[str, object]:
        total_requested = 1 + complementary_k + exploratory_k + int(include_counterexample)
        available = len(self.search_engine.df)
        if available == 0:
            raise ValueError("Search corpus is empty.")
        if not 0 <= query_index < available:
            raise IndexError(f"query_index out of range: {query_index}")
        if available < total_requested:
            raise ValueError(
                f"Search corpus size {available} is smaller than requested bundle size {total_requested}."
            )

        pbo_space = self.search_engine.pbo_space
        df = self.search_engine.df
        query_vector = pbo_space[query_index]

        all_distances = self._cosine_distances(pbo_space, query_vector)
        nearest_pool = self.search_engine.search_top_k(
            query_index=query_index,
            top_k=min(max(near_pool_size, total_requested * 3), available),
        )

        selected_indices = {query_index}
        references: List[Dict[str, object]] = [
            self._build_reference(
                df=df,
                distances=all_distances,
                index=query_index,
                role="best",
                role_label="Optimal baseline",
                selection_reason="Current best-discovered sample; primary anchor for subject, style, and parameter inheritance.",
            )
        ]

        complementary_candidates = [
            item for item in nearest_pool
            if int(item["index"]) != query_index
        ]
        complementary_indices = self._select_diverse_near_neighbors(
            candidate_indices=[int(item["index"]) for item in complementary_candidates],
            distances=all_distances,
            count=complementary_k,
            pbo_space=pbo_space,
        )
        for offset, idx in enumerate(complementary_indices, start=1):
            selected_indices.add(idx)
            references.append(
                self._build_reference(
                    df=df,
                    distances=all_distances,
                    index=idx,
                    role="complementary_knn",
                    role_label=f"Complementary KNN #{offset}",
                    selection_reason="Close semantic neighbor selected for high relevance with complementary variation, not simple duplication.",
                )
            )

        exploratory_indices = self._select_exploratory_neighbors(
            distances=all_distances,
            selected_indices=selected_indices,
            count=exploratory_k,
        )
        for offset, idx in enumerate(exploratory_indices, start=1):
            selected_indices.add(idx)
            references.append(
                self._build_reference(
                    df=df,
                    distances=all_distances,
                    index=idx,
                    role="exploratory",
                    role_label=f"Exploratory sample #{offset}",
                    selection_reason="Second-tier relevant sample chosen to expand the idea space beyond the tight nearest-neighbor cluster.",
                )
            )

        counterexample_reference = None
        if include_counterexample:
            counterexample_index = self._select_counterexample(
                distances=all_distances,
                selected_indices=selected_indices,
                far_pool_size=far_pool_size,
            )
            selected_indices.add(counterexample_index)
            counterexample_reference = self._build_reference(
                df=df,
                distances=all_distances,
                index=counterexample_index,
                role="counterexample",
                role_label="Counterexample",
                selection_reason="Most irrelevant sample retained as a negative guardrail; use it to avoid drift, not to imitate.",
            )
            references.append(counterexample_reference)

        return {
            "query_index": query_index,
            "selection_summary": {
                "strategy": "fixed_1_2_2_1",
                "total_references": len(references),
                "near_pool_size": min(max(near_pool_size, total_requested * 3), available),
                "far_pool_size": min(far_pool_size, available),
            },
            "counts": {
                "best": 1,
                "complementary_knn": len(complementary_indices),
                "exploratory": len(exploratory_indices),
                "counterexample": 1 if counterexample_reference else 0,
            },
            "references": references,
        }

    def get_all_data(self):
        return self.search_engine.df

    @staticmethod
    def _cosine_distances(pbo_space: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query vector norm is zero; cosine distance is undefined.")
        doc_norms = np.linalg.norm(pbo_space, axis=1)
        safe_norms = np.where(doc_norms == 0, 1.0, doc_norms)
        cosine_similarity = (pbo_space @ query_vector) / (safe_norms * query_norm)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        return 1.0 - cosine_similarity

    def _select_diverse_near_neighbors(
        self,
        candidate_indices: List[int],
        distances: np.ndarray,
        count: int,
        pbo_space: np.ndarray,
    ) -> List[int]:
        if count <= 0:
            return []
        if len(candidate_indices) < count:
            raise ValueError("Insufficient near-neighbor candidates for complementary selection.")

        ordered = sorted(candidate_indices, key=lambda idx: distances[idx])
        ordered_distances = [float(distances[idx]) for idx in ordered]
        nearest_distance = float(ordered_distances[0])
        quantile_distance = float(np.quantile(ordered_distances, 0.35))
        ratio_distance = nearest_distance * 1.6 if nearest_distance > 0 else quantile_distance
        distance_gate = max(quantile_distance, ratio_distance)
        focused_candidates = [
            idx for idx in ordered
            if float(distances[idx]) <= distance_gate
        ]
        if len(focused_candidates) < count:
            fallback_size = min(len(ordered), max(count + 3, count * 3))
            focused_candidates = ordered[:fallback_size]
        selected = [focused_candidates[0]]
        while len(selected) < count:
            best_idx = None
            best_score = None
            for idx in focused_candidates:
                if idx in selected:
                    continue
                diversity_bonus = min(
                    np.linalg.norm(pbo_space[idx] - pbo_space[chosen]) for chosen in selected
                )
                score = (-float(distances[idx]), float(diversity_bonus) * 0.25)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
        if len(selected) < count:
            raise ValueError("Failed to select enough complementary near neighbors.")
        return selected

    @staticmethod
    def _select_exploratory_neighbors(
        distances: np.ndarray,
        selected_indices: set[int],
        count: int,
    ) -> List[int]:
        if count <= 0:
            return []
        ordered = [
            idx for idx in np.argsort(distances).tolist()
            if idx not in selected_indices
        ]
        if len(ordered) < count:
            raise ValueError("Insufficient candidates for exploratory selection.")

        candidate_count = max(count * 3, count + 2)
        exploratory_pool = ordered[:candidate_count]
        start = len(exploratory_pool) // 2
        chosen = exploratory_pool[start:start + count]
        if len(chosen) < count:
            chosen.extend(exploratory_pool[:count - len(chosen)])
        return chosen[:count]

    @staticmethod
    def _select_counterexample(
        distances: np.ndarray,
        selected_indices: set[int],
        far_pool_size: int,
    ) -> int:
        ordered = [
            idx for idx in np.argsort(distances)[::-1].tolist()
            if idx not in selected_indices
        ]
        if not ordered:
            raise ValueError("Insufficient candidates for counterexample selection.")
        far_pool = ordered[:max(1, far_pool_size)]
        return far_pool[0]

    @staticmethod
    def _build_reference(
        df,
        distances: np.ndarray,
        index: int,
        role: str,
        role_label: str,
        selection_reason: str,
    ) -> Dict[str, object]:
        row = df.iloc[index]
        item = row.to_dict()
        item.update(
            {
                "index": int(index),
                "distance": round(float(distances[index]), 4),
                "role": role,
                "role_label": role_label,
                "selection_reason": selection_reason,
            }
        )
        for key, value in list(item.items()):
            if isinstance(value, np.integer):
                item[key] = int(value)
            elif isinstance(value, np.floating):
                item[key] = float(value)
        return item
