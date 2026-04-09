import os
import sys
import types
import unittest

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


embedding_stub = types.ModuleType("app.modules.embedding_v4")


class _DummyImageEmbeddingSearch:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Test should not construct the real ImageEmbeddingSearch.")


embedding_stub.ImageEmbeddingSearch = _DummyImageEmbeddingSearch
sys.modules.setdefault("app.modules.embedding_v4", embedding_stub)

config_stub = types.ModuleType("app.core.config")
config_stub.settings = types.SimpleNamespace(
    METADATA_PATH="",
    GALLERY_DIR="",
    GOOGLE_API_KEY="test-key",
    GEMINI_MODEL="test-model",
)
sys.modules.setdefault("app.core.config", config_stub)

genai_stub = types.ModuleType("app.core.genai_client")


class _DummyGenAIModel:
    def __init__(self, *args, **kwargs):
        pass


genai_stub.GenAIModel = _DummyGenAIModel
sys.modules.setdefault("app.core.genai_client", genai_stub)

google_stub = types.ModuleType("google")
google_stub.genai = types.SimpleNamespace(Client=lambda api_key: object())
sys.modules.setdefault("google", google_stub)

from app.repositories.llm_repository import LLMRepository
from app.repositories.search_repository import SearchRepository


class FakeRow:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return dict(self._data)


class FakeILoc:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, index):
        return FakeRow(self._rows[index])


class FakeDataFrame:
    def __init__(self, rows):
        self._rows = rows
        self.iloc = FakeILoc(rows)

    def __len__(self):
        return len(self._rows)


class SearchDiversityTests(unittest.TestCase):
    def setUp(self):
        rows = [
            self._row("best", "hero portrait cinematic lighting", 7.0, 30, "DPM++ 2M KARRAS"),
            self._row("near_1", "hero portrait warm grade", 7.0, 30, "DPM++ 2M KARRAS"),
            self._row("near_2", "hero portrait studio composition", 6.5, 28, "DPM++ 2M KARRAS"),
            self._row("near_3", "hero portrait fashion editorial", 6.0, 27, "DPM++ 2M KARRAS"),
            self._row("mid_1", "hero portrait dramatic shadows", 5.5, 26, "EULER A"),
            self._row("mid_2", "hero portrait painterly texture", 5.0, 24, "EULER A"),
            self._row("far_1", "abstract architecture blueprint", 4.5, 22, "DDIM"),
            self._row("far_2", "macro insect documentary photo", 4.0, 20, "DDIM"),
        ]
        pbo_space = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.02],
                [0.94, 0.33],
                [0.90, 0.42],
                [0.76, 0.64],
                [0.67, 0.74],
                [-0.2, 0.98],
                [-1.0, 0.0],
            ],
            dtype=float,
        )
        repo = SearchRepository.__new__(SearchRepository)
        repo.search_engine = types.SimpleNamespace(
            df=FakeDataFrame(rows),
            pbo_space=pbo_space,
            search_top_k=self._make_search_top_k(rows, pbo_space),
        )
        self.repo = repo

    @staticmethod
    def _row(identifier, prompt, cfgscale, steps, sampler):
        return {
            "id": identifier,
            "prompt": prompt,
            "style": "cinematic",
            "model": "SDXL",
            "sampler": sampler,
            "cfgscale": cfgscale,
            "steps": steps,
            "clipskip": 2.0,
            "loras": "film_style",
            "negative_prompt": "blurry, low quality",
            "image_url": "",
            "local_path": "",
        }

    @staticmethod
    def _make_search_top_k(rows, pbo_space):
        def search_top_k(query_index=None, query_vector=None, top_k=5):
            if query_vector is None:
                query_vector = pbo_space[query_index]
            norms = np.linalg.norm(pbo_space, axis=1) * np.linalg.norm(query_vector)
            similarities = np.divide(
                pbo_space @ query_vector,
                norms,
                out=np.zeros(len(pbo_space), dtype=float),
                where=norms != 0,
            )
            distances = 1.0 - np.clip(similarities, -1.0, 1.0)
            indices = np.argsort(distances)[:top_k]
            results = []
            for idx in indices:
                item = dict(rows[idx])
                item["index"] = int(idx)
                item["distance"] = round(float(distances[idx]), 4)
                results.append(item)
            return results

        return search_top_k

    def test_search_diverse_references_returns_fixed_role_counts(self):
        bundle = self.repo.search_diverse_references(query_index=0)

        self.assertEqual(bundle["counts"]["best"], 1)
        self.assertEqual(bundle["counts"]["complementary_knn"], 2)
        self.assertEqual(bundle["counts"]["exploratory"], 2)
        self.assertEqual(bundle["counts"]["counterexample"], 1)

        roles = [item["role"] for item in bundle["references"]]
        self.assertEqual(roles.count("best"), 1)
        self.assertEqual(roles.count("complementary_knn"), 2)
        self.assertEqual(roles.count("exploratory"), 2)
        self.assertEqual(roles.count("counterexample"), 1)

    def test_search_diverse_references_keeps_unique_indices_and_expected_order(self):
        bundle = self.repo.search_diverse_references(query_index=0)
        references = bundle["references"]

        self.assertEqual(references[0]["index"], 0)
        self.assertEqual(references[0]["role"], "best")

        seen_indices = [item["index"] for item in references]
        self.assertEqual(len(seen_indices), len(set(seen_indices)))

        complementary_distances = [item["distance"] for item in references if item["role"] == "complementary_knn"]
        exploratory_distances = [item["distance"] for item in references if item["role"] == "exploratory"]
        counterexample_distance = next(item["distance"] for item in references if item["role"] == "counterexample")

        self.assertTrue(all(distance < min(exploratory_distances) for distance in complementary_distances))
        self.assertTrue(all(distance < counterexample_distance for distance in exploratory_distances))


class LLMRepositoryPromptTests(unittest.TestCase):
    def test_generation_prompt_separates_positive_and_negative_references(self):
        bundle = {
            "query_index": 0,
            "counts": {"best": 1, "complementary_knn": 2, "exploratory": 2, "counterexample": 1},
            "selection_summary": {"strategy": "fixed_1_2_2_1"},
            "references": [
                {"id": "a", "role": "best", "prompt": "hero portrait", "distance": 0.0},
                {"id": "b", "role": "complementary_knn", "prompt": "warm cinematic portrait", "distance": 0.01},
                {"id": "c", "role": "exploratory", "prompt": "painterly portrait", "distance": 0.08},
                {"id": "x", "role": "counterexample", "prompt": "macro insect photo", "distance": 1.5},
            ],
        }

        message = LLMRepository._build_generation_user_message(
            bundle,
            "Create a premium cinematic portrait.",
            previous_output='{"bad": }',
            validation_error="JSONDecodeError",
        )

        self.assertIn("<positive_references>", message)
        self.assertIn("<negative_reference>", message)
        self.assertIn("macro insect photo", message)
        self.assertIn("Use the counterexample only to understand what to avoid", message)
        self.assertIn("<validation_error>", message)
        self.assertIn("<previous_invalid_output>", message)


if __name__ == "__main__":
    unittest.main()
