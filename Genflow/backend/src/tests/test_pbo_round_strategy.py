import os
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


pandas_stub = types.ModuleType("pandas")
pandas_stub.DataFrame = object
pandas_stub.isna = lambda value: False
sys.modules.setdefault("pandas", pandas_stub)

requests_stub = types.ModuleType("requests")
requests_stub.HTTPError = Exception
sys.modules.setdefault("requests", requests_stub)

pil_stub = types.ModuleType("PIL")
pil_stub.Image = object
sys.modules.setdefault("PIL", pil_stub)

sklearn_stub = types.ModuleType("sklearn")
sys.modules.setdefault("sklearn", sklearn_stub)

preprocessing_stub = types.ModuleType("sklearn.preprocessing")
preprocessing_stub.StandardScaler = object
preprocessing_stub.OneHotEncoder = object
sys.modules.setdefault("sklearn.preprocessing", preprocessing_stub)

decomposition_stub = types.ModuleType("sklearn.decomposition")
decomposition_stub.PCA = object
sys.modules.setdefault("sklearn.decomposition", decomposition_stub)

neighbors_stub = types.ModuleType("sklearn.neighbors")
neighbors_stub.NearestNeighbors = object
sys.modules.setdefault("sklearn.neighbors", neighbors_stub)

gaussian_process_stub = types.ModuleType("sklearn.gaussian_process")
gaussian_process_stub.GaussianProcessRegressor = object
sys.modules.setdefault("sklearn.gaussian_process", gaussian_process_stub)

kernels_stub = types.ModuleType("sklearn.gaussian_process.kernels")


class _FakeMatern:
    def __init__(self, length_scale=1.0, nu=1.5):
        self.length_scale = length_scale
        self.nu = nu

    def __rmul__(self, other):
        return self


kernels_stub.Matern = _FakeMatern
sys.modules.setdefault("sklearn.gaussian_process.kernels", kernels_stub)


from app.modules.embedding_v4 import ImageEmbeddingSearch


class _FakeGP:
    def __init__(self, kernel=None, alpha=None, n_restarts_optimizer=None):
        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer

    def fit(self, X, y):
        self.last_fit = (X, y)
        return self

    def predict(self, X, return_std=False):
        mu = np.array([0.98, 0.96, 0.92, 0.89, 0.87, 0.85, 0.50, 0.45, 0.40, 0.35], dtype=float)
        sigma = np.array([0.05, 0.12, 0.09, 0.16, 0.15, 0.14, 0.20, 0.18, 0.95, 0.90], dtype=float)
        if return_std:
            return mu, sigma
        return mu


class PBORoundStrategyTests(unittest.TestCase):
    def setUp(self):
        self.search = ImageEmbeddingSearch.__new__(ImageEmbeddingSearch)
        self.search.pbo_space = np.array(
            [
                [0.00, 0.00],
                [0.02, 0.00],
                [0.04, 0.00],
                [0.06, 0.00],
                [0.08, 0.00],
                [0.10, 0.00],
                [1.00, 0.00],
                [1.10, 0.00],
                [2.00, 0.00],
                [2.20, 0.00],
            ],
            dtype=float,
        )

    @patch("app.modules.embedding_v4.GaussianProcessRegressor", _FakeGP)
    def test_run_pbo_round_returns_six_unique_candidates_with_remote_explore(self):
        candidates = self.search.run_pbo_round(
            X_train=[self.search.pbo_space[0], self.search.pbo_space[1]],
            y_train=[1.0, 0.5],
            selected_indices=[0, 1],
            batch_size=6,
            consecutive_skips=0,
        )

        self.assertEqual(len(candidates), 6)
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertIn(8, candidates)

    @patch("app.modules.embedding_v4.GaussianProcessRegressor", _FakeGP)
    def test_run_pbo_round_keeps_six_candidates_when_local_pool_is_small(self):
        candidates = self.search.run_pbo_round(
            X_train=[self.search.pbo_space[0], self.search.pbo_space[9]],
            y_train=[1.0, 0.0],
            selected_indices=[0],
            batch_size=6,
            consecutive_skips=2,
        )

        self.assertEqual(len(candidates), 6)
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertIn(8, candidates)

    def test_run_pbo_round_random_fallback_respects_batch_size(self):
        np.random.seed(0)
        candidates = self.search.run_pbo_round(
            X_train=[self.search.pbo_space[0]],
            y_train=[1.0],
            selected_indices=[0],
            batch_size=6,
            consecutive_skips=0,
        )

        self.assertEqual(len(candidates), 6)
        self.assertEqual(len(candidates), len(set(candidates)))


if __name__ == "__main__":
    unittest.main()
