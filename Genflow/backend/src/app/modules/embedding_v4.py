import json
import sys
import os
import re
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from requests import HTTPError
import warnings
warnings.filterwarnings('ignore')

class ImageEmbeddingSearch:
    def __init__(self, metadata_path, gallery_dir):
        self.metadata_path = metadata_path
        self.gallery_dir = gallery_dir
        self.df = None
        self.pbo_space = None
        self.text_features = None
        self.num_features = None
        self.sampler_features = None
        self.model_features = None
        self.scaler = None
        self.encoder_sampler = None
        self.encoder_model = None
        self.text_model = None
        self.text_scaler = None
        self.pca_text = None
        self.pca_final = None
        self._projection_artifacts = None
        self._projection_text_model = None
        self._load_and_initialize()

    def _load_and_initialize(self):
        print(f"Initializing ImageEmbeddingSearch with: {self.metadata_path}", flush=True)
        if self._try_load_precomputed_pbo_space():
            print("Initialization complete (precomputed PBO space).", flush=True)
            return

        self.df = self.load_and_clean_data(self.metadata_path)
        self.text_features, self.num_features, self.sampler_features, self.model_features, self.scaler, self.encoder_sampler, self.encoder_model, self.text_model = self.build_features(self.df)
        self.pbo_space, self.pca_text, self.pca_final = self.perform_two_stage_pca(self.text_features, self.num_features, self.sampler_features, self.model_features, final_dim=8)
        print("Initialization complete.", flush=True)

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def extract_model(meta_str):
            if pd.isna(meta_str):
                return 'UNKNOWN'
            match = re.search(r'Model:\s*([^,]+)', str(meta_str))
            return match.group(1).strip() if match else 'UNKNOWN'

        def extract_loras(meta_str):
            if pd.isna(meta_str):
                return ''
            loras = re.findall(r'<lora:([^:]+):', str(meta_str))
            return ' '.join(loras)

        if 'full_metadata_string' in df.columns:
            df['model'] = df['full_metadata_string'].apply(extract_model)
            df['loras'] = df['full_metadata_string'].apply(extract_loras)
        else:
            if 'model' not in df.columns:
                df['model'] = 'UNKNOWN'
            if 'loras' not in df.columns:
                df['loras'] = ''

        if 'prompt' not in df.columns:
            df['prompt'] = ''
        df['prompt'] = df['prompt'].fillna('')
        df['enhanced_prompt'] = df['prompt'] + " " + df['loras'].fillna('')

        if 'negative_prompt' not in df.columns:
            df['negative_prompt'] = ''
        df['negative_prompt'] = df['negative_prompt'].fillna('')

        if 'clipskip' not in df.columns:
            df['clipskip'] = 2
        df['clipskip'] = df['clipskip'].fillna(2).astype(float)

        if 'cfgscale' not in df.columns:
            df['cfgscale'] = np.nan
        df['cfgscale'] = pd.to_numeric(df['cfgscale'], errors='coerce')
        if df['cfgscale'].isna().all():
            df['cfgscale'] = 7.0
        else:
            df['cfgscale'] = df['cfgscale'].fillna(df['cfgscale'].median())

        if 'steps' not in df.columns:
            df['steps'] = np.nan
        df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
        if df['steps'].isna().all():
            df['steps'] = 20
        else:
            df['steps'] = df['steps'].fillna(df['steps'].median())

        if 'sampler' not in df.columns:
            df['sampler'] = 'UNKNOWN'
        df['sampler'] = df['sampler'].fillna('UNKNOWN').astype(str).str.upper()

        if 'id' not in df.columns:
            df['id'] = df.index.astype(str)
        df['id'] = df['id'].astype(str)

        return df

    def load_and_clean_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        return self._clean_df(df)

    def _try_load_precomputed_pbo_space(self) -> bool:
        supabase_url = os.getenv("SUPABASE_URL", "").strip()
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        supabase_key = service_role_key or os.getenv("SUPABASE_KEY", "").strip()
        table = os.getenv("SUPABASE_PBO_TABLE", "image_embeddings_v4").strip()
        schema = os.getenv("SUPABASE_SCHEMA", "").strip()
        page_size_env = os.getenv("SUPABASE_PAGE_SIZE", "").strip()
        page_size = int(page_size_env) if page_size_env.isdigit() else 1000
        strict_env = os.getenv("SUPABASE_STRICT", "").strip().lower()
        strict = strict_env in {"1", "true", "yes", "y"}

        if not supabase_url or not supabase_key:
            return False

        try:
            records = self._fetch_supabase_embeddings(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                table_name=table,
                schema=schema,
                page_size=page_size,
            )
        except Exception as e:
            if strict:
                raise
            print(f"Failed to load precomputed PBO space from Supabase ({e}). Falling back to local embedding.", flush=True)
            return False

        if not records:
            print(
                f"Supabase table '{table}' returned no records with current key. "
                f"url={supabase_url.rstrip('/')} schema={(schema or 'public')} "
                f"key_type={'service_role' if service_role_key else 'publishable_or_anon'}. "
                "Possible causes: RLS blocks SELECT for this key, wrong project URL, or empty table. "
                "Falling back to local embedding.",
                flush=True,
            )
            return False

        artifact_table = os.getenv("SUPABASE_ARTIFACT_TABLE", "embedding_projection_artifacts").strip()
        model_version = os.getenv("SUPABASE_MODEL_VERSION", "v4").strip() or "v4"

        vectors = []
        rows = []
        for row in records:
            vec = row.get("pbo_embedding")
            if vec is None:
                continue
            if isinstance(vec, str):
                vec = json.loads(vec)

            meta = row.get("metadata")
            if isinstance(meta, str):
                meta = json.loads(meta)
            if not isinstance(meta, dict):
                meta = {}

            rows.append(
                {
                    "id": str(row.get("id", meta.get("id", ""))),
                    "prompt": str(row.get("prompt", meta.get("prompt", "")) or ""),
                    "style": str(row.get("style", meta.get("style", "")) or ""),
                    "model": str(row.get("model", meta.get("model", "UNKNOWN")) or "UNKNOWN"),
                    "sampler": str(row.get("sampler", meta.get("sampler", "UNKNOWN")) or "UNKNOWN"),
                    "cfgscale": row.get("cfgscale", meta.get("cfgscale", 7.0)),
                    "steps": row.get("steps", meta.get("steps", 20)),
                    "clipskip": row.get("clipskip", meta.get("clipskip", 2.0)),
                    "loras": str(meta.get("loras", "") or ""),
                    "negative_prompt": str(meta.get("negative_prompt", "") or ""),
                    "image_url": str(meta.get("image_url", "") or ""),
                    "local_path": str(meta.get("local_path", "") or ""),
                }
            )
            vectors.append(vec)

        if not vectors:
            raise ValueError(f"Supabase table '{table}' has no usable pbo_embedding rows")

        pbo_space = np.asarray(vectors, dtype=float)
        df = self._clean_df(pd.DataFrame(rows))

        if len(df) != len(pbo_space):
            raise ValueError(f"Supabase data length mismatch: df={len(df)} pbo_space={len(pbo_space)}")

        self.df = df.reset_index(drop=True)
        self.pbo_space = pbo_space
        self.text_features = None
        self.num_features = None
        self.sampler_features = None
        self.model_features = None
        self.scaler = None
        self.encoder_sampler = None
        self.encoder_model = None
        self.text_model = None
        self.text_scaler = None
        self.pca_text = None
        self.pca_final = None

        try:
            self._projection_artifacts = self._fetch_projection_artifacts(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                table_name=artifact_table,
                model_version=model_version,
                schema=schema,
            )
        except Exception as e:
            if strict:
                raise
            self._projection_artifacts = None
            print(f"Projection artifacts load failed ({e}). Query projection may use fallback path.", flush=True)

        print(f"Loaded precomputed PBO space from Supabase table '{table}' ({len(self.df)} rows).", flush=True)
        return True

    def _fetch_supabase_embeddings(self, supabase_url: str, supabase_key: str, table_name: str, schema: str, page_size: int):
        base_url = supabase_url.rstrip("/")
        url = f"{base_url}/rest/v1/{table_name}"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Accept": "application/json",
        }
        if schema:
            headers["Accept-Profile"] = schema

        all_rows = []
        offset = 0
        while True:
            params = {
                "select": "id,prompt,style,model,sampler,cfgscale,steps,clipskip,pbo_embedding,metadata",
                "order": "id.asc",
                "limit": str(page_size),
                "offset": str(offset),
            }
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            try:
                resp.raise_for_status()
            except HTTPError as e:
                detail = ""
                try:
                    detail = resp.text
                except Exception:
                    detail = ""

                schema_hint = f" (schema={schema})" if schema else ""
                raise ValueError(
                    f"Supabase REST fetch failed: status={resp.status_code} table={table_name}{schema_hint} url={url} detail={detail}"
                ) from e
            batch = resp.json()
            if not batch:
                break
            all_rows.extend(batch)
            offset += len(batch)
            if len(batch) < page_size:
                break

        return all_rows

    def _fetch_projection_artifacts(self, supabase_url: str, supabase_key: str, table_name: str, model_version: str, schema: str):
        base_url = supabase_url.rstrip("/")
        url = f"{base_url}/rest/v1/{table_name}"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Accept": "application/json",
        }
        if schema:
            headers["Accept-Profile"] = schema
        params = {
            "select": "model_version,artifacts",
            "model_version": f"eq.{model_version}",
            "limit": "1",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        artifacts = rows[0].get("artifacts")
        if isinstance(artifacts, str):
            artifacts = json.loads(artifacts)
        return artifacts if isinstance(artifacts, dict) else None

    def build_features(self, df):
        from sentence_transformers import SentenceTransformer

        text_model = SentenceTransformer('clip-ViT-B-32')
        text_features = text_model.encode(df['enhanced_prompt'].tolist(), show_progress_bar=False)

        scaler = StandardScaler()
        num_features = scaler.fit_transform(df[['cfgscale', 'steps', 'clipskip']])
        
        encoder_sampler = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        sampler_features = encoder_sampler.fit_transform(df[['sampler']])
        
        encoder_model = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        model_features = encoder_model.fit_transform(df[['model']])
        
        return text_features, num_features, sampler_features, model_features, scaler, encoder_sampler, encoder_model, text_model

    def perform_two_stage_pca(self, text_features, num_features, sampler_features, model_features, final_dim=8):
        n_text_components = min(20, text_features.shape[0], text_features.shape[1])
        pca_text = PCA(n_components=n_text_components)
        text_reduced = pca_text.fit_transform(text_features)
        
        scaler_text = StandardScaler()
        text_reduced_scaled = scaler_text.fit_transform(text_reduced)
        self.text_scaler = scaler_text
        
        TEXT_WEIGHT = 4.0
        MODEL_WEIGHT = 3.0
        PARAM_WEIGHT = 0.5
        SAMPLER_WEIGHT = 0.5
        
        combined_features = np.hstack([
            text_reduced_scaled * TEXT_WEIGHT, 
            model_features * MODEL_WEIGHT,
            num_features * PARAM_WEIGHT, 
            sampler_features * SAMPLER_WEIGHT
        ])
        
        n_final_components = min(final_dim, combined_features.shape[0], combined_features.shape[1])
        pca_final = PCA(n_components=n_final_components)
        pbo_space = pca_final.fit_transform(combined_features)
        return pbo_space, pca_text, pca_final

    def _default_query_profile(self):
        model_mode = "UNKNOWN"
        sampler_mode = "DPM++ 2M KARRAS"
        if self.df is not None and len(self.df) > 0:
            if "model" in self.df.columns and not self.df["model"].empty:
                try:
                    model_mode = str(self.df["model"].mode().iloc[0])
                except Exception:
                    model_mode = str(self.df["model"].iloc[0])
            if "sampler" in self.df.columns and not self.df["sampler"].empty:
                try:
                    sampler_mode = str(self.df["sampler"].mode().iloc[0])
                except Exception:
                    sampler_mode = str(self.df["sampler"].iloc[0])
        return {
            "cfgscale": float(self.df["cfgscale"].median()) if self.df is not None and "cfgscale" in self.df.columns else 5.0,
            "steps": float(self.df["steps"].median()) if self.df is not None and "steps" in self.df.columns else 30.0,
            "clipskip": float(self.df["clipskip"].median()) if self.df is not None and "clipskip" in self.df.columns else 2.0,
            "sampler": sampler_mode,
            "model": model_mode,
        }

    def transform_query_to_pbo(self, prompt: str, cfgscale=None, steps=None, clipskip=None, sampler=None, model=None):
        profile = self._default_query_profile()
        cfgscale = profile["cfgscale"] if cfgscale is None else cfgscale
        steps = profile["steps"] if steps is None else steps
        clipskip = profile["clipskip"] if clipskip is None else clipskip
        sampler = profile["sampler"] if sampler is None else sampler
        model = profile["model"] if model is None else model

        if self.text_model is not None and self.pca_text is not None and self.pca_final is not None and self.text_scaler is not None:
            text_features = self.text_model.encode([prompt], show_progress_bar=False)
            text_reduced = self.pca_text.transform(text_features)
            text_reduced_scaled = self.text_scaler.transform(text_reduced)
            numeric = np.array([[cfgscale, steps, clipskip]], dtype=float)
            num_features = self.scaler.transform(numeric)
            sampler_features = self.encoder_sampler.transform(pd.DataFrame({"sampler": [str(sampler).upper()]}))
            model_features = self.encoder_model.transform(pd.DataFrame({"model": [str(model)]}))
            combined_features = np.hstack([
                text_reduced_scaled * 4.0,
                model_features * 3.0,
                num_features * 0.5,
                sampler_features * 0.5,
            ])
            return self.pca_final.transform(combined_features)[0]

        if not self._projection_artifacts:
            raise RuntimeError("Query projection unavailable: local PCA state and projection artifacts are both missing.")

        if self._projection_text_model is None:
            from sentence_transformers import SentenceTransformer
            self._projection_text_model = SentenceTransformer(self._projection_artifacts.get("text_model_name", "clip-ViT-B-32"))

        art = self._projection_artifacts
        text_features = self._projection_text_model.encode([prompt], show_progress_bar=False)
        pca_text_components = np.asarray(art["pca_text_components"], dtype=float)
        pca_text_mean = np.asarray(art["pca_text_mean"], dtype=float)
        text_reduced = (text_features - pca_text_mean) @ pca_text_components.T

        scaler_text_mean = np.asarray(art["scaler_text_mean"], dtype=float)
        scaler_text_scale = np.asarray(art["scaler_text_scale"], dtype=float)
        scaler_text_scale = np.where(scaler_text_scale == 0, 1.0, scaler_text_scale)
        text_reduced_scaled = (text_reduced - scaler_text_mean) / scaler_text_scale

        scaler_num_mean = np.asarray(art["scaler_num_mean"], dtype=float)
        scaler_num_scale = np.asarray(art["scaler_num_scale"], dtype=float)
        scaler_num_scale = np.where(scaler_num_scale == 0, 1.0, scaler_num_scale)
        numeric = np.array([[cfgscale, steps, clipskip]], dtype=float)
        num_features = (numeric - scaler_num_mean) / scaler_num_scale

        sampler_categories = [str(x).upper() for x in art.get("sampler_categories", [])]
        model_categories = [str(x) for x in art.get("model_categories", [])]
        sampler_features = np.zeros((1, len(sampler_categories)), dtype=float)
        model_features = np.zeros((1, len(model_categories)), dtype=float)
        sampler_value = str(sampler).upper()
        model_value = str(model)
        if sampler_value in sampler_categories:
            sampler_features[0, sampler_categories.index(sampler_value)] = 1.0
        if model_value in model_categories:
            model_features[0, model_categories.index(model_value)] = 1.0

        weights = art.get("weights", {})
        text_w = float(weights.get("text", 4.0))
        model_w = float(weights.get("model", 3.0))
        param_w = float(weights.get("param", 0.5))
        sampler_w = float(weights.get("sampler", 0.5))

        combined_features = np.hstack([
            text_reduced_scaled * text_w,
            model_features * model_w,
            num_features * param_w,
            sampler_features * sampler_w,
        ])

        pca_final_components = np.asarray(art["pca_final_components"], dtype=float)
        pca_final_mean = np.asarray(art["pca_final_mean"], dtype=float)
        return ((combined_features - pca_final_mean) @ pca_final_components.T)[0]

    def search_top_k(self, query_index=None, query_vector=None, top_k=5):
        if query_vector is None and query_index is not None:
            query_vector = self.pbo_space[query_index].reshape(1, -1)
        elif query_vector is None:
            raise ValueError("Either query_index or query_vector must be provided")

        knn = NearestNeighbors(n_neighbors=top_k, metric='cosine')
        knn.fit(self.pbo_space)
        distances, indices = knn.kneighbors(query_vector)
        
        results = []
        for i, idx in enumerate(indices[0]):
            row = self.df.iloc[idx]
            res_dict = row.to_dict()
            res_dict["index"] = int(idx)
            res_dict["distance"] = round(float(distances[0][i]), 4)
            # Ensure all values are serializable
            for k, v in res_dict.items():
                if isinstance(v, (np.integer, np.floating)):
                    res_dict[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif pd.isna(v):
                    res_dict[k] = None
            results.append(res_dict)
        return results

    @staticmethod
    def _take_ranked_indices(pool_indices, scores, count, excluded=None):
        excluded = set() if excluded is None else set(excluded)
        ranked = sorted(pool_indices, key=lambda idx: float(scores[idx]), reverse=True)
        chosen = []
        for idx in ranked:
            if idx in excluded:
                continue
            chosen.append(int(idx))
            if len(chosen) >= count:
                break
        return chosen

    def run_pbo_round(self, X_train, y_train, selected_indices=None, batch_size=6, consecutive_skips=0):
        # This can be used for iterative rounds if needed by a websocket or multi-step API
        kappa_base = 1.5
        cur_kappa = kappa_base + (2.5 * consecutive_skips)
        selected_indices = list(selected_indices or [])
        
        if len(X_train) < 2:
            candidate_indices = np.random.choice(len(self.pbo_space), batch_size, replace=False).tolist()
        else:
            kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5)
            gp.fit(np.array(X_train), np.array(y_train))
            mu, sigma = gp.predict(self.pbo_space, return_std=True)

            if selected_indices:
                local_center = np.mean(self.pbo_space[selected_indices], axis=0)
            else:
                local_center = np.mean(np.array(X_train), axis=0)

            distances = np.linalg.norm(self.pbo_space - local_center, axis=1)
            local_cutoff = float(np.quantile(distances, 0.30))
            remote_cutoff = float(np.quantile(distances, 0.80))

            all_indices = list(range(len(self.pbo_space)))
            local_pool = [idx for idx, dist in enumerate(distances) if float(dist) <= local_cutoff]
            remote_pool = [idx for idx, dist in enumerate(distances) if float(dist) >= remote_cutoff]

            local_explore_score = mu + (0.6 * sigma)
            strong_explore_score = mu + (cur_kappa * sigma)
            remote_score = strong_explore_score if consecutive_skips >= 2 else sigma

            candidate_indices = []
            candidate_indices.extend(self._take_ranked_indices(local_pool, mu, 2))
            candidate_indices.extend(self._take_ranked_indices(local_pool, local_explore_score, 3, excluded=candidate_indices))
            candidate_indices.extend(self._take_ranked_indices(remote_pool, remote_score, 1, excluded=candidate_indices))

            if len(candidate_indices) < batch_size:
                candidate_indices.extend(
                    self._take_ranked_indices(local_pool, mu, batch_size - len(candidate_indices), excluded=candidate_indices)
                )
            if len(candidate_indices) < batch_size:
                candidate_indices.extend(
                    self._take_ranked_indices(local_pool, local_explore_score, batch_size - len(candidate_indices), excluded=candidate_indices)
                )
            if len(candidate_indices) < batch_size:
                candidate_indices.extend(
                    self._take_ranked_indices(remote_pool, remote_score, batch_size - len(candidate_indices), excluded=candidate_indices)
                )
            if len(candidate_indices) < batch_size:
                candidate_indices.extend(
                    self._take_ranked_indices(all_indices, strong_explore_score, batch_size - len(candidate_indices), excluded=candidate_indices)
                )

            candidate_indices = list(dict.fromkeys(candidate_indices))[:batch_size]
        
        return candidate_indices
