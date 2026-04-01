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
        supabase_key = os.getenv("SUPABASE_KEY", "").strip()
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
            print(f"Supabase table '{table}' returned no records. Falling back to local embedding.", flush=True)
            return False

        vectors = []
        metas = []
        for row in records:
            vec = row.get("pbo_embedding")
            meta = row.get("metadata")
            if vec is None or meta is None:
                continue

            if isinstance(vec, str):
                vec = json.loads(vec)
            if isinstance(meta, str):
                meta = json.loads(meta)
            if not isinstance(meta, dict):
                meta = {"metadata": meta}

            if "id" not in meta and "id" in row:
                meta["id"] = str(row["id"])

            vectors.append(vec)
            metas.append(meta)

        if not vectors:
            raise ValueError(f"Supabase table '{table}' has no usable (pbo_embedding, metadata) rows")

        pbo_space = np.asarray(vectors, dtype=float)
        df = pd.DataFrame(metas)
        df = self._clean_df(df)

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
                "select": "id,pbo_embedding,metadata",
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
        if self.text_model is None or self.pca_text is None or self.pca_final is None or self.text_scaler is None:
            raise RuntimeError("Query projection requires local embedding/PCA state. Precomputed-only mode cannot transform new prompts.")

        profile = self._default_query_profile()
        cfgscale = profile["cfgscale"] if cfgscale is None else cfgscale
        steps = profile["steps"] if steps is None else steps
        clipskip = profile["clipskip"] if clipskip is None else clipskip
        sampler = profile["sampler"] if sampler is None else sampler
        model = profile["model"] if model is None else model

        text_features = self.text_model.encode([prompt], show_progress_bar=False)
        text_reduced = self.pca_text.transform(text_features)
        text_reduced_scaled = self.text_scaler.transform(text_reduced)

        numeric = np.array([[cfgscale, steps, clipskip]], dtype=float)
        num_features = self.scaler.transform(numeric)

        sampler_features = self.encoder_sampler.transform(pd.DataFrame({"sampler": [str(sampler).upper()]}))
        model_features = self.encoder_model.transform(pd.DataFrame({"model": [str(model)]}))

        TEXT_WEIGHT = 4.0
        MODEL_WEIGHT = 3.0
        PARAM_WEIGHT = 0.5
        SAMPLER_WEIGHT = 0.5

        combined_features = np.hstack([
            text_reduced_scaled * TEXT_WEIGHT,
            model_features * MODEL_WEIGHT,
            num_features * PARAM_WEIGHT,
            sampler_features * SAMPLER_WEIGHT,
        ])
        return self.pca_final.transform(combined_features)[0]

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

    def run_pbo_round(self, X_train, y_train, batch_size=4, consecutive_skips=0):
        # This can be used for iterative rounds if needed by a websocket or multi-step API
        kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5)
        
        kappa_base = 1.5
        cur_kappa = kappa_base + (2.5 * consecutive_skips)
        
        if len(X_train) < 2:
            candidate_indices = np.random.choice(len(self.pbo_space), batch_size, replace=False).tolist()
        else:
            gp.fit(np.array(X_train), np.array(y_train))
            mu, sigma = gp.predict(self.pbo_space, return_std=True)
            
            # Stagnation detection (Global Exploration)
            if consecutive_skips >= 2:
                print(f">>> Stagnation detected: Activating global exploration (Escaping local optima)...", flush=True)
                pool_size = min(len(self.pbo_space), 100)
                sigma_pool_indices = np.argsort(sigma)[-pool_size:].tolist()
                np.random.shuffle(sigma_pool_indices)
                candidate_indices = sigma_pool_indices[:batch_size]
            else:
                ucb = mu + cur_kappa * sigma
                
                # Dynamic pool size based on skips
                base_pool_size = max(50, len(self.pbo_space) // 3)
                top_n = min(len(self.pbo_space), base_pool_size + 50 * consecutive_skips)
                top_k_indices = np.argsort(ucb)[-top_n:].tolist()
                
                candidate_indices = [top_k_indices.pop(-1)]
                
                while len(candidate_indices) < batch_size and top_k_indices:
                    max_min_dist = -1
                    best_idx_in_pool = -1
                    for pool_idx in top_k_indices:
                        dist_to_selected = min([np.linalg.norm(self.pbo_space[pool_idx] - self.pbo_space[s]) for s in candidate_indices])
                        if dist_to_selected > max_min_dist:
                            max_min_dist = dist_to_selected
                            best_idx_in_pool = pool_idx
                    candidate_indices.append(best_idx_in_pool)
                    top_k_indices.remove(best_idx_in_pool)
        
        return candidate_indices
