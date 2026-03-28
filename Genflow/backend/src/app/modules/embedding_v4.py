import json
import sys
import os
import re
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
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
        self.pca_text = None
        self.pca_final = None
        self._load_and_initialize()

    def _load_and_initialize(self):
        print(f"Initializing ImageEmbeddingSearch with: {self.metadata_path}", flush=True)
        self.df = self.load_and_clean_data(self.metadata_path)
        self.text_features, self.num_features, self.sampler_features, self.model_features, self.scaler, self.encoder_sampler = self.build_features(self.df)
        self.pbo_space, self.pca_text, self.pca_final = self.perform_two_stage_pca(self.text_features, self.num_features, self.sampler_features, self.model_features, final_dim=8)
        print("Initialization complete.", flush=True)

    def load_and_clean_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        def extract_model(meta_str):
            if pd.isna(meta_str): return 'UNKNOWN'
            match = re.search(r'Model:\s*([^,]+)', str(meta_str))
            return match.group(1).strip() if match else 'UNKNOWN'
            
        def extract_loras(meta_str):
            if pd.isna(meta_str): return ''
            loras = re.findall(r'<lora:([^:]+):', str(meta_str))
            return ' '.join(loras)

        if 'full_metadata_string' in df.columns:
            df['model'] = df['full_metadata_string'].apply(extract_model)
            df['loras'] = df['full_metadata_string'].apply(extract_loras)
        else:
            df['model'] = 'UNKNOWN'
            df['loras'] = ''
            
        df['prompt'] = df['prompt'].fillna('')
        df['enhanced_prompt'] = df['prompt'] + " " + df['loras']
        
        df['negative_prompt'] = df['negative_prompt'].fillna('')
        df['clipskip'] = df['clipskip'].fillna(2).astype(float)
        df['cfgscale'] = pd.to_numeric(df['cfgscale'], errors='coerce')
        df['cfgscale'] = df['cfgscale'].fillna(df['cfgscale'].median())
        df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
        df['steps'] = df['steps'].fillna(df['steps'].median())
        df['sampler'] = df['sampler'].fillna('UNKNOWN').str.upper()
        
        return df

    def build_features(self, df):
        model = SentenceTransformer('clip-ViT-B-32') 
        text_features = model.encode(df['enhanced_prompt'].tolist(), show_progress_bar=False)

        scaler = StandardScaler()
        num_features = scaler.fit_transform(df[['cfgscale', 'steps', 'clipskip']])
        
        encoder_sampler = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        sampler_features = encoder_sampler.fit_transform(df[['sampler']])
        
        encoder_model = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        model_features = encoder_model.fit_transform(df[['model']])
        
        return text_features, num_features, sampler_features, model_features, scaler, encoder_sampler

    def perform_two_stage_pca(self, text_features, num_features, sampler_features, model_features, final_dim=8):
        n_text_components = min(20, text_features.shape[0], text_features.shape[1])
        pca_text = PCA(n_components=n_text_components)
        text_reduced = pca_text.fit_transform(text_features)
        
        scaler_text = StandardScaler()
        text_reduced_scaled = scaler_text.fit_transform(text_reduced)
        
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