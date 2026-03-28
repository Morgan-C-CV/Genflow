import json
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from supabase import create_client, Client
import os

# Supabase configuration
SUPABASE_URL = "https://jxuyiqdunphnvevkhpsf.supabase.co"
SUPABASE_KEY = "sb_publishable_jYjPubXkv_TrVgzlQGMljw_cKjn8zx0"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_and_clean_data_v4(file_path):
    print(f"Loading data: {file_path}")
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
    df['clipskip'] = df.get('clipskip', pd.Series([2] * len(df))).fillna(2).astype(float)
    df['cfgscale'] = pd.to_numeric(df['cfgscale'], errors='coerce').fillna(7.0)
    df['steps'] = pd.to_numeric(df['steps'], errors='coerce').fillna(20)
    df['sampler'] = df['sampler'].fillna('UNKNOWN').str.upper()
    
    df = df.drop_duplicates(subset=['id'])
    return df

def build_features_v4(df):
    print("Generating CLIP embeddings (v4 style)...")
    model = SentenceTransformer('clip-ViT-B-32')
    text_features = model.encode(df['enhanced_prompt'].tolist(), show_progress_bar=True)
    
    scaler = StandardScaler()
    num_features = scaler.fit_transform(df[['cfgscale', 'steps', 'clipskip']])
    
    enc_sampler = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sampler_features = enc_sampler.fit_transform(df[['sampler']])
    
    enc_model = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    model_features = enc_model.fit_transform(df[['model']])
    
    return text_features, num_features, sampler_features, model_features

def calculate_pca_v4(text_features, num_features, sampler_features, model_features):
    print("Performing two-stage PCA (v4 logic)...")
    
    # Stage 1: Text compression to 20
    pca_text = PCA(n_components=min(20, text_features.shape[0], text_features.shape[1]))
    text_reduced = pca_text.fit_transform(text_features)
    text_reduced_scaled = StandardScaler().fit_transform(text_reduced)
    
    # Weights from v4.py
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
    
    pca_final = PCA(n_components=8)
    pbo_space = pca_final.fit_transform(combined_features)
    
    return pbo_space

def ingest_to_supabase_v4(df, text_features, model_features, sampler_features, num_features, pbo_embs):
    print(f"Ingesting {len(df)} records for v4 version...")
    
    TARGET_DIM = 700
    records = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Build original combined vector: prompt(512) + model + sampler + num(3)
        original_combined = np.concatenate([
            text_features[i],
            model_features[i],
            sampler_features[i],
            num_features[i]
        ])
        
        # Padding
        if len(original_combined) < TARGET_DIM:
            original_combined = np.pad(original_combined, (0, TARGET_DIM - len(original_combined)))
        else:
            original_combined = original_combined[:TARGET_DIM]
            
        record = {
            "id": int(row['id']),
            "prompt": row['prompt'],
            "model": row['model'],
            "sampler": row['sampler'],
            "cfgscale": float(row['cfgscale']),
            "steps": int(row['steps']),
            "clipskip": float(row['clipskip']),
            "original_embedding": original_combined.tolist(),
            "pbo_embedding": pbo_embs[i].tolist(),
            "metadata": {
                "loras": row['loras'],
                "negative_prompt": row['negative_prompt'],
                "image_url": row.get('image_url', ''),
                "local_path": row.get('local_path', '')
            }
        }
        records.append(record)
        
        if len(records) >= 50:
            supabase.table("image_embeddings_v4").upsert(records).execute()
            records = []
            
    if records:
        supabase.table("image_embeddings_v4").upsert(records).execute()
    
    print("Ingestion v4 complete!")

if __name__ == "__main__":
    meta_path = "/Users/mgccvmacair/Myproject/Academic/Genflow/Genflow/lib/metadata.json"
    df = load_and_clean_data_v4(meta_path)
    text_embs, num_feat, samp_feat, mod_feat = build_features_v4(df)
    pbo_embs = calculate_pca_v4(text_embs, num_feat, samp_feat, mod_feat)
    ingest_to_supabase_v4(df, text_embs, mod_feat, samp_feat, num_feat, pbo_embs)
