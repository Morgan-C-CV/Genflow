import json
import os
import numpy as np
import pandas as pd
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import re
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_paths = [
    os.path.join(base_dir, "../../backend/src/.env"),
    os.path.join(base_dir, ".env"),
    os.path.join(os.getcwd(), ".env")
]

for dp in dotenv_paths:
    if os.path.exists(dp):
        load_dotenv(dp)
        break

warnings.filterwarnings('ignore')

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: SUPABASE_URL and SUPABASE_KEY must be set in environment variables or .env file.")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_and_clean_data(file_path):
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
        df['loras_extracted'] = df['full_metadata_string'].apply(extract_loras)
    else:
        df['model'] = 'UNKNOWN'
        df['loras_extracted'] = ''
        
    df['style'] = df.get('style', pd.Series([''] * len(df))).fillna('')
    df['lora'] = df.get('lora', pd.Series(['none'] * len(df))).fillna('none')
    df['prompt'] = df['prompt'].fillna('')
    df['enhanced_prompt'] = df['prompt'] + " " + df['loras_extracted']
    df['negative_prompt'] = df['negative_prompt'].fillna('')
    df['clipskip'] = df.get('clipskip', pd.Series([2] * len(df))).fillna(2).astype(float)
    df['cfgscale'] = pd.to_numeric(df['cfgscale'], errors='coerce').fillna(7.0)
    df['steps'] = pd.to_numeric(df['steps'], errors='coerce').fillna(20)
    df['sampler'] = df['sampler'].fillna('UNKNOWN').str.upper()
    
    # Deduplicate by ID
    df = df.drop_duplicates(subset=['id'])
    print(f"Data cleaned. Records after deduplication: {len(df)}")
    
    return df

def build_embeddings(df):
    print("Generating CLIP embeddings...")
    model = SentenceTransformer('clip-ViT-B-32')
    prompt_embeddings = model.encode(df['enhanced_prompt'].tolist(), show_progress_bar=True)
    style_embeddings = model.encode(df['style'].tolist(), show_progress_bar=True)
    return prompt_embeddings, style_embeddings

def calculate_pca(df, prompt_embeddings, style_embeddings):
    print("Calculating PCA components (v5 logic)...")
    # Standardize and One-Hot encode as per v5.py
    scaler = StandardScaler()
    num_features = scaler.fit_transform(df[['cfgscale', 'steps', 'clipskip']])
    
    enc_sampler = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sampler_features = enc_sampler.fit_transform(df[['sampler']])
    
    enc_model = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    model_features = enc_model.fit_transform(df[['model']])
    
    # PCA Stage 1
    pca_text = PCA(n_components=min(20, prompt_embeddings.shape[0], prompt_embeddings.shape[1]))
    text_reduced = StandardScaler().fit_transform(pca_text.fit_transform(prompt_embeddings))
    
    pca_style = PCA(n_components=min(20, style_embeddings.shape[0], style_embeddings.shape[1]))
    style_reduced = StandardScaler().fit_transform(pca_style.fit_transform(style_embeddings))
    
    # Combined features with weights
    combined = np.hstack([
        text_reduced * 4.0, 
        style_reduced * 3.0,
        model_features * 3.0,
        num_features * 0.5, 
        sampler_features * 0.5
    ])
    
    # PCA Stage 2
    pca_final = PCA(n_components=min(8, combined.shape[0], combined.shape[1]))
    pbo_space = pca_final.fit_transform(combined)
    
    return pbo_space

def ingest_to_supabase(df, prompt_embs, style_embs, pbo_embs, model_features, sampler_features, num_features):
    print(f"Ingesting {len(df)} records to Supabase...")
    
    # Concatenate original features: prompt(512) + style(512) + model_oh(?) + sampler_oh(?) + num(3)
    # The dimension depends on the OneHot encoding. We'll pad to 1200.
    TARGET_DIM = 1200
    
    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Build original combined vector
        original_combined = np.concatenate([
            prompt_embs[i], 
            style_embs[i],
            model_features[i],
            sampler_features[i],
            num_features[i]
        ])
        
        # Pad to TARGET_DIM
        if len(original_combined) < TARGET_DIM:
            original_combined = np.pad(original_combined, (0, TARGET_DIM - len(original_combined)))
        else:
            original_combined = original_combined[:TARGET_DIM]
            
        record = {
            "id": int(row['id']),
            "prompt": row['prompt'],
            "style": row['style'],
            "model": row['model'],
            "sampler": row['sampler'],
            "cfgscale": float(row['cfgscale']),
            "steps": int(row['steps']),
            "clipskip": float(row['clipskip']),
            "original_embedding": original_combined.tolist(),
            "pbo_embedding": pbo_embs[i].tolist(),
            "metadata": {
                "loras": row['loras_extracted'],
                "negative_prompt": row['negative_prompt'],
                "image_url": row.get('image_url', ''),
                "local_path": row.get('local_path', '')
            }
        }
        records.append(record)
        
        # Batch insert every 50 records (larger payloads due to vector size)
        if len(records) >= 50:
            supabase.table("image_embeddings_v4").upsert(records).execute()
            records = []
            print(f"Progress: {i+1}/{len(df)}")
            
    if records:
        supabase.table("image_embeddings_v4").upsert(records).execute()
    
    print("Ingestion complete!")

if __name__ == "__main__":
    # Candidate paths for metadata.json
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, "../../../spider/civitai_gallery/metadata.json"),
        os.path.join(base_dir, "../../../spider/civitai_gallery_res/metadata.json"),
        os.path.join(base_dir, "../metadata.json"),
        "spider/civitai_gallery/metadata.json",
        "metadata.json"
    ]
    
    meta_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            meta_path = path
            break
            
    if not meta_path:
        print("Error: metadata.json not found in candidate paths.")
        exit(1)
        
    print(f"Using metadata: {meta_path}")
    df = load_and_clean_data(meta_path)
    prompt_embs, style_embs = build_embeddings(df)
    
    # Extract features for concatenation
    scaler = StandardScaler()
    num_features = scaler.fit_transform(df[['cfgscale', 'steps', 'clipskip']])
    enc_sampler = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sampler_features = enc_sampler.fit_transform(df[['sampler']])
    enc_model = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    model_features = enc_model.fit_transform(df[['model']])
    
    pbo_embs = calculate_pca(df, prompt_embs, style_embs)
    ingest_to_supabase(df, prompt_embs, style_embs, pbo_embs, model_features, sampler_features, num_features)
