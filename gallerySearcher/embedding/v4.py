import json
import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def load_and_clean_data(file_path):
    print(f"Loading data: {file_path}", flush=True)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Raw data sample count: {len(df)}", flush=True)
    
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
    
    print("Data cleaning complete.", flush=True)
    return df

def build_features(df):
    print("\n[Phase 1] Extracting text features using CLIP model (this may take a few seconds)...", flush=True)
    model = SentenceTransformer('clip-ViT-B-32') 
    
    text_features = model.encode(df['enhanced_prompt'].tolist(), show_progress_bar=True)
    print(f"Raw text feature dimensions (Positive + LoRA): {text_features.shape}", flush=True)

    print("\n[Phase 2] Processing generation parameters (Standardization & One-Hot Encoding)...", flush=True)
    scaler = StandardScaler()
    num_features = scaler.fit_transform(df[['cfgscale', 'steps', 'clipskip']])
    
    encoder_sampler = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sampler_features = encoder_sampler.fit_transform(df[['sampler']])
    
    encoder_model = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    model_features = encoder_model.fit_transform(df[['model']])
    
    print(f"Numerical feature dimensions: {num_features.shape}", flush=True)
    print(f"Categorical feature (Sampler) dimensions: {sampler_features.shape}", flush=True)
    print(f"Categorical feature (Model) dimensions: {model_features.shape}", flush=True)
    
    return text_features, num_features, sampler_features, model_features, scaler, encoder_sampler

def perform_two_stage_pca(text_features, num_features, sampler_features, model_features, final_dim=8):
    print(f"\n[Phase 3] Performing two-stage PCA (Target dimensions: {final_dim})...", flush=True)
    n_text_components = min(20, text_features.shape[0], text_features.shape[1])
    pca_text = PCA(n_components=n_text_components)
    text_reduced = pca_text.fit_transform(text_features)
    print(f"Phase 1: Text features compressed to {n_text_components} dimensions", flush=True)
    
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
    explained_variance = sum(pca_final.explained_variance_ratio_) * 100
    print(f"Phase 2: Final PBO search space shape: {pbo_space.shape}", flush=True)
    print(f"Final {n_final_components} dimensions retained {explained_variance:.2f}% of variance information.", flush=True)
    return pbo_space, pca_text, pca_final

def simulate_pbo_retrieval(pbo_space, df, target_index=0, top_k=5):
    print("\n[Phase 4] Simulating KNN retrieval after PBO finds optimal point...", flush=True)
    predicted_optimal_point = pbo_space[target_index].reshape(1, -1) 
    
    knn = NearestNeighbors(n_neighbors=top_k, metric='cosine')
    knn.fit(pbo_space)
    distances, indices = knn.kneighbors(predicted_optimal_point)
    print(f"\n==== Top {top_k} Reference Metadata for LLM Agent ====\n", flush=True)
    top_metadata_for_llm = []
    for i, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        metadata_dict = {
            "distance": round(distances[0][i], 4),
            "id": row.get('id', 'N/A'),
            "prompt": row['prompt'][:100] + "..." if len(row['prompt']) > 100 else row['prompt'],
            "model": row.get('model', 'UNKNOWN'),
            "loras": row.get('loras', ''),
            "cfgscale": row['cfgscale'],
            "steps": row['steps'],
            "sampler": row['sampler']
        }
        top_metadata_for_llm.append(metadata_dict)
        print(f"Rank {i+1} (Cosine Distance: {metadata_dict['distance']}): ID {metadata_dict['id']}", flush=True)
        print(f"  - Model: {metadata_dict['model']} | LoRAs: {metadata_dict['loras']}", flush=True)
        print(f"  - Prompt: {metadata_dict['prompt']}", flush=True)
        print(f"  - CFG: {metadata_dict['cfgscale']} | Steps: {metadata_dict['steps']} | Sampler: {metadata_dict['sampler']}", flush=True)
        print("-" * 40, flush=True)
        
    print(f"\n[Final Display] Showing Top {top_k} matching images based on optimization results...", flush=True)
    display_images(df, indices[0].tolist(), title=f"Top {top_k} Recommended Images", filename="pbo_results_top5.png")
    return top_metadata_for_llm

def display_images(df, indices, title="Images", filename="pbo_display.png"):
    print(f"Preparing to display images: {title} (Indices: {indices})", flush=True)
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
    if n == 1: axes = [axes]
    fig.suptitle(title, fontsize=16)
    gallery_dir = '/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery'
    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        local_filename = row.get('local_path')
        img_url = row.get('image_url')
        try:
            img = None
            if local_filename:
                local_path = os.path.join(gallery_dir, local_filename)
                if os.path.exists(local_path): img = Image.open(local_path)
            if img is None and img_url:
                try:
                    response = requests.get(img_url, timeout=3)
                    img = Image.open(BytesIO(response.content))
                except: pass
            if img is None: raise FileNotFoundError(f"Cannot find image for ID {row.get('id')}")
            axes[i].imshow(img)
            axes[i].set_title(f"Idx: {idx}\nID: {row.get('id', 'N/A')}")
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Image Load Error\n{e}", ha='center', va='center', fontsize=8)
            axes[i].set_title(f"Idx: {idx} (Error)")
            axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Images saved to: {os.path.abspath(filename)}", flush=True)
    try:
        plt.show(block=False)
        plt.pause(0.5) 
        plt.close()
    except: pass

def run_pbo_loop(pbo_space, df, iterations=10, batch_size=4):
    print(f"\n[Phase 4] Starting PBO optimization process ({iterations} rounds, {batch_size} images per round)...", flush=True)
    kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5)
    X_train = []
    y_train = []
    
    kappa_base = 1.5
    consecutive_skips = 0
    
    current_iter = 0
    while current_iter < iterations:
        print(f"\n--- Round {current_iter+1} / {iterations} ---", flush=True)
        
        cur_kappa = kappa_base + (2.5 * consecutive_skips)
            
        if len(X_train) < 2:
            candidate_indices = np.random.choice(len(pbo_space), batch_size, replace=False).tolist()
        else:
            gp.fit(np.array(X_train), np.array(y_train))
            mu, sigma = gp.predict(pbo_space, return_std=True)
            ucb = mu + cur_kappa * sigma
            
            if consecutive_skips >= 2:
                print(f">>> Stagnation detected: Activating global exploration (Escaping local optima)...", flush=True)
                pool_size = min(len(pbo_space), 100)
                sigma_pool_indices = np.argsort(sigma)[-pool_size:].tolist()
                np.random.shuffle(sigma_pool_indices)
                candidate_indices = sigma_pool_indices[:batch_size]
                
            else:
                base_pool_size = max(50, len(pbo_space) // 3)
                top_n = min(len(pbo_space), base_pool_size + 50 * consecutive_skips)
                top_k_indices = np.argsort(ucb)[-top_n:].tolist()
                
                candidate_indices = [top_k_indices.pop(-1)]
                
                while len(candidate_indices) < batch_size and top_k_indices:
                    max_min_dist = -1
                    best_idx_in_pool = -1
                    for pool_idx in top_k_indices:
                        dist_to_selected = min([np.linalg.norm(pbo_space[pool_idx] - pbo_space[s]) for s in candidate_indices])
                        if dist_to_selected > max_min_dist:
                            max_min_dist = dist_to_selected
                            best_idx_in_pool = pool_idx
                    
                    candidate_indices.append(best_idx_in_pool)
                    top_k_indices.remove(best_idx_in_pool)
            
        display_images(df, candidate_indices, title=f"Round {current_iter+1} Candidates (Interactive)", filename=f"pbo_rounds.png")
        print(f"Please provide feedback for the following {batch_size} images (Check popup window):", flush=True)
        for idx, c_idx in enumerate(candidate_indices):
            row = df.iloc[c_idx]
            print(f"  [{idx+1}] ID: {row.get('id', 'N/A')} | Prompt: {row['prompt'][:80]}...", flush=True)
            
        while True:
            try:
                line = input(f"Enter the numbers for [Best] and [Worst] (e.g., 1 4), or enter 0 to skip this round: ")
                line = line.strip()
                if line == '0':
                    consecutive_skips += 1
                    for c_idx in candidate_indices:
                        X_train.append(pbo_space[c_idx])
                        y_train.append(0.0)
                    print(f"Round skipped. Total consecutive skips: {consecutive_skips}. Penalty recorded (This round is not counted).", flush=True)
                    gp.fit(np.array(X_train), np.array(y_train))
                    break 
                
                parts = line.split()
                if len(parts) == 2:
                    best_idx = int(parts[0]) - 1
                    worst_idx = int(parts[1]) - 1
                    if 0 <= best_idx < batch_size and 0 <= worst_idx < batch_size:
                        consecutive_skips = 0
                        for idx, c_idx in enumerate(candidate_indices):
                            X_train.append(pbo_space[c_idx])
                            if idx == best_idx: y_train.append(1.0)
                            elif idx == worst_idx: y_train.append(0.0)
                            else: y_train.append(0.5)
                        current_iter += 1
                        break
                print(f"Please enter two numbers between 1 and {batch_size}, or enter 0 to skip.", flush=True)
            except ValueError:
                print("Please enter valid numbers or 0.", flush=True)
        
        print(f"Feedback recorded. Currently collected {len(X_train)} data points.", flush=True)
    print("\nOptimization finished. Calculating final recommendations...", flush=True)
    gp.fit(np.array(X_train), np.array(y_train))
    final_scores = gp.predict(pbo_space)
    best_idx = np.argmax(final_scores)
    print(f"\n==== PBO Predicted [Optimal Generation Parameters] ====", flush=True)
    row = df.iloc[best_idx]
    print(f"ID: {row.get('id', 'N/A')}", flush=True)
    print(f"Model: {row.get('model', 'UNKNOWN')} | LoRAs: {row.get('loras', '')}", flush=True)
    print(f"Prompt: {row['prompt']}", flush=True)
    print(f"CFG: {row['cfgscale']} | Steps: {row['steps']} | Sampler: {row['sampler']}", flush=True)
    print("========================================================\n", flush=True)
    return best_idx

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = '/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery/metadata.json' 
    
    try:
        df = load_and_clean_data(file_path)
        text_features, num_features, sampler_features, model_features, scaler, encoder = build_features(df)
        pbo_space, pca_text, pca_final = perform_two_stage_pca(text_features, num_features, sampler_features, model_features, final_dim=8)
        
        target_index = np.random.randint(0, len(df))
        # print(f"\nRandomly selected image index {target_index} as the reference target for this optimization...", flush=True)
        display_images(df, [target_index], title="Target Image (Goal Reference)", filename="pbo_target.png")
        input("After viewing the target image, press Enter to start the PBO loop...")
        
        best_pbo_index = run_pbo_loop(pbo_space, df, iterations=10, batch_size=4)
        
        print("\n[Final Result] Showing the best image found by PBO...", flush=True)
        display_images(df, [best_pbo_index], title="Final Optimized Result", filename="pbo_final.png")
        
        simulate_pbo_retrieval(pbo_space, df, target_index=best_pbo_index, top_k=5)
        
    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'.", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}", flush=True)