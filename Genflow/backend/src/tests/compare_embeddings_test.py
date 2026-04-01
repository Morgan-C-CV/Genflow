import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict

# Add src to sys.path so we can import app
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from app.modules.embedding_v4 import ImageEmbeddingSearch
from app.core.config import settings

class ForcedLocalSearch(ImageEmbeddingSearch):
    """
    Subclass that forces local calculation by overriding the Supabase loader.
    """
    def _try_load_precomputed_pbo_space(self) -> bool:
        print("Forcing local embedding calculation (skipping Supabase)...", flush=True)
        return False

def compare_vectors(v1: np.ndarray, v2: np.ndarray, tolerance: float = 1e-5):
    """
    Compare two vectors for similarity.
    """
    if v1.shape != v2.shape:
        return False, f"Shape mismatch: {v1.shape} vs {v2.shape}"
    
    mse = np.mean((v1 - v2)**2)
    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    is_close = np.allclose(v1, v2, atol=tolerance)
    return is_close, f"MSE: {mse:.8f}, Cosine Sim: {cosine_sim:.8f}"

def main():
    print("="*60)
    print("EMBEDDING VERSION COMPARISON TOOL (Local vs Supabase)")
    print("="*60)
    
    meta_path = settings.METADATA_PATH
    gallery_dir = settings.GALLERY_DIR
    
    if not os.path.exists(meta_path):
        print(f"Error: Metadata path not found at {meta_path}")
        return

    # 1. Initialize Supabase Version (Default)
    print("\n[Step 1] Initializing Supabase Version...")
    # Temporarily ensure SUPABASE_STRICT is True so we know if it fails
    os.environ["SUPABASE_STRICT"] = "true"
    try:
        supabase_search = ImageEmbeddingSearch(meta_path, gallery_dir)
        # Check if it actually loaded from Supabase
        is_precomputed = (supabase_search.text_features is None)
    except Exception as e:
        print(f"Failed to initialize Supabase version: {e}")
        print("Make sure you have populated the database and fixed RLS permissions.")
        return

    # 2. Initialize Forced Local Version
    print("\n[Step 2] Initializing Forced Local Version (Generating embeddings)...")
    local_search = ForcedLocalSearch(meta_path, gallery_dir)

    # 3. Structural Comparison
    print("\n" + "="*40)
    print("STRUCTURAL COMPARISON")
    print("="*40)
    
    print(f"Supabase DF Size: {len(supabase_search.df)}")
    print(f"Local DF Size:    {len(local_search.df)}")
    
    if len(supabase_search.df) != len(local_search.df):
        print("WARNING: Dataframe row counts do not match!")
    
    print(f"Supabase PBO Space Shape: {supabase_search.pbo_space.shape}")
    print(f"Local PBO Space Shape:    {local_search.pbo_space.shape}")

    # 4. Numerical Comparison (First few rows)
    print("\n" + "="*40)
    print("NUMERICAL PRECISION CHECK (PBO Space)")
    print("="*40)
    
    num_to_compare = min(5, len(supabase_search.pbo_space))
    for i in range(num_to_compare):
        v_sup = supabase_search.pbo_space[i]
        v_loc = local_search.pbo_space[i]
        match, info = compare_vectors(v_sup, v_loc)
        status = "✅ MATCH" if match else "❌ DIFF"
        print(f"Row {i} (ID {supabase_search.df.iloc[i].get('id')}): {status} | {info}")

    # 5. Search Result Comparison
    print("\n" + "="*40)
    print("SEARCH RESULT CONSISTENCY CHECK")
    print("="*40)
    
    # Pick a random query index
    query_idx = np.random.randint(0, len(local_search.pbo_space))
    query_vec = local_search.pbo_space[query_idx].reshape(1, -1)
    
    print(f"Querying for Index {query_idx} (ID: {local_search.df.iloc[query_idx].get('id')})...")
    
    res_sup = supabase_search.search_top_k(query_vector=query_vec, top_k=5)
    res_loc = local_search.search_top_k(query_vector=query_vec, top_k=5)
    
    print("\nTop 5 Results (Supabase):")
    for r in res_sup:
        print(f"  - ID: {r['id']}, Distance: {r['distance']}")
        
    print("\nTop 5 Results (Local):")
    for r in res_loc:
        print(f"  - ID: {r['id']}, Distance: {r['distance']}")

    # Compare the top-1 ID
    if res_sup[0]['id'] == res_loc[0]['id']:
        print("\n✅ Top-1 Result matches exactly!")
    else:
        print("\n❌ Top-1 Result mismatch!")

    # Compare ID sets
    set_sup = set(r['id'] for r in res_sup)
    set_loc = set(r['id'] for r in res_loc)
    overlap = set_sup.intersection(set_loc)
    print(f"Result Overlap (IDs): {len(overlap)}/5 ({len(overlap)/5*100:.0f}%)")

    print("\n" + "="*60)
    print("COMPARISON FINISHED")
    print("="*60)

if __name__ == "__main__":
    main()
