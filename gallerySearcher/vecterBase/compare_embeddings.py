import sys
import json
import os
import requests
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity

# Supabase configuration
SUPABASE_URL = "https://jxuyiqdunphnvevkhpsf.supabase.co"
SUPABASE_KEY = "sb_publishable_jYjPubXkv_TrVgzlQGMljw_cKjn8zx0"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

GALLERY_DIR = '/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery'
OUTPUT_DIR = '/Users/mgccvmacair/Myproject/Academic/Genflow/gallerySearcher/vecterBase/comparison_rounds'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_all_ids(table_name="image_embeddings"):
    print(f"Fetching all IDs from {table_name}...")
    res = supabase.table(table_name).select("id").execute()
    return [int(row['id']) for row in res.data]

def fetch_vectors(ids, table_name):
    print(f"Fetching vectors from {table_name} for IDs: {ids}")
    res = supabase.table(table_name).select("id, pbo_embedding, metadata").in_("id", ids).execute()
    data = res.data
    
    # Sort data by the order of input ids
    id_map = {int(row['id']): (json.loads(row['pbo_embedding']), row['metadata']) for row in data}
    vectors = []
    metadata_list = []
    found_ids = []
    for target_id in ids:
        if target_id in id_map:
            vec, meta = id_map[target_id]
            vectors.append(vec)
            metadata_list.append(meta)
            found_ids.append(target_id)
        else:
            print(f"Warning: ID {target_id} not found in {table_name}")
            
    return np.array(vectors), found_ids, metadata_list

def load_image(label, meta, zoom=0.15):
    img = None
    local_filename_standard = f"image_{label}.jpg"
    full_path_standard = os.path.join(GALLERY_DIR, local_filename_standard)
    local_path_meta = meta.get('local_path')
    
    try:
        if os.path.exists(full_path_standard):
            img = Image.open(full_path_standard)
        elif local_path_meta:
            full_path_meta = os.path.join(GALLERY_DIR, local_path_meta)
            if os.path.exists(full_path_meta):
                img = Image.open(full_path_meta)
        
        if img is None and meta.get('image_url'):
            response = requests.get(meta['image_url'], timeout=3)
            img = Image.open(BytesIO(response.content))
            
        if img:
            img = img.convert('RGB')
            img.thumbnail((150, 150))
            return OffsetImage(np.array(img), zoom=zoom)
    except Exception as e:
        print(f"Error loading image for {label}: {e}")
    
    placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 200 # Light gray
    return OffsetImage(placeholder, zoom=zoom)

def add_images_to_axis(ax, labels, metadata, n, zoom=0.25):
    for i in range(n):
        # Y-axis images (left side)
        oi_y = load_image(labels[i], metadata[i], zoom=zoom)
        ab_y = AnnotationBbox(oi_y, (-0.1, i), xybox=(-45, 0), 
                              frameon=True, xycoords='data', boxcoords="offset points",
                              pad=0.1, arrowprops=None)
        ax.add_artist(ab_y)
        
        # X-axis images (bottom side)
        oi_x = load_image(labels[i], metadata[i], zoom=zoom)
        ab_x = AnnotationBbox(oi_x, (i, n-0.9), xybox=(0, -60), 
                              frameon=True, xycoords='data', boxcoords="offset points",
                              pad=0.1, arrowprops=None)
        ax.add_artist(ab_x)
        
        # Add ID labels
        ax.text(-1.1, i, str(labels[i]), va='center', ha='right', fontsize=8, fontweight='bold')
        ax.text(i, n+0.4, str(labels[i]), va='top', ha='center', fontsize=8, fontweight='bold', rotation=45)

def plot_side_by_side(v4_data, v5_data, round_num, filename):
    v4_vectors, v4_ids, v4_meta = v4_data
    v5_vectors, v5_ids, v5_meta = v5_data
    
    if len(v4_vectors) == 0 or len(v5_vectors) == 0:
        print(f"Skipping round {round_num}: Missing data")
        return

    n = len(v4_ids)
    v4_sim = cosine_similarity(v4_vectors)
    v5_sim = cosine_similarity(v5_vectors)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Plot v4
    im1 = ax1.imshow(v4_sim, cmap='YlGnBu')
    ax1.set_title(f"Round {round_num}: v4 Similarity Matrix", pad=50, fontsize=16, fontweight='bold')
    ax1.set_xticks(np.arange(n))
    ax1.set_yticks(np.arange(n))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    add_images_to_axis(ax1, v4_ids, v4_meta, n)
    
    # Plot v5
    im2 = ax2.imshow(v5_sim, cmap='YlGnBu')
    ax2.set_title(f"Round {round_num}: v5 Similarity Matrix", pad=50, fontsize=16, fontweight='bold')
    ax2.set_xticks(np.arange(n))
    ax2.set_yticks(np.arange(n))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    add_images_to_axis(ax2, v5_ids, v5_meta, n)
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f"{v4_sim[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10, weight='bold')
            ax2.text(j, i, f"{v5_sim[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10, weight='bold')
            
    # Set limits and adjust layout
    for ax in [ax1, ax2]:
        ax.set_xlim(-1.2, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
    
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.85, wspace=0.3)
    
    # Add colorbars
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Round {round_num} comparison saved to {filename}")

def main():
    all_ids = fetch_all_ids()
    if len(all_ids) < 5:
        print("Not enough IDs in database to pick 5 random.")
        return
        
    for r in range(1, 11):
        print(f"\n--- Starting Round {r} ---")
        input_ids = random.sample(all_ids, 5)
        
        v4_data = fetch_vectors(input_ids, "image_embeddings_v4")
        v5_data = fetch_vectors(input_ids, "image_embeddings")
        
        filename = os.path.join(OUTPUT_DIR, f"comparison_round_{r}.png")
        plot_side_by_side(v4_data, v5_data, r, filename)

if __name__ == "__main__":
    main()
