import sys
import json
import os
import requests
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
    # Priority 1: Check standard local naming image_{id}.jpg
    local_filename_standard = f"image_{label}.jpg"
    full_path_standard = os.path.join(GALLERY_DIR, local_filename_standard)
    
    # Priority 2: Check metadata local_path
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
    
    # Return a placeholder if image fails to load
    placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 200 # Light gray
    return OffsetImage(placeholder, zoom=zoom)

def plot_similarity_matrix(vectors, labels, metadata, title, filename):
    if len(vectors) == 0:
        print(f"No vectors to plot for {title}")
        return
        
    similarity = cosine_similarity(vectors)
    n = len(labels)
    
    # Create figure with more space for images
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(similarity, cmap='YlGnBu')
    
    # Set tick positions
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    
    # Hide default tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add images and text to axes
    for i in range(n):
        # Y-axis images (left side)
        oi_y = load_image(labels[i], metadata[i], zoom=0.35)
        # Position image to the left of the matrix
        ab_y = AnnotationBbox(oi_y, (-0.1, i), xybox=(-60, 0), 
                              frameon=True, xycoords='data', boxcoords="offset points",
                              pad=0.1, arrowprops=None)
        ax.add_artist(ab_y)
        
        # X-axis images (bottom side)
        oi_x = load_image(labels[i], metadata[i], zoom=0.35)
        # Position image below the matrix
        ab_x = AnnotationBbox(oi_x, (i, n-0.9), xybox=(0, -80), 
                              frameon=True, xycoords='data', boxcoords="offset points",
                              pad=0.1, arrowprops=None)
        ax.add_artist(ab_x)
        
        # Add ID labels
        ax.text(-1.4, i, str(labels[i]), va='center', ha='right', fontsize=9, fontweight='bold')
        ax.text(i, n+0.5, str(labels[i]), va='top', ha='center', fontsize=9, fontweight='bold', rotation=45)

    # Loop over data dimensions and create text annotations inside the matrix
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{similarity[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=11, weight='bold')
    
    ax.set_title(title, pad=50, fontsize=16, fontweight='bold')
    
    # Explicitly set axis limits to ensure images aren't cut off
    ax.set_xlim(-1.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    
    # Adjust layout to accommodate the outer images
    plt.subplots_adjust(left=0.25, bottom=0.25, top=0.9)
    
    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Cosine Similarity', rotation=-90, va="bottom")
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Matrix saved to {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_embeddings.py <id1> <id2> <id3> ...")
        # Default IDs for demo
        input_ids = [125428448, 6399138, 124624786, 121820705, 125433047]
    else:
        input_ids = [int(x) for x in sys.argv[1:]]
        
    print(f"Comparing embeddings for IDs: {input_ids}")
    
    # Fetch v4 vectors
    v4_vectors, v4_ids, v4_meta = fetch_vectors(input_ids, "image_embeddings_v4")
    # Fetch v5 vectors
    v5_vectors, v5_ids, v5_meta = fetch_vectors(input_ids, "image_embeddings")
    
    # Plot v4
    plot_similarity_matrix(v4_vectors, v4_ids, v4_meta, "v4 Embedding Similarity Matrix", "v4_similarity.png")
    # Plot v5
    plot_similarity_matrix(v5_vectors, v5_ids, v5_meta, "v5 Embedding Similarity Matrix", "v5_similarity.png")
    
    plt.show()

if __name__ == "__main__":
    main()
