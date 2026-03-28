from supabase import create_client, Client
import json

# Supabase configuration
SUPABASE_URL = "https://jxuyiqdunphnvevkhpsf.supabase.co"
SUPABASE_KEY = "sb_publishable_jYjPubXkv_TrVgzlQGMljw_cKjn8zx0"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def verify_data():
    print("--- Database Verification ---")
    
    # Check total count
    res = supabase.table("image_embeddings").select("id", count="exact").limit(1).execute()
    count = res.count
    print(f"Total records in database: {count}")
    
    if count > 0:
        # Fetch one record to see dimensions
        sample = supabase.table("image_embeddings").select("*").limit(1).execute().data[0]
        print(f"\nSample Record (ID: {sample['id']}):")
        print(f"- Prompt: {sample['prompt'][:50]}...")
        print(f"- Original Embedding Dim: {len(json.loads(sample['original_embedding']))}")
        print(f"- PBO Embedding Dim: {len(json.loads(sample['pbo_embedding']))}")
        print(f"- Metadata keys: {list(sample['metadata'].keys())}")

def demo_search():
    print("\n--- Vector Search Demo (Cosine Similarity) ---")
    
    # 1. Get a random vector as query
    sample = supabase.table("image_embeddings").select("pbo_embedding").limit(1).execute().data[0]
    query_vector = sample['pbo_embedding']
    
    # 2. Perform search using RPC
    res = supabase.rpc("match_images", {
        "query_embedding": query_vector,
        "match_threshold": 0.5,
        "match_count": 5
    }).execute()
    
    print(f"Top 5 similar images (PBO Space):")
    for row in res.data:
        print(f"- ID: {row['id']} | Similarity: {row['similarity']:.4f} | Prompt: {row['prompt'][:60]}...")

if __name__ == "__main__":
    verify_data()
    demo_search()
