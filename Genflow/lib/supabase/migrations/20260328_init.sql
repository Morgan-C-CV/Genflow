-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to store image metadata and embeddings
CREATE TABLE IF NOT EXISTS public.image_embeddings_v4 (
    id BIGINT PRIMARY KEY, -- Civitai ID
    prompt TEXT,
    style TEXT,
    model TEXT,
    sampler TEXT,
    cfgscale FLOAT,
    steps INT,
    clipskip FLOAT,
    
    -- Version 1: Original High-Dim Embedding (Concatenated)
    -- Total dimension: prompt(512) + style(512) + model_oh(?) + sampler_oh(?) + parameters(3)
    -- We'll use a larger fixed dimension for original_embedding to allow for growth
    original_embedding vector(1200),
    
    -- Version 2: PCA-reduced result (Target dimensions from v5.py)
    pbo_embedding vector(8),
    
    -- Metadata blob for any additional info
    metadata JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for vector search (HNSW is recommended for large datasets)
-- We'll add indexes after data ingestion for better performance
