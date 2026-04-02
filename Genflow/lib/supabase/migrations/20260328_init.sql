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

CREATE TABLE IF NOT EXISTS public.embedding_projection_artifacts (
    model_version TEXT PRIMARY KEY,
    artifacts JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION public.set_updated_at_timestamp()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS embedding_projection_artifacts_updated_at
ON public.embedding_projection_artifacts;

CREATE TRIGGER embedding_projection_artifacts_updated_at
BEFORE UPDATE ON public.embedding_projection_artifacts
FOR EACH ROW
EXECUTE FUNCTION public.set_updated_at_timestamp();

CREATE INDEX IF NOT EXISTS image_embeddings_v4_pbo_hnsw_idx
ON public.image_embeddings_v4
USING hnsw (pbo_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS image_embeddings_v4_original_hnsw_idx
ON public.image_embeddings_v4
USING hnsw (original_embedding vector_cosine_ops);

CREATE OR REPLACE FUNCTION public.match_image_embeddings_v4(
    query_embedding vector(8),
    match_count INTEGER DEFAULT 12
)
RETURNS TABLE (
    id BIGINT,
    prompt TEXT,
    style TEXT,
    model TEXT,
    sampler TEXT,
    cfgscale FLOAT,
    steps INT,
    clipskip FLOAT,
    metadata JSONB,
    similarity DOUBLE PRECISION
)
LANGUAGE sql
AS $$
    SELECT
        ie.id,
        ie.prompt,
        ie.style,
        ie.model,
        ie.sampler,
        ie.cfgscale,
        ie.steps,
        ie.clipskip,
        ie.metadata,
        1 - (ie.pbo_embedding <=> query_embedding) AS similarity
    FROM public.image_embeddings_v4 AS ie
    WHERE ie.pbo_embedding IS NOT NULL
    ORDER BY ie.pbo_embedding <=> query_embedding
    LIMIT GREATEST(match_count, 1);
$$;
