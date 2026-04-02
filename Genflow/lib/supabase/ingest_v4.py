import json
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from supabase import Client, create_client


def _load_env() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_paths = [
        os.path.join(base_dir, "../../backend/src/.env"),
        os.path.join(base_dir, ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for dp in dotenv_paths:
        if os.path.exists(dp):
            load_dotenv(dp)
            return


_load_env()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_KEY", "").strip()
EMBED_TABLE = os.getenv("SUPABASE_PBO_TABLE", "image_embeddings_v4").strip()
ARTIFACT_TABLE = os.getenv("SUPABASE_ARTIFACT_TABLE", "embedding_projection_artifacts").strip()
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def _to_pgvector_literal(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ",".join(str(float(x)) for x in arr.tolist()) + "]"


def _extract_model(meta_str: str) -> str:
    if pd.isna(meta_str):
        return "UNKNOWN"
    match = re.search(r"Model:\s*([^,]+)", str(meta_str))
    return match.group(1).strip() if match else "UNKNOWN"


def _extract_loras(meta_str: str) -> str:
    if pd.isna(meta_str):
        return ""
    loras = re.findall(r"<lora:([^:]+):", str(meta_str))
    return " ".join(loras)


def load_and_clean_data_v4(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "full_metadata_string" in df.columns:
        df["model"] = df["full_metadata_string"].apply(_extract_model)
        df["loras"] = df["full_metadata_string"].apply(_extract_loras)
    else:
        df["model"] = "UNKNOWN"
        df["loras"] = ""
    df["prompt"] = df.get("prompt", pd.Series([""] * len(df))).fillna("")
    df["style"] = df.get("style", pd.Series([""] * len(df))).fillna("")
    df["enhanced_prompt"] = (df["prompt"] + " " + df["loras"]).str.strip()
    df["negative_prompt"] = df.get("negative_prompt", pd.Series([""] * len(df))).fillna("")
    df["clipskip"] = df.get("clipskip", pd.Series([2] * len(df))).fillna(2).astype(float)
    df["cfgscale"] = pd.to_numeric(df.get("cfgscale", pd.Series([7.0] * len(df))), errors="coerce").fillna(7.0)
    df["steps"] = pd.to_numeric(df.get("steps", pd.Series([20] * len(df))), errors="coerce").fillna(20)
    df["sampler"] = df.get("sampler", pd.Series(["UNKNOWN"] * len(df))).fillna("UNKNOWN").astype(str).str.upper()
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


def build_projection_bundle_v4(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    text_model_name = "clip-ViT-B-32"
    text_model = SentenceTransformer(text_model_name)
    text_features = text_model.encode(df["enhanced_prompt"].tolist(), show_progress_bar=True)

    scaler_num = StandardScaler()
    num_features = scaler_num.fit_transform(df[["cfgscale", "steps", "clipskip"]])

    enc_sampler = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    sampler_features = enc_sampler.fit_transform(df[["sampler"]])

    enc_model = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    model_features = enc_model.fit_transform(df[["model"]])

    pca_text = PCA(n_components=min(20, text_features.shape[0], text_features.shape[1]))
    text_reduced = pca_text.fit_transform(text_features)
    scaler_text = StandardScaler()
    text_reduced_scaled = scaler_text.fit_transform(text_reduced)

    weights = {"text": 4.0, "model": 3.0, "param": 0.5, "sampler": 0.5}
    combined_features = np.hstack(
        [
            text_reduced_scaled * weights["text"],
            model_features * weights["model"],
            num_features * weights["param"],
            sampler_features * weights["sampler"],
        ]
    )
    pca_final = PCA(n_components=min(8, combined_features.shape[0], combined_features.shape[1]))
    pbo_space = pca_final.fit_transform(combined_features)

    return {
        "text_model_name": text_model_name,
        "text_features": text_features,
        "num_features": num_features,
        "sampler_features": sampler_features,
        "model_features": model_features,
        "pbo_space": pbo_space,
        "scaler_num": scaler_num,
        "enc_sampler": enc_sampler,
        "enc_model": enc_model,
        "pca_text": pca_text,
        "scaler_text": scaler_text,
        "pca_final": pca_final,
        "weights": weights,
    }


def _serialize_projection_artifacts(bundle: Dict[str, np.ndarray]) -> Dict[str, object]:
    enc_sampler = bundle["enc_sampler"]
    enc_model = bundle["enc_model"]
    scaler_num = bundle["scaler_num"]
    pca_text = bundle["pca_text"]
    scaler_text = bundle["scaler_text"]
    pca_final = bundle["pca_final"]
    return {
        "model_version": "v4",
        "text_model_name": bundle["text_model_name"],
        "weights": bundle["weights"],
        "numeric_columns": ["cfgscale", "steps", "clipskip"],
        "sampler_categories": enc_sampler.categories_[0].tolist(),
        "model_categories": enc_model.categories_[0].tolist(),
        "scaler_num_mean": scaler_num.mean_.tolist(),
        "scaler_num_scale": scaler_num.scale_.tolist(),
        "pca_text_components": pca_text.components_.tolist(),
        "pca_text_mean": pca_text.mean_.tolist(),
        "scaler_text_mean": scaler_text.mean_.tolist(),
        "scaler_text_scale": scaler_text.scale_.tolist(),
        "pca_final_components": pca_final.components_.tolist(),
        "pca_final_mean": pca_final.mean_.tolist(),
    }


def ingest_embeddings(df: pd.DataFrame, bundle: Dict[str, np.ndarray], batch_size: int = 50) -> None:
    target_dim = 1200
    rows: List[Dict[str, object]] = []
    text_features = bundle["text_features"]
    model_features = bundle["model_features"]
    sampler_features = bundle["sampler_features"]
    num_features = bundle["num_features"]
    pbo_space = bundle["pbo_space"]

    for i in range(len(df)):
        row = df.iloc[i]
        original_combined = np.concatenate(
            [text_features[i], model_features[i], sampler_features[i], num_features[i]]
        )
        if len(original_combined) < target_dim:
            original_combined = np.pad(original_combined, (0, target_dim - len(original_combined)))
        else:
            original_combined = original_combined[:target_dim]
        rows.append(
            {
                "id": int(row["id"]),
                "prompt": str(row.get("prompt", "")),
                "style": str(row.get("style", "")),
                "model": str(row.get("model", "UNKNOWN")),
                "sampler": str(row.get("sampler", "UNKNOWN")),
                "cfgscale": float(row.get("cfgscale", 7.0)),
                "steps": int(row.get("steps", 20)),
                "clipskip": float(row.get("clipskip", 2)),
                "original_embedding": _to_pgvector_literal(original_combined),
                "pbo_embedding": _to_pgvector_literal(pbo_space[i]),
                "metadata": {
                    "loras": str(row.get("loras", "")),
                    "negative_prompt": str(row.get("negative_prompt", "")),
                    "image_url": str(row.get("image_url", "")),
                    "local_path": str(row.get("local_path", "")),
                },
            }
        )
        if len(rows) >= batch_size:
            supabase.table(EMBED_TABLE).upsert(rows).execute()
            rows = []
    if rows:
        supabase.table(EMBED_TABLE).upsert(rows).execute()


def upsert_projection_artifacts(bundle: Dict[str, np.ndarray]) -> None:
    payload = _serialize_projection_artifacts(bundle)
    supabase.table(ARTIFACT_TABLE).upsert(
        {
            "model_version": payload["model_version"],
            "artifacts": payload,
        }
    ).execute()


def resolve_metadata_path() -> str:
    meta_path = os.getenv("METADATA_PATH", "").strip()
    if meta_path and os.path.isfile(meta_path):
        return meta_path
    candidates = [
        os.path.join(os.getcwd(), "spider", "civitai_gallery", "metadata.json"),
        os.path.join(os.getcwd(), "spider", "civitai_gallery_res", "metadata.json"),
        os.path.join(os.getcwd(), "Genflow", "lib", "metadata.json"),
        os.path.join(os.path.dirname(__file__), "..", "metadata.json"),
    ]
    for path in candidates:
        full = os.path.abspath(path)
        if os.path.isfile(full):
            return full
    raise FileNotFoundError("metadata.json not found; set METADATA_PATH")


def main() -> None:
    meta_path = resolve_metadata_path()
    df = load_and_clean_data_v4(meta_path)
    bundle = build_projection_bundle_v4(df)
    ingest_embeddings(df, bundle)
    upsert_projection_artifacts(bundle)
    print(f"ingested={len(df)} table={EMBED_TABLE} artifacts_table={ARTIFACT_TABLE}")


if __name__ == "__main__":
    main()
