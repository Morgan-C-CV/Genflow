import json
import random
import pandas as pd
from typing import List, Dict, Any

def run_validation(jsonl_path: str):
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    df = pd.DataFrame(items)
    total = len(df)

    print(f"\n--- Validation 1.1: Row Count ---")
    print(f"Total parsed items: {total}")

    print(f"\n--- Validation 1.2: Uniqueness ---")
    dup_count = df["item_id"].duplicated().sum()
    print(f"Duplicate item_ids: {dup_count}")
    if dup_count > 0:
        print("WARNING: Non-unique IDs found!")

    print(f"\n--- Validation 1.3: Critical-field Coverage ---")
    critical_fields = [
        "prompt", "negative_prompt", "sampler", "steps", 
        "cfgscale", "seed", "width", "height", "model"
    ]
    for field in critical_fields:
        count = df[field].notnull().sum()
        # For strings, also check if not empty
        if df[field].dtype == object:
             count = (df[field].notnull() & (df[field] != "")).sum()
        percentage = (count / total) * 100
        print(f"{field:16}: {count:4} / {total} ({percentage:6.2f}%)")

    print(f"\n--- Validation 1.4: Random Samples (3) ---")
    samples = df.sample(min(3, total)).to_dict(orient="records")
    for i, s in enumerate(samples):
        print(f"\nSample {i+1}:")
        # Print a subset of fields for brevity
        subset = {k: v for k, v in s.items() if k != "raw_record"}
        print(json.dumps(subset, indent=2))

if __name__ == "__main__":
    OUT_JSONL = "/Users/mgccvmacair/Myproject/Academic/Genflow/gallerySearcher/normalized_items.jsonl"
    run_validation(OUT_JSONL)
