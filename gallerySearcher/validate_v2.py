import json
import random

def verify_traceability(jsonl_path: str, n=5):
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    
    samples = random.sample(items, min(n, len(items)))
    
    print("\n--- Phase 2 Verification: Traceability (Prompt -> Parsed) ---")
    for i, item in enumerate(samples):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {item['prompt'][:200]}...")
        print(f"LoRAs : {item['parsed']['lora_names']}")
        print(f"Content: {item['parsed']['content_keywords']}")
        print(f"Style  : {item['parsed']['style_keywords']}")
        print(f"Shot   : {item['parsed']['shot_keywords']}")
        print(f"Light  : {item['parsed']['lighting_keywords']}")

def report_empty_rates(jsonl_path: str):
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    
    total = len(items)
    categories = ["lora_names", "content_keywords", "style_keywords", "shot_keywords", "lighting_keywords", "quality_keywords"]
    
    print("\n--- Phase 2 Verification: Empty Parse Rates ---")
    for cat in categories:
        empty_count = sum(1 for item in items if not item["parsed"][cat])
        rate = (empty_count / total) * 100
        print(f"{cat:18}: {empty_count:4} / {total} ({rate:6.2f}% empty)")

if __name__ == "__main__":
    OUT_JSONL = "/Users/mgccvmacair/Myproject/Academic/Genflow/gallerySearcher/parsed_items.jsonl"
    verify_traceability(OUT_JSONL)
    report_empty_rates(OUT_JSONL)
