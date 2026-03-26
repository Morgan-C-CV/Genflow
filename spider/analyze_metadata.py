import json
import re
import os
from collections import Counter

def analyze_metadata(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    loras = Counter()
    checkpoints = Counter()
    
    # Regex patterns for parsing from full_metadata_string
    lora_pattern = re.compile(r"<lora:([^:]+):[^>]+>")
    model_pattern = re.compile(r"Model: ([^,]+)")
    base_model_attr_pattern = re.compile(r"baseModel: ([^,]+)")

    for item in data:
        # 1. Extract Checkpoint (Base Model)
        model_found = None
        
        # Check dedicated fields first
        if "resources" in item and item["resources"].get("base_model"):
            model_found = item["resources"]["base_model"].get("name")
        elif "baseModel" in item:
            model_found = item["baseModel"]
        
        # Fallback to parsing full_metadata_string
        if not model_found and "full_metadata_string" in item:
            ms = item["full_metadata_string"]
            # Look for "Model: Name"
            m = model_pattern.search(ms)
            if m:
                model_found = m.group(1).strip()
            else:
                # Look for "baseModel: Name"
                m = base_model_attr_pattern.search(ms)
                if m:
                    model_found = m.group(1).strip()
        
        if model_found:
            checkpoints[model_found] += 1
        else:
            checkpoints["Unknown"] += 1

        # 2. Extract LoRAs
        lora_list = []
        
        # Check dedicated fields
        if "resources" in item and item["resources"].get("loras"):
            for l in item["resources"]["loras"]:
                lora_list.append(l.get("name"))
        
        # Fallback/Supplemental parsing from full_metadata_string
        if "full_metadata_string" in item:
            ms = item["full_metadata_string"]
            matches = lora_pattern.findall(ms)
            for m in matches:
                name = m.strip()
                if name not in lora_list: # Avoid duplicates if already found in resources
                    lora_list.append(name)
        
        for lora in lora_list:
            loras[lora] += 1

    # Printing Results
    print("\n" + "="*50)
    print(f"Analyzed {len(data)} images in {file_path}")
    print("="*50)

    print("\n--- Checkpoints (Base Models) Rank ---")
    for name, count in checkpoints.most_common():
        print(f"{count:3d} | {name}")

    print("\n--- LoRAs Rank ---")
    if not loras:
        print("No LoRAs found.")
    for name, count in loras.most_common():
        print(f"{count:3d} | {name}")
    print("="*50 + "\n")

if __name__ == "__main__":
    METADATA_PATH = "/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery/metadata.json"
    analyze_metadata(METADATA_PATH)
