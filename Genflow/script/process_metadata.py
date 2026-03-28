import os
import json
import time
import re
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

# Configuration
SOURCE_PATH = "/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery/metadata.json"
OUTPUT_PATH = "/Users/mgccvmacair/Myproject/Academic/Genflow/Genflow/lib/metadata.json"
ENV_PATH = "/Users/mgccvmacair/Myproject/Academic/Genflow/Genflow/backend/src/.env"

BATCH_SIZE = 40  # Process records in groups to optimize API calls
MODEL_NAME = "gemini-3.1-flash-lite-preview"  # Matches user preferred model

# Load environment variables for GOOGLE_API_KEY
load_dotenv(ENV_PATH)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)

def clean_lora_name(lora_str):
    """
    Manual regex backup for Lora cleaning if LLM fails 
    (though we'll use LLM for the main task).
    """
    # Remove <lora: and :weight>
    name = re.sub(r'<lora:([^:]+):[^>]+>', r'\1', lora_str)
    # Remove version numbers like v1, v2, _v1, .v2
    name = re.sub(r'[._]?[vV]\d+(\.\d+)?', '', name)
    # Replace underscores and parentheses with spaces
    name = re.sub(r'[_\(\)]', ' ', name)
    # Clean multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def process_batch(batch):
    """
    Uses Gemini to extract styles and clean LoRAs for a batch of records.
    """
    prompts_data = []
    for item in batch:
        prompts_data.append({
            "id": item["id"],
            "prompt": item.get("prompt", ""),
            "full_metadata": item.get("full_metadata_string", "")
        })

    # System instruction for the LLM
    system_prompt = """
    You are a data extraction assistant specialized in Stable Diffusion metadata.
    For each record provided, perform two tasks:
    1. 'style': Extract specific keywords or phrases from the 'prompt' that describe the art style (e.g., 'Oil Painting', 'Ghibli style', 'photorealistic'). 
       IMPORTANT: The style words MUST be extracted exactly as they appear in the original prompt.
    2. 'lora': Identify all LoRAs mentioned in 'prompt' or 'full_metadata'. 
       Clean the LoRA names by removing version numbers (e.g., v1, v2.0), underscores, and weights. 
       Return just the clean, readable names (e.g., 'ArDec Buildings' instead of 'ArDec2_(Buildings)_v1:1.25').
       
    Respond EXCLUSIVELY with a JSON list of objects, each containing:
    {
      "id": "original_id",
      "style": "comma-separated styles or none",
      "lora": "comma-separated cleaned loras or none"
    }
    """

    user_data_input = json.dumps(prompts_data, ensure_ascii=False)
    
    try:
        response = model.generate_content(
            f"{system_prompt}\n\nData to process:\n{user_data_input}",
            generation_config={"response_mime_type": "application/json"}
        )
        
        extracted_results = json.loads(response.text)
        return {res["id"]: res for res in extracted_results}
    except Exception as e:
        # Fallback: if the batch fails (e.g., blocked content), try one by one
        print(f"Batch failed ({e}). Retrying individually...")
        individual_results = {}
        for item in prompts_data:
            try:
                single_input = json.dumps([item], ensure_ascii=False)
                resp = model.generate_content(
                    f"{system_prompt}\n\nData to process:\n{single_input}",
                    generation_config={"response_mime_type": "application/json"}
                )
                res = json.loads(resp.text)[0]
                individual_results[item["id"]] = res
            except Exception as ex:
                print(f"  ID {item['id']} blocked or failed: {ex}")
                individual_results[item["id"]] = {"style": "", "lora": ""}
        return individual_results

def main():
    if not os.path.exists(SOURCE_PATH):
        print(f"Source file not found: {SOURCE_PATH}")
        return

    with open(SOURCE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Recovery/Resume logic
    processed_ids = set()
    enriched_data = []
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    if "style" in item and "lora" in item:
                        processed_ids.add(str(item["id"]))
                        enriched_data.append(item)
            print(f"Resuming: found {len(processed_ids)} already enriched records.")
        except:
            print("Output file exists but couldn't be parsed. Starting fresh.")

    # Filter out already processed items
    remaining_data = [item for item in data if str(item["id"]) not in processed_ids]
    
    if not remaining_data:
        print("All records are already processed.")
        return

    print(f"Processing {len(remaining_data)} remaining records (Total: {len(data)})...")
    
    # Process in batches
    for i in tqdm(range(0, len(remaining_data), BATCH_SIZE)):
        batch = remaining_data[i:i + BATCH_SIZE]
        results_map = process_batch(batch)
        
        for item in batch:
            res = results_map.get(str(item["id"]), {"style": "", "lora": ""})
            item["style"] = res.get("style", "")
            item["lora"] = res.get("lora", "")
            enriched_data.append(item)
        
        # Incremental Save
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)
        
        time.sleep(1.0)

    print(f"\nEnrichment complete! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
