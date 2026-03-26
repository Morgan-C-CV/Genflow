import json
import re
import hashlib
import os
import csv
from typing import List, Dict, Any, Optional
from schema import ImageMetadata

class MetadataParser:
    def __init__(self, raw_file_path: str):
        self.raw_file_path = raw_file_path
        self.results: List[ImageMetadata] = []
        self.stats = {
            "total_raw_records": 0,
            "parsed_records": 0,
            "skipped_records": 0,
            "duplicate_ids": 0,
            "missing_prompt": 0,
            "missing_image": 0,
        }
        self.processed_ids = set()

    def generate_id(self, item: Dict[str, Any]) -> str:
        unique_str = (
            str(item.get("local_path", "")) +
            str(item.get("image_url", "")) +
            str(item.get("prompt", "")) +
            str(item.get("seed", ""))
        )
        return hashlib.sha256(unique_str.encode("utf-8")).hexdigest()

    def extract_from_string(self, metadata_str: str, key: str) -> Optional[str]:
        if not metadata_str:
            return None
        # Common patterns: "Key: Value", "Key: Value,"
        patterns = [
            re.compile(rf"{key}:\s*([^,]+)", re.IGNORECASE),
            re.compile(rf"{key}\s*=\s*([^,]+)", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(metadata_str)
            if match:
                return match.group(1).strip()
        return None

    def parse_item(self, item: Dict[str, Any]) -> Optional[ImageMetadata]:
        self.stats["total_raw_records"] += 1
        
        # 1. Identify raw fields
        raw_id = item.get("id")
        prompt = item.get("prompt", "")
        negative_prompt = item.get("negative_prompt", "")
        image_url = item.get("image_url", "")
        local_path = item.get("local_path", "")
        full_meta = item.get("full_metadata_string", "")

        # 2. Handle missing prompt/image
        if not prompt and not full_meta:
            self.stats["missing_prompt"] += 1
            # We might still want to parse it if we can extract prompt from full_meta
        
        if not image_url and not local_path:
            self.stats["missing_image"] += 1
            # Critical according to Phase 1: skip or log? 
            # Phase 1 says "records missing image path/url" should be counted.
            # I'll keep them for now but log.

        # 3. Create Stable ID
        item_id = raw_id if raw_id else self.generate_id(item)
        if item_id in self.processed_ids:
            self.stats["duplicate_ids"] += 1
            return None
        self.processed_ids.add(item_id)

        # 4. Resolve Local Path and Image URL
        # The user wants image_url to be the local path.
        source_dir = os.path.dirname(os.path.abspath(self.raw_file_path))
        abs_local_path = ""
        if local_path:
            abs_local_path = os.path.join(source_dir, local_path)
        
        # Override image_url with local path if requested
        final_image_url = abs_local_path if abs_local_path else image_url

        # 5. Extract fields with fallbacks
        def get_field(key: str, fallback_key: str = None):
            val = item.get(key)
            if val is None or val == "":
                val = self.extract_from_string(full_meta, fallback_key or key)
            return val

        steps = get_field("steps", "Steps")
        cfgscale = get_field("cfgscale", "CFG scale")
        sampler = get_field("sampler", "Sampler")
        seed = get_field("seed", "Seed")
        model = get_field("model", "Model")
        clipskip = get_field("clipskip", "Clip skip")
        ecosystem = get_field("ecosystem", "ecosystem")
        created_date = get_field("created_date", "Created Date")
        workflow = get_field("workflow", "workflow")

        # Handle width/height specially (e.g., "Size: 1024x1024")
        width = item.get("width")
        height = item.get("height")
        if width is None or height is None:
            size_match = re.search(r"Size:\s*(\d+)x(\d+)", full_meta, re.IGNORECASE)
            if size_match:
                width = size_match.group(1)
                height = size_match.group(2)
            else:
                # Try individual W/H regex
                width = width or self.extract_from_string(full_meta, "width")
                height = height or self.extract_from_string(full_meta, "height")

        # 5. Type Conversions
        try:
            steps = int(steps) if steps is not None else None
        except: steps = None
        
        try:
            cfgscale = float(cfgscale) if cfgscale is not None else None
        except: cfgscale = None
        
        try:
            seed = int(seed) if seed is not None else None
        except: seed = None
        
        try:
            width = int(width) if width is not None else None
            height = int(height) if height is not None else None
        except: width, height = None, None
        
        try:
            clipskip = int(clipskip) if clipskip is not None else None
        except: clipskip = None

        # 6. Normalize Prompts
        if not prompt and full_meta:
            # Basic prompt extraction from start of full_meta until "Negative prompt:" or "Steps:"
            prompt_match = re.split(r"Negative prompt:|Steps:", full_meta, 1, re.IGNORECASE)
            prompt = prompt_match[0].strip() if prompt_match else ""

        if not negative_prompt and full_meta:
            neg_match = re.search(r"Negative prompt:\s*(.*?)(?=Steps:|$)", full_meta, re.DOTALL | re.IGNORECASE)
            if neg_match:
                negative_prompt = neg_match.group(1).strip()

        # Final object
        meta = ImageMetadata(
            item_id=str(item_id),
            image_url=str(final_image_url or ""),
            local_path=str(abs_local_path or ""),
            prompt=str(prompt or ""),
            negative_prompt=str(negative_prompt or ""),
            model=str(model) if model else None,
            sampler=str(sampler) if sampler else None,
            steps=steps,
            cfgscale=cfgscale,
            seed=seed,
            width=width,
            height=height,
            workflow=str(workflow) if workflow else None,
            clipskip=clipskip,
            ecosystem=str(ecosystem) if ecosystem else None,
            created_date=str(created_date) if created_date else None,
            raw_record=item
        )
        self.stats["parsed_records"] += 1
        return meta

    def run(self, output_jsonl: str, output_csv: str):
        if not os.path.exists(self.raw_file_path):
            print(f"Error: Raw file not found at {self.raw_file_path}")
            return

        with open(self.raw_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            parsed = self.parse_item(item)
            if parsed:
                self.results.append(parsed)
            else:
                self.stats["skipped_records"] += 1

        # Save to JSONL
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for item in self.results:
                f.write(json.dumps(item.to_dict()) + "\n")

        # Save to CSV
        if self.results:
            keys = self.results[0].to_dict().keys()
            with open(output_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for item in self.results:
                    # Filter out raw_record for CSV to keep it clean
                    row = item.to_dict()
                    row["raw_record"] = json.dumps(row["raw_record"]) # compact raw record in csv
                    writer.writerow(row)

        print("\n=== Parser Results Summary ===")
        print(json.dumps(self.stats, indent=2))
        print(f"Saved {len(self.results)} normalized items to {output_jsonl} and {output_csv}")

if __name__ == "__main__":
    RAW_PATH = "/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery/metadata.json"
    OUT_JSONL = "/Users/mgccvmacair/Myproject/Academic/Genflow/gallerySearcher/normalized_items.jsonl"
    OUT_CSV = "/Users/mgccvmacair/Myproject/Academic/Genflow/gallerySearcher/normalized_items.csv"
    
    parser = MetadataParser(RAW_PATH)
    parser.run(OUT_JSONL, OUT_CSV)
