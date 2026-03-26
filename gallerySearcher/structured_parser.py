import json
import re
import os
from typing import List, Dict, Any, Set
from collections import Counter
from schema import ImageMetadata, ParsedFields

class StructuredParser:
    def __init__(self):
        # Keyword Lists based on Phase 2 requirements
        self.keywords = {
            "content": [
                "portrait", "people", "person", "woman", "man", "girl", "boy", "character", "1girl", "1boy",
                "landscape", "mountain", "forest", "lake", "ocean", "beach", "sea", "sky", "cloud",
                "architecture", "building", "interior", "room", "house", "city", "street", "castle", "temple",
                "object", "product", "item", "tool", "machine", "vehicle", "car", "plane", "mercedes",
                "animal", "creature", "dragon", "cat", "dog", "wolf", "bird", "horse", "snake",
                "fantasy", "sci-fi", "anime", "illustration", "mythical", "wizard", "sorceress"
            ],
            "style": [
                "cinematic", "photorealistic", "realistic", "photo", "film", "analog", "vintage", "retro",
                "anime", "illustration", "drawing", "painting", "watercolor", "oil painting", "digital painting",
                "pixel art", "fantasy", "dark fantasy", "cyberpunk", "retro futurism", "synthwave",
                "classic", "modern", "origami", "art deco", "sculpture", "flat", "minimalist", "gothic",
                "1930s", "1950s", "1970s", "technicolor", "poster", "art", "sketch"
            ],
            "shot": [
                "close-up", "close up", "extreme close-up", "portrait", "full body", "medium shot", "wide shot",
                "top-down", "side view", "front view", "back view", "macro", "from below", "from above",
                "cowboy shot", "ranch shot", "low angle", "high angle", "bokeh", "depth of field", "85mm", "35mm"
            ],
            "lighting": [
                "rim light", "backlighting", "dramatic lighting", "soft light", "hard light",
                "golden hour", "sunset", "moonlight", "moody light", "low key", "high key", 
                "volumetric lighting", "god rays", "neon", "bioluminescent", "glowing",
                "side lighting", "studio light", "natural light", "ambient light", "sunlight"
            ],
            "quality": [
                "masterpiece", "best quality", "highres", "highly detailed", "ultra detailed",
                "award-winning", "8k", "4k", "uhd", "professional", "amazing quality",
                "meticulous", "intricate", "detailed skin", "photorealistic", "hyperrealistic", "absurdres"
            ]
        }
        self.lora_pattern = re.compile(r"<lora:([^:]+):?([\d\.]+)?", re.IGNORECASE)

    def parse_loras(self, prompt: str) -> (List[str], List[float]):
        names = []
        strengths = []
        matches = self.lora_pattern.findall(prompt)
        for name, strength in matches:
            names.append(name.strip())
            try:
                strengths.append(float(strength) if strength else 1.0)
            except:
                strengths.append(1.0)
        return names, strengths

    def extract_keywords(self, prompt: str, category_list: List[str]) -> List[str]:
        found = []
        # Normalizing prompt: lowercase and remove punctuation for basic matching
        clean_prompt = prompt.lower()
        for kw in category_list:
            # Simple word boundary matching
            if re.search(rf"\b{re.escape(kw)}\b", clean_prompt):
                found.append(kw)
        return found

    def process_item(self, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        prompt = item_dict.get("prompt", "")
        
        # Parse LoRAs
        lora_names, lora_strengths = self.parse_loras(prompt)
        
        # Parse Keywords
        content_kws = self.extract_keywords(prompt, self.keywords["content"])
        style_kws = self.extract_keywords(prompt, self.keywords["style"])
        shot_kws = self.extract_keywords(prompt, self.keywords["shot"])
        lighting_kws = self.extract_keywords(prompt, self.keywords["lighting"])
        quality_kws = self.extract_keywords(prompt, self.keywords["quality"])

        # Update ParsedFields
        parsed = ParsedFields(
            lora_names=lora_names,
            lora_strengths=lora_strengths,
            content_keywords=content_kws,
            style_keywords=style_kws,
            shot_keywords=shot_kws,
            lighting_keywords=lighting_kws,
            quality_keywords=quality_kws
        )
        
        # Inplace update of the dict (or return new)
        item_dict["parsed"] = {
            "lora_names": lora_names,
            "lora_strengths": lora_strengths,
            "content_keywords": content_kws,
            "style_keywords": style_kws,
            "shot_keywords": shot_kws,
            "lighting_keywords": lighting_kws,
            "quality_keywords": quality_kws
        }
        return item_dict

    def run(self, input_jsonl: str, output_jsonl: str):
        processed_items = []
        all_stats = {
            "lora_counts": Counter(),
            "keyword_counts": {cat: Counter() for cat in self.keywords}
        }

        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                processed = self.process_item(item)
                processed_items.append(processed)
                
                # Stats
                for name in processed["parsed"]["lora_names"]:
                    all_stats["lora_counts"][name] += 1
                for cat in self.keywords:
                    for kw in processed["parsed"][f"{cat}_keywords"]:
                        all_stats["keyword_counts"][cat][kw] += 1

        with open(output_jsonl, "w", encoding="utf-8") as f:
            for item in processed_items:
                f.write(json.dumps(item) + "\n")

        # Save stats
        stats_serializable = {
            "lora_counts": dict(all_stats["lora_counts"].most_common(50)),
            "keyword_counts": {cat: dict(all_stats["keyword_counts"][cat].most_common(50)) for cat in self.keywords}
        }
        with open("gallerySearcher/keyword_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_serializable, f, indent=2)

        print(f"\nPhase 2 Complete. Parsed {len(processed_items)} items.")
        print(f"Stats saved to gallerySearcher/keyword_stats.json")

if __name__ == "__main__":
    IN_JSONL = "gallerySearcher/normalized_items.jsonl"
    OUT_JSONL = "gallerySearcher/parsed_items.jsonl"
    parser = StructuredParser()
    parser.run(IN_JSONL, OUT_JSONL)
