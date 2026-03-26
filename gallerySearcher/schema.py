import json
import re
import hashlib
import os
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict, field

@dataclass
class ParsedFields:
    lora_names: List[str] = field(default_factory=list)
    lora_strengths: List[float] = field(default_factory=list)
    content_keywords: List[str] = field(default_factory=list)
    style_keywords: List[str] = field(default_factory=list)
    shot_keywords: List[str] = field(default_factory=list)
    lighting_keywords: List[str] = field(default_factory=list)
    quality_keywords: List[str] = field(default_factory=list)

@dataclass
class ImageMetadata:
    item_id: str
    image_url: str
    local_path: str
    prompt: str = ""
    negative_prompt: str = ""
    model: Optional[str] = None
    sampler: Optional[str] = None
    steps: Optional[int] = None
    cfgscale: Optional[float] = None
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    workflow: Optional[str] = None
    clipskip: Optional[int] = None
    ecosystem: Optional[str] = None
    created_date: Optional[str] = None
    parsed: ParsedFields = field(default_factory=ParsedFields)
    raw_record: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        return d
