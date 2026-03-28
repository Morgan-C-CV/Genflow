from pydantic import BaseModel
from typing import List, Optional

class SearchResult(BaseModel):
    id: str
    prompt: str
    model: str
    loras: str
    cfgscale: float
    steps: int
    sampler: str
    distance: float
    image_url: Optional[str] = None

class SearchResponse(BaseModel):
    query_index: int
    top_results: List[SearchResult]
    llm_summary: str

class SummarizeRequest(BaseModel):
    index: int
    top_k: int = 5
