from fastapi import APIRouter, HTTPException, Depends
from app.models.search import SummarizeRequest, SearchResponse
from app.services.search_service import SearchService
from app.repositories.search_repository import SearchRepository
from app.repositories.llm_repository import LLMRepository

router = APIRouter()

# Global instances for repository and service
# In a larger app, these would be managed with dependency injection or app state
_search_repo = None
_llm_repo = None
_search_service = None

def get_search_service():
    global _search_repo, _llm_repo, _search_service
    if _search_repo is None:
        _search_repo = SearchRepository()
    if _llm_repo is None:
        _llm_repo = LLMRepository()
    if _search_service is None:
        _search_service = SearchService(_search_repo, _llm_repo)
    return _search_service

@router.post("/summarize", response_model=SearchResponse)
def summarize(request: SummarizeRequest, service: SearchService = Depends(get_search_service)):
    try:
        result = service.summarize_search_results(request.index, request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
