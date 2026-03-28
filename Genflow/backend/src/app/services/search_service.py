from app.repositories.search_repository import SearchRepository
from app.repositories.llm_repository import LLMRepository

class SearchService:
    def __init__(self, search_repo: SearchRepository, llm_repo: LLMRepository):
        self.search_repo = search_repo
        self.llm_repo = llm_repo

    def summarize_search_results(self, index: int, top_k: int = 5):
        # 1. Get search results from repository
        results = self.search_repo.search_by_index(index, top_k)
        
        # 2. Extract metadata for LLM summary
        summary = self.llm_repo.generate_summary(results)
        
        return {
            "query_index": index,
            "top_results": results,
            "llm_summary": summary
        }
