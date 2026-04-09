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

    def build_diverse_reference_bundle(self, index: int):
        return self.search_repo.search_diverse_references(query_index=index)

    def generate_image_metadata(
        self,
        reference_bundle: dict,
        user_intent: str,
        previous_output: str = "",
        validation_error: str = "",
    ):
        metadata_json = self.llm_repo.generate_metadata_from_intent(
            reference_bundle,
            user_intent,
            previous_output=previous_output,
            validation_error=validation_error,
        )
        return metadata_json
