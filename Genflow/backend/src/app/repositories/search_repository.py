from app.modules.embedding_v4 import ImageEmbeddingSearch
from app.core.config import settings

class SearchRepository:
    def __init__(self):
        self.search_engine = ImageEmbeddingSearch(
            metadata_path=settings.METADATA_PATH,
            gallery_dir=settings.GALLERY_DIR
        )

    def search_by_index(self, index: int, top_k: int = 5):
        return self.search_engine.search_top_k(query_index=index, top_k=top_k)

    def get_all_data(self):
        return self.search_engine.df
