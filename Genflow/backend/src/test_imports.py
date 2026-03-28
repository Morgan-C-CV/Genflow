import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.main import app
    print("FastAPI app imported successfully.")
    
    from app.services.search_service import SearchService
    print("SearchService imported successfully.")
    
    from app.repositories.search_repository import SearchRepository
    from app.repositories.llm_repository import LLMRepository
    print("Repositories imported successfully.")
    
    from app.modules.embedding_v4 import ImageEmbeddingSearch
    print("Embedding module imported successfully.")
    
    print("\n--- Project structure check PASSED ---")
except Exception as e:
    print(f"\n--- Project structure check FAILED ---")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
