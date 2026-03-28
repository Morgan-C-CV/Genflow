from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery/metadata.json")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite-preview-02-05")
    GALLERY_DIR: str = os.getenv("GALLERY_DIR", "/Users/mgccvmacair/Myproject/Academic/Genflow/spider/civitai_gallery")

    class Config:
        env_file = ".env"

settings = Settings()
