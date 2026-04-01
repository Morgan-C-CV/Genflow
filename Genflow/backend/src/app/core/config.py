from pydantic_settings import BaseSettings
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _compute_repo_relative_defaults():
    here = Path(__file__).resolve()
    candidates_meta = []
    candidates_gallery = []

    for parent in [here] + list(here.parents):
        base = parent
        candidates_meta.extend([
            base.joinpath("spider", "civitai_gallery", "metadata.json"),
            base.joinpath("spider", "civitai_gallery_res", "metadata.json"),
            base.joinpath("Genflow", "lib", "metadata.json"),
        ])
        candidates_gallery.extend([
            base.joinpath("spider", "civitai_gallery"),
            base.joinpath("spider", "civitai_gallery_res"),
        ])

    default_meta = None
    for p in candidates_meta:
        if p.is_file():
            default_meta = str(p)
            break
    default_gallery = None
    for d in candidates_gallery:
        if d.is_dir():
            default_gallery = str(d)
            break

    return default_meta, default_gallery


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    METADATA_PATH: str = ""
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite-preview-02-05")
    GALLERY_DIR: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        default_meta, default_gallery = _compute_repo_relative_defaults()
        env_meta = os.getenv("METADATA_PATH", "").strip()
        env_gallery = os.getenv("GALLERY_DIR", "").strip()

        if env_meta and Path(env_meta).is_file():
            self.METADATA_PATH = env_meta
        elif self.METADATA_PATH and Path(self.METADATA_PATH).is_file():
            pass
        elif default_meta:
            self.METADATA_PATH = default_meta

        if env_gallery and Path(env_gallery).is_dir():
            self.GALLERY_DIR = env_gallery
        elif self.GALLERY_DIR and Path(self.GALLERY_DIR).is_dir():
            pass
        elif default_gallery:
            self.GALLERY_DIR = default_gallery

    class Config:
        env_file = ".env"

settings = Settings()
