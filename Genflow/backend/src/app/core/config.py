from pydantic_settings import BaseSettings, SettingsConfigDict
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


def _load_deepseek_credentials() -> str:
    """Read the DeepSeek API key from the ``.env_ds`` file, if present.

    Genflow keeps the DeepSeek key in a separate file so the Gemini ``.env``
    stays untouched. Both ``api_key=`` and ``DEEPSEEK_API_KEY=`` are accepted.
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / ".env_ds"
        if not candidate.is_file():
            continue
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ""
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if key.strip().lower() in {"api_key", "deepseek_api_key"}:
                return value.strip().strip('"').strip("'")
        return ""
    return ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    METADATA_PATH: str = ""
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite-preview-02-05")
    GALLERY_DIR: str = ""

    # --- LLM provider selection -----------------------------------------
    # "gemini" | "deepseek" | "" (auto: DeepSeek when a key is available).
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "") or _load_deepseek_credentials()
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    LLM_REQUEST_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))

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

settings = Settings()
