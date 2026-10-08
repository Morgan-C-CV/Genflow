"""Cheap access to gallery images.

Resolving an image for the UI must not require the embedding stack, which takes
tens of seconds to initialise. ``metadata.json`` is 1:1 with the search
dataframe (verified: same length, same ids, same local paths), so it is a safe
and cheap index source — and the showcase pages need no search engine at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.core.config import settings

_catalog: Optional[List[Dict[str, Any]]] = None

# Gallery originals average ~1.8 MB (some exceed 20 MB), so grids are served
# downscaled. Thumbnails are cached next to the backend, not regenerated.
THUMBNAIL_DIR = Path(__file__).resolve().parents[3] / ".thumbnails"
THUMBNAIL_WIDTH = 640


class GalleryImageNotFound(KeyError):
    """Raised when an index does not exist in the gallery."""


def _load() -> List[Dict[str, Any]]:
    global _catalog
    if _catalog is None:
        path = Path(settings.METADATA_PATH) if settings.METADATA_PATH else None
        if path is None or not path.is_file():
            _catalog = []
        else:
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                records = []
            _catalog = records if isinstance(records, list) else []
    return _catalog


def total() -> int:
    return len(_load())


def record_at(index: int) -> Dict[str, Any]:
    records = _load()
    if not isinstance(index, int) or index < 0 or index >= len(records):
        raise GalleryImageNotFound(f"Gallery index out of range: {index}")
    row = records[index]
    return {
        "index": index,
        "id": str(row.get("id", "") or ""),
        "local_path": str(row.get("local_path", "") or ""),
        "prompt": str(row.get("prompt", "") or ""),
    }


def listing(limit: int, offset: int = 0) -> List[Dict[str, Any]]:
    """Return image descriptors with URLs, in catalogue (index) order."""
    records = _load()
    window = records[offset : offset + limit] if limit > 0 else records[offset:]
    return [
        {
            "index": position,
            "id": str(row.get("id", "") or ""),
            "url": image_url(position),
        }
        for position, row in enumerate(window, start=offset)
    ]


def image_url(index: int, width: int = THUMBNAIL_WIDTH) -> str:
    """URL for an index, downscaled by default."""
    if width and width > 0:
        return f"/api/v1/gallery/image/{index}?w={width}"
    return f"/api/v1/gallery/image/{index}"


def thumbnail_path(index: int, width: int) -> Path:
    """Build (once) and return a downscaled JPEG for an index."""
    source = image_path(index)
    width = max(64, min(int(width), 4096))
    target_dir = THUMBNAIL_DIR
    target = target_dir / f"{index}_{width}_{source.stat().st_mtime_ns}.jpg"
    if target.is_file():
        return target

    from PIL import Image

    target_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((width, width), Image.LANCZOS)
            image.save(temporary, "JPEG", quality=82, optimize=True)
        # Atomic swap so concurrent requests never see a partial file.
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target


def image_path(index: int) -> Path:
    """Resolve an index to a file, refusing anything outside GALLERY_DIR."""
    row = record_at(index)
    gallery_dir = Path(settings.GALLERY_DIR) if settings.GALLERY_DIR else None
    if gallery_dir is None or not gallery_dir.is_dir():
        raise FileNotFoundError("GALLERY_DIR is not configured.")

    name = row["local_path"] or f"image_{row['id']}.jpg"
    candidate = (gallery_dir / name).resolve()
    root = gallery_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise PermissionError("Resolved image path escapes the gallery directory.")
    if not candidate.is_file():
        raise FileNotFoundError(f"Image file not found: {name}")
    return candidate


def media_type(path: Path) -> str:
    """Sniff the real image type; gallery files carry misleading extensions."""
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
    except OSError:
        return "application/octet-stream"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith(b"GIF8"):
        return "image/gif"
    return "application/octet-stream"
