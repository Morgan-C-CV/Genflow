"""Gallery image serving.

Deliberately independent of the search/embedding stack so a page can render
thumbnails immediately instead of waiting for the model to warm up.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.modules import gallery_catalog

router = APIRouter()


class GalleryImageRef(BaseModel):
    index: int
    id: str = ""
    url: str


class GalleryListingResponse(BaseModel):
    total: int
    offset: int
    images: List[GalleryImageRef] = Field(default_factory=list)


@router.get("/images", response_model=GalleryListingResponse)
def list_images(
    limit: int = Query(default=48, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    return GalleryListingResponse(
        total=gallery_catalog.total(),
        offset=offset,
        images=gallery_catalog.listing(limit=limit, offset=offset),
    )


@router.get("/image/{index}")
def gallery_image(index: int, w: int = Query(default=0, ge=0, le=4096)):
    """Serve the original (``w=0``) or a cached downscaled thumbnail."""
    try:
        if w > 0:
            path = gallery_catalog.thumbnail_path(index, w)
            return FileResponse(path, media_type="image/jpeg")
        path = gallery_catalog.image_path(index)
    except gallery_catalog.GalleryImageNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, media_type=gallery_catalog.media_type(path))
