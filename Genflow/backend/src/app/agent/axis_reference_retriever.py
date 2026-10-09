"""Retrieve real gallery references along a modification axis.

The shift/modify stage does not synthesise proposals. For every dissatisfaction
axis the LLM interpreted, this projects an axis-direction query into the PBO
space and picks three gallery records that sit at increasing distance from the
current result *along that direction*: near, mid and far.

Concretely, with ``c`` the current result's vector, ``q`` the axis query vector
and ``d = normalize(q - c)`` the axis direction, each record's position is the
signed projection ``t = <v - c, d>``. Records with ``t > 0`` have moved toward
the axis intent; the band they land in is what near/mid/far name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np


BANDS = ("near", "mid", "far")


@dataclass
class AxisReference:
    """One gallery record proposed as a reference for one axis."""

    probe_id: str = ""
    axis: str = ""
    band: str = ""
    gallery_index: int = -1
    prompt: str = ""
    negative_prompt: str = ""
    model: str = ""
    sampler: str = ""
    cfgscale: float = 0.0
    steps: float = 0.0
    clipskip: float = 0.0
    loras: str = ""
    # Cosine similarity between the record and the axis query.
    alignment: float = 0.0
    # Signed distance from the current result along the axis direction.
    axis_distance: float = 0.0
    retrieval_rationale: str = ""


def _unit(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return array / norm


def _field(row: Any, name: str, default: Any) -> Any:
    if row is None:
        return default
    try:
        value = row.get(name, default)
    except Exception:  # pragma: no cover - defensive for odd frame shapes
        return default
    if value is None:
        return default
    return value


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    # NaN/inf would be emitted as invalid JSON and break the client's parse.
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def retrieve_axis_references(
    *,
    search_engine: Any,
    axis: str,
    query_text: str,
    current_index: Optional[int] = None,
    per_axis: int = 3,
    exclude: Sequence[int] = (),
    numeric_profile: Optional[dict] = None,
) -> List[AxisReference]:
    """Return ``per_axis`` references for ``axis``, ordered near -> far."""
    if search_engine is None or not query_text.strip():
        return []

    space = getattr(search_engine, "pbo_space", None)
    if space is None:
        return []
    space = np.asarray(space, dtype=float)
    if space.ndim != 2 or space.shape[0] == 0:
        return []

    profile = dict(numeric_profile or {})
    try:
        query_vector = search_engine.transform_query_to_pbo(query_text, **profile)
    except Exception:
        return []
    if query_vector is None:
        return []

    total = space.shape[0]
    anchor = current_index if isinstance(current_index, int) and 0 <= current_index < total else None
    current = space[anchor] if anchor is not None else space.mean(axis=0)

    direction = _unit(np.asarray(query_vector, dtype=float).reshape(-1) - current)
    if not np.any(direction):
        return []

    query_unit = _unit(query_vector)
    excluded = {int(item) for item in exclude}
    if anchor is not None:
        excluded.add(anchor)

    scored: List[tuple[int, float, float]] = []
    for index in range(total):
        if index in excluded:
            continue
        vector = space[index]
        distance = float(np.dot(vector - current, direction))
        alignment = float(np.dot(_unit(vector), query_unit))
        scored.append((index, distance, alignment))

    if not scored:
        return []

    ordered = sorted(scored, key=lambda item: item[1])
    forward = [item for item in ordered if item[1] > 0.0]
    pool = forward if len(forward) >= per_axis else ordered

    # Split the axis span into contiguous bands and let the best-aligned record
    # represent each one, so a band is never filled by an outlier.
    picked: List[tuple[int, float, float]] = []
    count = len(pool)
    if count <= per_axis:
        picked = list(pool)
    else:
        for position in range(per_axis):
            low = (position * count) // per_axis
            high = ((position + 1) * count) // per_axis
            chunk = pool[low:high]
            if not chunk:
                continue
            picked.append(max(chunk, key=lambda item: item[2]))

    frame = getattr(search_engine, "df", None)
    references: List[AxisReference] = []
    for position, (index, distance, alignment) in enumerate(picked):
        band = BANDS[position] if position < len(BANDS) else BANDS[-1]
        row = frame.iloc[index] if frame is not None and index < len(frame) else None
        references.append(
            AxisReference(
                probe_id=f"ref_{axis}_{band}",
                axis=axis,
                band=band,
                gallery_index=int(index),
                prompt=str(_field(row, "prompt", "")),
                negative_prompt=str(_field(row, "negative_prompt", "")),
                model=str(_field(row, "model", "")),
                sampler=str(_field(row, "sampler", "")),
                cfgscale=_to_float(_field(row, "cfgscale", 0.0)),
                steps=_to_float(_field(row, "steps", 0.0)),
                clipskip=_to_float(_field(row, "clipskip", 0.0)),
                loras=str(_field(row, "loras", "")),
                alignment=_to_float(alignment),
                axis_distance=_to_float(distance),
                retrieval_rationale=(
                    f"{axis}: {band} band, {distance:+.3f} along the axis direction, "
                    f"alignment {alignment:+.3f}"
                ),
            )
        )
    return references
