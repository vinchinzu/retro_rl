"""SNES-12 RLE seed format for ALttP TAS slices.

Same JSON shape as Super Metroid. ``game_name`` is ``Zelda3-Snes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from super_metroid.tas.rle import (
    FORMAT,
    compress_snes12_rle,
    expand_snes12_rle,
    load_snes12_rle_seed,
    write_snes12_rle_seed,
)
from super_metroid.tas.rle import (
    frames_to_snes12_rle_payload as _sm_payload,
)

GAME_NAME = "Zelda3-Snes"

__all__ = [
    "FORMAT",
    "GAME_NAME",
    "compress_snes12_rle",
    "expand_snes12_rle",
    "frames_to_snes12_rle_payload",
    "load_snes12_rle_seed",
    "write_snes12_rle_seed",
]


def frames_to_snes12_rle_payload(
    frames: list[list[int]],
    *,
    route_id: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``snes12_rle`` seed with ALttP ``game_name``."""
    payload = _sm_payload(
        frames,
        route_id=route_id,
        source=source,
        extra=extra,
    )
    payload["game_name"] = GAME_NAME
    return payload


def write_payload(path: Path | str, payload: dict[str, Any]) -> Path:
    """Write seed JSON (compact when long)."""
    return write_snes12_rle_seed(path, payload)
