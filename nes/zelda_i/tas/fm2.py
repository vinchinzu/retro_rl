"""FCEUX ``.fm2`` import for Legend of Zelda seeds.

Reuses the NES-generic SMB FM2 parser (same button layout). Builds
``nes9_rle`` payloads tagged for ``LegendOfZelda-Nes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smb.tas.fm2 import Fm2Movie, fm2_to_nes9_frames, parse_fm2

from zelda_i.paths import GAME

__all__ = [
    "Fm2Movie",
    "fm2_to_nes9_frames",
    "frames_to_nes9_rle_payload",
    "parse_fm2",
]


def frames_to_nes9_rle_payload(
    frames: list[list[int]],
    *,
    route_id: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``nes9_rle`` seed dict from raw NES-9 frames (no verify)."""
    from smb.policy import compress_nes9_rle

    payload: dict[str, Any] = {
        "format": "nes9_rle",
        "route_id": route_id,
        "game_name": GAME,
        "num_frames": len(frames),
        "source": source,
        "segments": compress_nes9_rle(frames),
    }
    if extra:
        payload.update(extra)
    return payload


def export_rle_seed(
    path: Path | str,
    *,
    out: Path | str,
    route_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse FM2 → write ``nes9_rle`` JSON seed."""
    path = Path(path)
    out = Path(out)
    movie = parse_fm2(path)
    payload = frames_to_nes9_rle_payload(
        movie.frames,
        route_id=route_id,
        source=str(path),
        extra={
            "fm2_author": movie.author,
            "fm2_rom": movie.rom_filename,
            "fm2_rerecords": movie.header.get("rerecordCount"),
            **(extra or {}),
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
