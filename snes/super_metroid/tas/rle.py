"""SNES-12 RLE seed format for Super Metroid TAS slices.

JSON shape (mirrors SMB ``nes9_rle``, SM ``load_rle_json``)::

    {
      "format": "snes12_rle",
      "route_id": "...",
      "num_frames": N,
      "source": "...",
      "segments": [{"n": frames, "b": ["RIGHT", "B"]}, ...]
    }

Button names are :data:`retro_harness.controls.SNES_BUTTON_NAMES`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retro_harness.actions import SNES_ACTION_SIZE
from retro_harness.controls import (
    SNES_BUTTON_NAME_TO_INDEX,
    SNES_BUTTON_NAMES,
)

FORMAT = "snes12_rle"


def _frame_key(frame: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if len(frame) != SNES_ACTION_SIZE:
        raise ValueError(f"expected {SNES_ACTION_SIZE} buttons, got {len(frame)}")
    return tuple(1 if int(v) else 0 for v in frame)


def compress_snes12_rle(frames: list[list[int]]) -> list[dict[str, Any]]:
    """Compress SNES-12 frames to ``[{n, b}, ...]`` runs."""
    if not frames:
        return []
    segments: list[dict[str, Any]] = []
    cur = _frame_key(frames[0])
    n = 1
    for fr in frames[1:]:
        key = _frame_key(fr)
        if key == cur:
            n += 1
            continue
        segments.append(
            {
                "n": n,
                "b": [SNES_BUTTON_NAMES[i] for i, v in enumerate(cur) if v],
            }
        )
        cur = key
        n = 1
    segments.append(
        {
            "n": n,
            "b": [SNES_BUTTON_NAMES[i] for i, v in enumerate(cur) if v],
        }
    )
    return segments


def expand_snes12_rle(data: dict[str, Any] | list[Any]) -> list[list[int]]:
    """Expand a seed dict or raw segment list to SNES-12 frames."""
    if isinstance(data, dict):
        segments = data.get("segments")
        if segments is None:
            raise ValueError("snes12_rle seed missing 'segments'")
    else:
        segments = data

    frames: list[list[int]] = []
    for i, row in enumerate(segments):
        if isinstance(row, dict):
            n = int(row["n"])
            buttons = row.get("b") or row.get("buttons") or ()
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            n = int(row[0])
            buttons = row[1] or ()
        else:
            raise ValueError(f"bad RLE row {i}: {row!r}")
        action = [0] * SNES_ACTION_SIZE
        for name in buttons:
            key = str(name).upper()
            if key in ("", "IDLE", "NOOP", "."):
                continue
            if key not in SNES_BUTTON_NAME_TO_INDEX:
                raise ValueError(f"unknown button {name!r} in row {i}")
            idx = SNES_BUTTON_NAME_TO_INDEX[key]
            if idx is not None:
                action[idx] = 1
        for _ in range(n):
            frames.append(list(action))
    return frames


def frames_to_snes12_rle_payload(
    frames: list[list[int]],
    *,
    route_id: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``snes12_rle`` seed dict from raw frames."""
    payload: dict[str, Any] = {
        "format": FORMAT,
        "route_id": route_id,
        "game_name": "SuperMetroid-Snes",
        "num_frames": len(frames),
        "source": source,
        "segments": compress_snes12_rle(frames),
    }
    if extra:
        payload.update(extra)
    return payload


def load_snes12_rle_seed(path: Path | str) -> dict[str, Any]:
    """Load a snes12_rle JSON seed from disk."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"seed must be object: {path}")
    return data


def write_snes12_rle_seed(
    path: Path | str,
    payload: dict[str, Any],
    *,
    compact: bool | None = None,
) -> Path:
    """Write seed JSON and return path.

    Uses compact separators when ``num_frames`` ≥ 5000 (or ``compact=True``)
    so full-movie seeds stay smaller on disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(payload.get("num_frames") or 0)
    use_compact = compact if compact is not None else n >= 5_000
    if use_compact:
        text = json.dumps(payload, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return path
