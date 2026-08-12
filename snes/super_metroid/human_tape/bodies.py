"""Export per-hop SNES-12 bodies from a guided_human tape.

Each body is the open-loop unit for hop-replay, hill-climb seed, and skill
bank ``body_path``. Format matches trim seed export.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def _frame_vec(frame: Sequence[int] | Any) -> list[int]:
    return [int(x) for x in frame]


def _safe_slug(text: str, *, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    s = s.strip("_") or "hop"
    return s[:max_len]


def hop_bodies_dir(task_path: Path | str) -> Path:
    path = Path(task_path)
    return path.with_name(path.stem + "_hops")


def export_hop_body(
    frames: Sequence[Sequence[int]],
    hop: Mapping[str, Any],
    out_path: Path | str,
    *,
    source_task: str | None = None,
    entry_anchor: str | None = None,
    hop_key: str | None = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write one hop body JSON (frames slice + meta)."""
    start_i = int(hop.get("start_index") if hop.get("start_index") is not None else 0)
    end_i = int(hop.get("end_index") if hop.get("end_index") is not None else start_i)
    n = len(frames)
    start_i = max(0, min(start_i, n - 1 if n else 0))
    end_i = max(start_i, min(end_i, n - 1 if n else 0))
    body_frames = [_frame_vec(frames[i]) for i in range(start_i, end_i + 1)] if n else []

    meta: dict[str, Any] = {
        "kind": "super_metroid_hop_body",
        "schemaVersion": 1,
        "source_task": source_task,
        "hop_index": hop.get("index"),
        "room": hop.get("room") or (
            f"0x{int(hop['room_id']):04X}" if hop.get("room_id") is not None else None
        ),
        "name": hop.get("name"),
        "start_index": start_i,
        "end_index": end_i,
        "dwell": len(body_frames),
        "entry_anchor": entry_anchor,
        "hop_key": hop_key,
        "end_xy": hop.get("end_xy"),
        "items": hop.get("items"),
    }
    if extra_meta:
        meta.update(dict(extra_meta))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frames": body_frames,
        "frame_count": len(body_frames),
        "meta": meta,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def export_hop_bodies(
    task_path: Path | str,
    hops: Sequence[Mapping[str, Any]],
    *,
    frames: Sequence[Sequence[int]] | None = None,
    body_dir: Path | str | None = None,
    hop_keys: Sequence[str | None] | None = None,
    entry_anchors: Sequence[str | None] | None = None,
) -> list[Path]:
    """Export all hops under ``<stem>_hops/hop_{i:02d}_*.json``.

    Returns paths in hop order (one per hop with a valid index span).
    """
    path = Path(task_path)
    if frames is None:
        from super_metroid.human_tape.hops import load_task_json

        data = load_task_json(path)
        frames = list(data.get("frames") or [])
    out_dir = Path(body_dir) if body_dir is not None else hop_bodies_dir(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i, hop in enumerate(hops):
        idx = int(hop.get("index", i))
        room = hop.get("room") or "room"
        name = _safe_slug(str(hop.get("name") or room))
        out = out_dir / f"hop_{idx:02d}_{name}.json"
        key = None
        if hop_keys is not None and i < len(hop_keys):
            key = hop_keys[i]
        anchor = None
        if entry_anchors is not None and i < len(entry_anchors):
            anchor = entry_anchors[i]
        export_hop_body(
            frames,
            hop,
            out,
            source_task=str(path),
            entry_anchor=anchor,
            hop_key=key,
        )
        written.append(out)

    # Do not purge other hop_*.json here. Prior takes / alternate lines stay
    # on disk for hill-climb seeds and "better jump, slower overall" layers.
    # Immutable history also lives under tasks/<name>_segments/sN/hops/ when
    # the same --name is reused. Compose/replay select by hop index + anchors,
    # not by wiping the directory.

    return written
