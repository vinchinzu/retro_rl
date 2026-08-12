"""Immutable segment archive for guided_human takes.

When the same ``--name`` is reused (resume / next segment), the previous task
JSON would otherwise be overwritten and **button bodies** for earlier seams
would be lost. Anchors on disk may survive as orphans, but open-loop needs
frames.

On collision, archive the prior take under::

    tasks/<name>_segments/s{N}/
      tape.json           full frames + trace
      anchors.json        index snapshot at archive time
      extract.json        optional
      run_timing.json     optional
      join.json           seam metadata for stitch / compose

State dumps stay in ``tasks/<name>_anchors/`` (shared by mtime sessions for
timing stitch). Only JSON sidecars are copied into the segment folder.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_SEG_DIR_RE = re.compile(r"^s(\d+)$")


def segments_dir_for(task_path: Path | str) -> Path:
    """``tasks/foo_segments`` next to ``tasks/foo.json``."""
    path = Path(task_path)
    return path.with_name(path.stem + "_segments")


def list_segment_ids(segments_dir: Path | str) -> list[int]:
    root = Path(segments_dir)
    if not root.is_dir():
        return []
    ids: list[int] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = _SEG_DIR_RE.match(child.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def next_segment_id(segments_dir: Path | str) -> int:
    ids = list_segment_ids(segments_dir)
    return (max(ids) + 1) if ids else 0


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def build_join_metadata(
    task_data: Mapping[str, Any],
    *,
    segment_id: int,
    task_path: Path,
    archived_from: str,
) -> dict[str, Any]:
    """Seam contract row for later hop-compose / stitch."""
    meta = task_data.get("metadata") if isinstance(task_data.get("metadata"), Mapping) else {}
    meta = meta or {}
    end_fp = meta.get("end_fingerprint") if isinstance(meta.get("end_fingerprint"), Mapping) else None
    return {
        "schemaVersion": 1,
        "kind": "super_metroid_segment_join",
        "segment_id": int(segment_id),
        "segment_name": f"s{int(segment_id)}",
        "task_name": str(task_data.get("name") or task_path.stem),
        "archived_from": archived_from,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "start_state": task_data.get("start_state"),
        "power_on": bool(meta.get("power_on")),
        "frame_count": int(
            task_data.get("frame_count")
            or len(task_data.get("frames") or [])
            or 0
        ),
        "end_fingerprint": dict(end_fp) if end_fp else None,
        "source_path": meta.get("source_path"),
        "route": meta.get("route"),
        "note": (
            "Immutable button tape for this segment. Compose is pin→body per hop "
            "(not frame-append across seams)."
        ),
    }


def archive_existing_take(
    task_path: Path | str,
    *,
    force: bool = False,
) -> Path | None:
    """If *task_path* exists with frames, copy it into the next segment slot.

    Returns the segment directory written, or ``None`` if nothing to archive.
    Does not delete the live task path — caller overwrites on next F5.
    """
    path = Path(task_path)
    if not path.is_file():
        return None

    data = _safe_load_json(path)
    if data is None:
        return None
    frames = data.get("frames") or []
    if not frames and not force:
        return None

    seg_root = segments_dir_for(path)
    seg_id = next_segment_id(seg_root)
    dest = seg_root / f"s{seg_id}"
    dest.mkdir(parents=True, exist_ok=True)

    # Primary tape body (buttons + trace).
    tape_dest = dest / "tape.json"
    shutil.copy2(path, tape_dest)

    # Anchors index snapshot (state files stay in shared anchors dir).
    anchors_idx = path.with_name(path.stem + "_anchors.json")
    if anchors_idx.is_file():
        shutil.copy2(anchors_idx, dest / "anchors.json")

    for suffix, out_name in (
        ("_extract.json", "extract.json"),
        ("_run_timing.json", "run_timing.json"),
        ("_stitched.json", "stitched.json"),
    ):
        side = path.with_name(path.stem + suffix)
        if side.is_file():
            shutil.copy2(side, dest / out_name)

    # Per-hop SNES-12 bodies (hill-climb / bank seeds) — same take as tape.
    hops_src = path.with_name(path.stem + "_hops")
    hop_body_count = 0
    if hops_src.is_dir():
        hops_dest = dest / "hops"
        if hops_dest.exists():
            shutil.rmtree(hops_dest)
        shutil.copytree(hops_src, hops_dest)
        hop_body_count = sum(1 for p in hops_dest.glob("*.json") if p.is_file())

    join = build_join_metadata(
        data,
        segment_id=seg_id,
        task_path=path,
        archived_from=str(path.resolve()),
    )
    join["hop_bodies"] = hop_body_count
    join["hops_dir"] = "hops" if hop_body_count else None
    (dest / "join.json").write_text(
        json.dumps(join, indent=2) + "\n", encoding="utf-8"
    )

    # Refresh segment registry (list of archived segments + current pointer).
    registry = {
        "schemaVersion": 1,
        "kind": "super_metroid_segment_registry",
        "task": str(path.name),
        "task_stem": path.stem,
        "segments": [],
    }
    reg_path = seg_root / "registry.json"
    if reg_path.is_file():
        prev = _safe_load_json(reg_path)
        if prev and isinstance(prev.get("segments"), list):
            registry["segments"] = list(prev["segments"])
    registry["segments"].append(
        {
            "id": seg_id,
            "dir": f"s{seg_id}",
            "frame_count": join["frame_count"],
            "start_state": join.get("start_state"),
            "end_fingerprint": join.get("end_fingerprint"),
            "archived_at": join["archived_at"],
            "tape": f"s{seg_id}/tape.json",
            "hop_bodies": hop_body_count,
            "hops": f"s{seg_id}/hops" if hop_body_count else None,
        }
    )
    reg_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return dest


def list_archived_tapes(task_path: Path | str) -> list[dict[str, Any]]:
    """Rows describing archived segments (tape path + join meta)."""
    path = Path(task_path)
    seg_root = segments_dir_for(path)
    rows: list[dict[str, Any]] = []
    for seg_id in list_segment_ids(seg_root):
        dest = seg_root / f"s{seg_id}"
        tape = dest / "tape.json"
        join = _safe_load_json(dest / "join.json") or {}
        rows.append(
            {
                "segment_id": seg_id,
                "dir": str(dest),
                "tape": str(tape) if tape.is_file() else None,
                "frame_count": join.get("frame_count"),
                "start_state": join.get("start_state"),
                "end_fingerprint": join.get("end_fingerprint"),
                "join": join,
            }
        )
    return rows
