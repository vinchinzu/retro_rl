from __future__ import annotations

from pathlib import Path
import gzip
import json
from typing import Iterable


def ensure_gzip_state(game_dir: Path, game: str, state: str) -> None:
    """Ensure a .state file is gzip-compressed for stable-retro."""
    if state.lower() == "none":
        return
    state_path = game_dir / "custom_integrations" / game / f"{state}.state"
    if not state_path.exists():
        return
    with state_path.open("rb") as f:
        header = f.read(2)
    if header == b"\x1f\x8b":
        return
    raw = state_path.read_bytes()
    with gzip.open(state_path, "wb") as f:
        f.write(raw)


def append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def iter_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def find_latest_recording(record_dir: Path, game: str | None = None) -> Path | None:
    if not record_dir.exists():
        return None
    candidates = []
    for path in record_dir.glob("*.bk2"):
        if game and game not in path.name:
            continue
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_latest_recording_from_manifest(
    manifest_path: Path,
    record_dir: Path,
    *,
    level_id: int | None = None,
    level_name: str | None = None,
) -> Path | None:
    entries = iter_jsonl(manifest_path)
    if not entries:
        return None
    norm_name = level_name.lower() if level_name else None
    candidates = []
    for entry in entries:
        if level_id is not None:
            if entry.get("start_level_id") != level_id:
                continue
        if norm_name is not None:
            entry_name = entry.get("start_level_name")
            if not isinstance(entry_name, str) or entry_name.lower() != norm_name:
                continue
        rec_name = entry.get("recording")
        if not isinstance(rec_name, str):
            continue
        rec_path = record_dir / rec_name
        if not rec_path.exists():
            continue
        candidates.append((entry.get("recording_mtime", 0), rec_path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
