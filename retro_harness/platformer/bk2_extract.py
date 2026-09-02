"""Extract action sequences from bk2 recordings.

SNES movies go through :func:`retro_harness.bk2.parse_bk2` (LogKey, then
legacy reverse). Explicit ``bk2_to_env`` maps stay for NES 9-button files.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from retro_harness.bk2 import LEGACY_BK2_TO_ENV, parse_bk2
from retro_harness.platformer.actions import (
    NUM_BUTTONS,
    DEFAULT_PLATFORMER_ACTIONS,
    buttons_to_action_index,
)

# Fallback when a caller still passes no LogKey and no map (legacy reverse).
DEFAULT_BK2_TO_ENV = list(LEGACY_BK2_TO_ENV)


def extract_raw_actions_from_bk2(
    bk2_path: Path,
    bk2_to_env: list[int] | None = None,
) -> list[list[int]]:
    """Extract raw button arrays from a bk2 file.

    ``bk2_to_env is None`` (the SNES default) reads the LogKey, then falls
    back to the legacy reversed hardware order. Pass an explicit map for
    NES 9-button movies that are not SNES-12 LogKey.
    """
    bk2_path = Path(bk2_path)
    if bk2_to_env is None:
        return parse_bk2(bk2_path).frames

    mapping = bk2_to_env
    raw_frames: list[list[int]] = []
    with zipfile.ZipFile(bk2_path, "r") as zf:
        with zf.open("Input Log.txt") as f:
            lines = f.read().decode("utf-8").splitlines()

    for line in lines:
        line = line.strip()
        if not line or not line.startswith("|") or line.startswith("["):
            continue

        groups = [g for g in line.split("|") if g]
        if len(groups) < 2:
            continue

        p1_chars = groups[1] if len(groups) > 1 else ""
        width = min(len(mapping), len(p1_chars))
        if width == 0:
            continue

        env_action = [0] * NUM_BUTTONS
        for bk2_idx in range(width):
            if p1_chars[bk2_idx] == ".":
                continue
            env_idx = mapping[bk2_idx]
            if 0 <= env_idx < NUM_BUTTONS:
                env_action[env_idx] = 1
        raw_frames.append(env_action)

    return raw_frames


def extract_action_indices_from_bk2(
    bk2_path: Path,
    action_table: list[list[int]] | None = None,
    bk2_to_env: list[int] | None = None,
) -> list[int]:
    """Extract a sequence of action indices from a bk2 file.

    Each frame's raw buttons are mapped to the closest action in the table.
    """
    raw = extract_raw_actions_from_bk2(bk2_path, bk2_to_env=bk2_to_env)
    return [buttons_to_action_index(frame, action_table=action_table) for frame in raw]


def save_actions(actions: list[int], output_path: Path, metadata: dict | None = None) -> None:
    """Save action sequence to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "actions": actions,
        "num_frames": len(actions),
    }
    if metadata:
        data["metadata"] = metadata
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(actions)} frames to {output_path}")


def load_actions(path: Path) -> list[int]:
    """Load action sequence from JSON file."""
    data = json.loads(Path(path).read_text())
    return data["actions"]


def load_raw_buttons(path: Path) -> list[list[int]] | None:
    """Load raw 12-button arrays from a recording.

    Checks for a companion ``*_raw.json`` file first, then falls back to
    ``raw_buttons`` embedded in the main file.  Returns None if no raw
    data is available.
    """
    path = Path(path)
    # Try companion raw file
    raw_path = path.with_name(path.stem + "_raw.json")
    if raw_path.exists():
        data = json.loads(raw_path.read_text())
        if "raw_buttons" in data:
            return data["raw_buttons"]
    # Try embedded in main file
    data = json.loads(path.read_text())
    if "raw_buttons" in data:
        return data["raw_buttons"]
    return None
