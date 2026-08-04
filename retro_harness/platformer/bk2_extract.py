"""Extract action sequences from bk2 recordings with correct button mapping.

BK2 files are zip archives containing an 'Input Log.txt'. The SNES button
order in BK2 is reversed from the env logical order. This module handles
the conversion and maps raw buttons to the closest action in a given table.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from retro_harness.platformer.actions import (
    NUM_BUTTONS,
    DEFAULT_PLATFORMER_ACTIONS,
    buttons_to_action_index,
)

# Default BK2-to-ENV mapping: SNES standard reversed
# BK2 hardware:  [R, L, X, A, Right, Left, Down, Up, Start, Select, Y, B]
# ENV logical:   [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
DEFAULT_BK2_TO_ENV = [11 - i for i in range(12)]


def extract_raw_actions_from_bk2(
    bk2_path: Path,
    bk2_to_env: list[int] | None = None,
) -> list[list[int]]:
    """Extract raw 12-element button arrays from a bk2 file.

    Args:
        bk2_path: Path to the .bk2 recording file.
        bk2_to_env: Button index mapping. Defaults to SNES standard.

    Returns:
        List of frames, each a 12-element button array in env order.
    """
    mapping = bk2_to_env or DEFAULT_BK2_TO_ENV
    bk2_path = Path(bk2_path)
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

        # P1 buttons are in group index 1 (12 chars)
        p1_chars = groups[1] if len(groups) > 1 else ""
        if len(p1_chars) < NUM_BUTTONS:
            continue

        env_action = [0] * NUM_BUTTONS
        for bk2_idx in range(NUM_BUTTONS):
            char = p1_chars[bk2_idx]
            pressed = char != "."
            env_idx = mapping[bk2_idx]
            env_action[env_idx] = 1 if pressed else 0

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
