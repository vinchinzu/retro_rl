"""Autobot policies for Super Mario Bros. segments.

Frame-perfect NES button replay for 1-1 (seed ``smb/models/smb_1_1_clear.json``).

- **M3 isolated:** start ``Level1_1`` (x≈40), no settle.
- **M4 natural-entry:** power-on boot, then **1 idle frame** settle before
  replaying the same seed (phase alignment; settle=0/2 desyncs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from retro_harness.nes import NES_ACTION_SIZE, nes_idle_action
from retro_harness.input_script import FrameAction

GAME_DIR = Path(__file__).resolve().parent
DEFAULT_1_1_SEED = GAME_DIR / "models" / "smb_1_1_clear.json"
DEFAULT_1_2_WARP_SEED = GAME_DIR / "models" / "smb_1_2_warp_w4.json"
# State-gated 1-2 fragments (control-relative underground + optional surface macro).
DEFAULT_1_2_REACTIVE_FRAGMENTS = GAME_DIR / "models" / "smb_1_2_reactive_fragments.json"
DEFAULT_WARP_SUFFIX_SEED = GAME_DIR / "models" / "smb_warp_mid_to_ending.json"
# Natural-entry 4-2 fragment used by the optimized continuous fold.
DEFAULT_FAST_4_2_SEED = GAME_DIR / "models" / "smb_4_2_fast_w8.json"
# Continuous Level1_1 → ending (no mid-1-2 splice). Built by fold_continuous_policy.
DEFAULT_CONTINUOUS_SEED = GAME_DIR / "models" / "smb_1_1_to_ending.json"
# Idle frames after Level1_1 before DEFAULT_CONTINUOUS_SEED (phase align).
CONTINUOUS_SETTLE_FRAMES = 14
# Power-on Clean: fixed boot script frames then idle, then same continuous seed.
# Verified 3/3: boot=350 + settle=16 → 8-4 ending, zero mid-attempt loads.
POWERON_BOOT_FRAMES = 350
POWERON_SETTLE_FRAMES = 16


def load_nes9_rle_seed(path: Path | str) -> dict[str, Any]:
    """Load a compact ``nes9_rle`` action seed."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "nes9_rle":
        raise ValueError(f"unsupported seed format: {data.get('format')!r}")
    if not data.get("segments"):
        raise ValueError(f"seed has no segments: {path}")
    return data


def expand_nes9_rle(data: dict[str, Any]) -> list[list[int]]:
    """Expand RLE segments to a list of 9-button NES frames."""
    frames: list[list[int]] = []
    for seg in data["segments"]:
        buttons = [int(b) for b in seg["b"]]
        if len(buttons) != 9:
            raise ValueError(f"expected 9 buttons, got {len(buttons)}")
        n = int(seg["n"])
        if n <= 0:
            raise ValueError(f"non-positive RLE count: {n}")
        frames.extend([buttons] * n)
    return frames


def compress_nes9_rle(frames: list[list[int]]) -> list[dict[str, Any]]:
    """Compress NES button frames into ``{"b": buttons, "n": count}`` rows."""
    segments: list[dict[str, Any]] = []
    for raw in frames:
        buttons = [int(b) for b in raw[:NES_ACTION_SIZE]]
        if len(buttons) != NES_ACTION_SIZE:
            raise ValueError(f"expected at least 9 buttons, got {len(raw)}")
        if segments and segments[-1]["b"] == buttons:
            segments[-1]["n"] += 1
        else:
            segments.append({"b": buttons, "n": 1})
    return segments


def frames_to_actions(
    frames: list[list[int]],
    *,
    action_size: int = NES_ACTION_SIZE,
) -> Iterator[np.ndarray]:
    """Yield env action arrays (pad/truncate to ``action_size``)."""
    for frame in frames:
        buttons = list(frame[:action_size])
        if len(buttons) < action_size:
            buttons.extend([0] * (action_size - len(buttons)))
        yield np.array(buttons, dtype=np.int8)


@dataclass
class Nes9ReplayPolicy:
    """Generic NES-9 RLE seed replay (shared by 1-1 and 1-2 warp)."""

    seed_path: Path
    action_size: int = NES_ACTION_SIZE
    frames: list[list[int]] = field(default_factory=list, repr=False)
    index: int = 0
    exhausted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frames:
            data = load_nes9_rle_seed(self.seed_path)
            self.frames = expand_nes9_rle(data)
            self.metadata = {
                "seed_path": str(self.seed_path),
                "num_frames": len(self.frames),
                "start_state": data.get("start_state"),
                "level_id": data.get("level_id"),
                "target": data.get("target"),
                "source": data.get("source"),
            }

    def reset(self) -> None:
        self.index = 0
        self.exhausted = False

    @property
    def remaining(self) -> int:
        return max(0, len(self.frames) - self.index)

    def step(self) -> FrameAction:
        if self.index >= len(self.frames):
            self.exhausted = True
            idle = np.asarray(nes_idle_action(), dtype=np.int8)
            if idle.shape[0] != self.action_size:
                idle = np.zeros(self.action_size, dtype=np.int8)
            return FrameAction(idle, "seed_exhausted")
        buttons = list(self.frames[self.index][: self.action_size])
        if len(buttons) < self.action_size:
            buttons.extend([0] * (self.action_size - len(buttons)))
        self.index += 1
        return FrameAction(np.array(buttons, dtype=np.int8), f"replay_{self.index}")

    def report(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "index": self.index,
            "exhausted": self.exhausted,
            "remaining": self.remaining,
        }


@dataclass
class Level11ReplayPolicy:
    """Frame-perfect 1-1 clear from a verified seed (isolated checkpoint).

    Not robust to natural-entry desync — use only from the documented start
    state (default ``Level1_1``, Mario x≈40).
    """

    seed_path: Path = DEFAULT_1_1_SEED
    action_size: int = NES_ACTION_SIZE
    frames: list[list[int]] = field(default_factory=list, repr=False)
    index: int = 0
    exhausted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frames:
            data = load_nes9_rle_seed(self.seed_path)
            self.frames = expand_nes9_rle(data)
            self.metadata = {
                "seed_path": str(self.seed_path),
                "num_frames": len(self.frames),
                "start_state": data.get("start_state"),
                "verified_clear_frames": data.get("verified_clear_frames"),
                "source": data.get("source"),
            }

    def reset(self) -> None:
        self.index = 0
        self.exhausted = False

    @property
    def remaining(self) -> int:
        return max(0, len(self.frames) - self.index)

    def step(self) -> FrameAction:
        """Next controller action, or idle once the seed is exhausted."""
        if self.index >= len(self.frames):
            self.exhausted = True
            idle = np.asarray(nes_idle_action(), dtype=np.int8)
            if idle.shape[0] != self.action_size:
                idle = np.zeros(self.action_size, dtype=np.int8)
            return FrameAction(idle, "seed_exhausted")
        buttons = list(self.frames[self.index][: self.action_size])
        if len(buttons) < self.action_size:
            buttons.extend([0] * (self.action_size - len(buttons)))
        self.index += 1
        reason = f"replay_{self.index}"
        return FrameAction(np.array(buttons, dtype=np.int8), reason)

    def report(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "index": self.index,
            "exhausted": self.exhausted,
            "remaining": self.remaining,
        }
