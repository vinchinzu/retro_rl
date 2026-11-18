"""Button tapes: every committed ``env.step`` of a trial, replayable.

A lookahead run restores the core thousands of times, but only to the live
frame or its own future. The tape holds just the committed frames, so
replaying it from the same entry with no policy and no lookahead must land
on the same RAM. ``crcs`` are RAM CRC32 checkpoints taken *before* the
action of that frame; the replay fails on the first mismatch.

The published Clean video is the replay of a tape, never the live search
run (same rule as ``nes/zelda_i``: lookahead allowed, the recording is the
tape played back with none).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tmnt_iv.paths import INTEGRATION_DIR

TAPE_SUFFIX = ".tape.npz"
CHECK_EVERY = 600


def rom_sha1() -> str:
    rom = INTEGRATION_DIR / "rom.sfc"
    return hashlib.sha1(rom.read_bytes()).hexdigest() if rom.exists() else ""


@dataclass
class Tape:
    """Committed button vectors from one entry (``NONE`` = power-on)."""

    entry: str
    turtle: str
    actions: list[tuple[int, ...]] = field(default_factory=list)
    crcs: dict[int, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)

    def action(self, frame: int) -> list[int]:
        return list(self.actions[frame])

    def save(self, path: Path) -> Path:
        path = Path(path)
        if not path.name.endswith(TAPE_SUFFIX):
            path = path.with_name(path.name + TAPE_SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = np.array(sorted(self.crcs), dtype=np.int64)
        meta = {
            "entry": self.entry,
            "turtle": self.turtle,
            "rom_sha1": rom_sha1(),
            "frames": len(self.actions),
            **self.meta,
        }
        np.savez_compressed(
            path,
            actions=np.asarray(self.actions, dtype=np.uint8).reshape(-1, 12),
            crc_frames=frames,
            crc_values=np.array([self.crcs[int(f)] for f in frames], dtype=np.uint32),
            meta=np.array(json.dumps(meta, default=str)),
        )
        return path


def load_tape(path: Path) -> Tape:
    path = Path(path)
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        sha = meta.get("rom_sha1")
        if sha and rom_sha1() and sha != rom_sha1():
            raise ValueError(f"tape ROM {sha} != installed ROM {rom_sha1()}")
        meta["source_path"] = str(path)
        meta["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return Tape(
            entry=str(meta.get("entry", "NONE")),
            turtle=str(meta.get("turtle", "mike")),
            actions=[tuple(int(v) for v in row) for row in data["actions"]],
            crcs={int(f): int(c) for f, c in zip(data["crc_frames"], data["crc_values"])},
            meta=meta,
        )


def tape_path_for(report: Path) -> Path:
    """``foo.json`` -> ``foo.tape.npz`` beside it."""
    report = Path(report)
    return report.with_name(report.stem + TAPE_SUFFIX)
