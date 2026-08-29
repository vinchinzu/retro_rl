"""Offline named-skill windows from Super Metroid ``snes12_rle`` TAS slices.

Scans button streams for TASVideos GameResources patterns (shoulder/arm
pump, mockball). RAM-free: windows are slice-local frame ranges, not graph
edges. Do not STATUS-claim from a window.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from retro_harness.controls import (
    SNES_A,
    SNES_DOWN,
    SNES_L,
    SNES_LEFT,
    SNES_R,
    SNES_RIGHT,
)
from super_metroid.tas.rle import expand_snes12_rle, load_snes12_rle_seed
from super_metroid.tas.slice import SLICE_DIR

_MIN_ARM_PUMP_EDGES = 4
_MAX_ARM_PUMP_GAP = 4
_MIN_MOCKBALL_DELAY = 1
_MAX_MOCKBALL_DELAY = 24
_MIN_DOWN_HOLD = 2


@dataclass(frozen=True)
class SkillWindow:
    """One named button-pattern window.

    ``start`` inclusive, ``end`` exclusive, movie-relative (slice-local).
    """

    skill: str
    start: int
    end: int
    movie_id: str = ""


def detect_skill_windows(
    frames: Sequence[Sequence[int]],
    *,
    movie_id: str = "",
) -> tuple[SkillWindow, ...]:
    """Scan SNES-12 frames for named TAS skill patterns."""
    body = [tuple(1 if int(v) else 0 for v in fr) for fr in frames]
    windows = _detect_arm_pumps(body, movie_id) + _detect_mockballs(body, movie_id)
    windows.sort(key=lambda w: (w.start, w.skill))
    return tuple(windows)


def detect_slice_skills(slice_id: str, *, slice_dir: Path | None = None) -> tuple[SkillWindow, ...]:
    """Load an exported ``snes12_rle`` slice and detect skill windows."""
    path = (slice_dir or SLICE_DIR) / f"{slice_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing slice {slice_id}: {path}")
    frames = expand_snes12_rle(load_snes12_rle_seed(path))
    return detect_skill_windows(frames, movie_id=slice_id)


def _held(frame: Sequence[int], idx: int) -> bool:
    return bool(frame[idx]) if idx < len(frame) else False


def _run_dir(frame: Sequence[int]) -> str | None:
    left = _held(frame, SNES_LEFT)
    right = _held(frame, SNES_RIGHT)
    if left and not right:
        return "LEFT"
    if right and not left:
        return "RIGHT"
    return None


def _shoulder_rising(cur: Sequence[int], prev: Sequence[int] | None) -> bool:
    l_now, r_now = _held(cur, SNES_L), _held(cur, SNES_R)
    if prev is None:
        return l_now or r_now
    l_was, r_was = _held(prev, SNES_L), _held(prev, SNES_R)
    return (l_now and not l_was) or (r_now and not r_was)


def _detect_arm_pumps(
    frames: Sequence[Sequence[int]],
    movie_id: str,
) -> list[SkillWindow]:
    """Shoulder pumping: L/R rising edges while running (GameResources)."""
    edges: list[int] = []
    for i, fr in enumerate(frames):
        if _run_dir(fr) is None:
            continue
        prev = frames[i - 1] if i else None
        if _shoulder_rising(fr, prev):
            edges.append(i)

    windows: list[SkillWindow] = []
    cluster: list[int] = []
    for edge in edges:
        if cluster and edge - cluster[-1] > _MAX_ARM_PUMP_GAP:
            _emit_arm_pump(windows, cluster, frames, movie_id)
            cluster = []
        cluster.append(edge)
    _emit_arm_pump(windows, cluster, frames, movie_id)
    return windows


def _emit_arm_pump(
    windows: list[SkillWindow],
    cluster: list[int],
    frames: Sequence[Sequence[int]],
    movie_id: str,
) -> None:
    if len(cluster) < _MIN_ARM_PUMP_EDGES:
        return
    start = cluster[0]
    last = cluster[-1]
    period = cluster[-1] - cluster[-2] if len(cluster) > 1 else 2
    end = last + 1
    side = _run_dir(frames[last])
    while (
        end < len(frames)
        and end - last < period
        and _run_dir(frames[end]) == side
    ):
        end += 1
    windows.append(
        SkillWindow(skill="arm_pump", start=start, end=end, movie_id=movie_id)
    )


def _rising(cur: Sequence[int], prev: Sequence[int] | None, idx: int) -> bool:
    now = _held(cur, idx)
    was = _held(prev, idx) if prev is not None else False
    return now and not was


def _down_hold_end(frames: Sequence[Sequence[int]], start: int) -> int:
    end = start
    while end < len(frames) and _held(frames[end], SNES_DOWN):
        end += 1
    return end


def _detect_mockballs(
    frames: Sequence[Sequence[int]],
    movie_id: str,
) -> list[SkillWindow]:
    """Jump while running, then DOWN to morph — mockball button signature."""
    windows: list[SkillWindow] = []
    i = 0
    n = len(frames)
    while i < n:
        prev = frames[i - 1] if i else None
        fr = frames[i]
        if _rising(fr, prev, SNES_A) and _run_dir(fr) is not None:
            lo = i + _MIN_MOCKBALL_DELAY
            hi = min(n, i + _MAX_MOCKBALL_DELAY + 1)
            found: SkillWindow | None = None
            for j in range(lo, hi):
                if not _rising(frames[j], frames[j - 1], SNES_DOWN):
                    continue
                end = _down_hold_end(frames, j)
                if end - j >= _MIN_DOWN_HOLD:
                    found = SkillWindow(
                        skill="mockball",
                        start=i,
                        end=end,
                        movie_id=movie_id,
                    )
                break
            if found is not None:
                windows.append(found)
                i = found.end
                continue
        i += 1
    return windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List named skill windows in a snes12_rle TAS slice (no emulator).",
    )
    parser.add_argument("--slice", dest="slice_id", required=True, help="Catalog slice id")
    args = parser.parse_args(argv)
    try:
        windows = detect_slice_skills(args.slice_id)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not windows:
        print(f"{args.slice_id}: no skill windows")
        return 0
    for w in windows:
        print(
            f"{w.skill:12s} {w.start:5d}:{w.end:<5d} "
            f"({w.end - w.start}f)  {w.movie_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
