"""RLE-native action sequence representation and mutation operators.

Frame-expanded seeds (~20k buttons) make GA/hillclimb sample-inefficient:
point mutations destroy long holds and ignore the structure that already
works.  This module keeps individuals as run-length lists::

    [(button_tuple | action_index, duration), ...]

and mutates *runs* (duration, mask, split, merge, insert atomic patterns).

Works for both:

- **action indices** (``list[int]`` seeds used by ``genetic.py``)
- **raw button frames** (``list[list[int]]`` / nes9_rle)
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

# Button / action run: (payload, duration). Payload is hashable for merge.
RleRun = tuple[T, int]
RleSeq = list[RleRun[T]]


def _as_key(payload: T) -> T:
    """Normalize list payloads to tuples for equality / hashing."""
    if isinstance(payload, list):
        return tuple(int(x) for x in payload)  # type: ignore[return-value]
    return payload


def _as_payload(key: T, *, as_list: bool) -> T:
    if as_list and isinstance(key, tuple):
        return list(key)  # type: ignore[return-value]
    return key


def compress_rle(frames: Sequence[T]) -> RleSeq[T]:
    """Compress a frame-expanded sequence into RLE runs."""
    if not frames:
        return []
    runs: RleSeq[T] = []
    cur = _as_key(frames[0])
    n = 1
    for frame in frames[1:]:
        key = _as_key(frame)
        if key == cur:
            n += 1
        else:
            runs.append((cur, n))  # type: ignore[arg-type]
            cur = key
            n = 1
    runs.append((cur, n))  # type: ignore[arg-type]
    return runs


def expand_rle(runs: Sequence[RleRun[T]], *, as_list: bool = False) -> list[T]:
    """Expand RLE runs back to a frame list."""
    out: list[T] = []
    for payload, duration in runs:
        if duration <= 0:
            continue
        item = _as_payload(payload, as_list=as_list)
        n = int(duration)
        # Per-frame copies so mutating one frame never aliases the whole run.
        if as_list:
            out.extend(list(item) for _ in range(n))  # type: ignore[list-item]
        else:
            out.extend([item] * n)  # type: ignore[list-item]
    return out


def rle_total_frames(runs: Sequence[RleRun[T]]) -> int:
    return sum(max(0, int(d)) for _, d in runs)


def rle_normalize(runs: Sequence[RleRun[T]]) -> RleSeq[T]:
    """Drop zero-length runs and merge adjacent equal payloads."""
    out: RleSeq[T] = []
    for payload, duration in runs:
        d = int(duration)
        if d <= 0:
            continue
        key = _as_key(payload)
        if out and out[-1][0] == key:
            out[-1] = (key, out[-1][1] + d)  # type: ignore[assignment]
        else:
            out.append((key, d))  # type: ignore[arg-type]
    return out


def rle_frame_to_run_index(runs: Sequence[RleRun[T]], frame: int) -> tuple[int, int]:
    """Map absolute frame index → (run_index, offset_within_run)."""
    if frame < 0:
        return 0, 0
    acc = 0
    for i, (_, d) in enumerate(runs):
        if frame < acc + d:
            return i, frame - acc
        acc += d
    if not runs:
        return 0, 0
    return len(runs) - 1, max(0, runs[-1][1] - 1)


def rle_slice_frames(runs: Sequence[RleRun[T]], start: int, end: int) -> RleSeq[T]:
    """Extract runs covering absolute frames [start, end)."""
    if end <= start or not runs:
        return []
    total = rle_total_frames(runs)
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    out: RleSeq[T] = []
    acc = 0
    for payload, d in runs:
        run_end = acc + d
        if run_end <= start:
            acc = run_end
            continue
        if acc >= end:
            break
        lo = max(start, acc)
        hi = min(end, run_end)
        out.append((_as_key(payload), hi - lo))  # type: ignore[arg-type]
        acc = run_end
    return rle_normalize(out)


def rle_replace_window(
    runs: Sequence[RleRun[T]],
    start: int,
    end: int,
    replacement: Sequence[RleRun[T]],
) -> RleSeq[T]:
    """Replace absolute frame window [start, end) with *replacement* runs."""
    total = rle_total_frames(runs)
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    head = rle_slice_frames(runs, 0, start)
    tail = rle_slice_frames(runs, end, total)
    return rle_normalize([*head, *replacement, *tail])


# -- Mutation operators -------------------------------------------------------


@dataclass
class RleMutateConfig:
    """Weights / bounds for RLE mutations."""

    max_duration_delta: int = 8
    min_run_duration: int = 1
    max_insert_duration: int = 12
    # Atomic patterns as payload lists (action indices or button tuples).
    # Empty → only clone neighboring payloads on insert.
    atomic_patterns: tuple[tuple[object, ...], ...] = ()
    num_actions: int = 14  # for index payloads
    button_size: int = 12  # for raw button payloads
    button_flip_indices: tuple[int, ...] = (0, 5, 6, 7, 8)  # B, DN, L, R, A-ish


def mutate_duration(
    runs: RleSeq[T],
    *,
    max_delta: int = 8,
    min_duration: int = 1,
) -> RleSeq[T]:
    """Change duration of one random run by ±1…max_delta."""
    if not runs:
        return []
    out = list(runs)
    i = random.randrange(len(out))
    payload, d = out[i]
    delta = random.randint(1, max_delta) * random.choice((-1, 1))
    new_d = max(min_duration, d + delta)
    out[i] = (payload, new_d)
    return rle_normalize(out)


def mutate_payload(
    runs: RleSeq[T],
    *,
    num_actions: int = 14,
    button_size: int = 12,
    button_flip_indices: Sequence[int] = (0, 5, 6, 7, 8),
) -> RleSeq[T]:
    """Change the payload of one run (action index or button mask)."""
    if not runs:
        return []
    out = list(runs)
    i = random.randrange(len(out))
    payload, d = out[i]
    if isinstance(payload, int):
        new_p: object = random.randint(0, max(0, num_actions - 1))
    else:
        # button vector (tuple)
        buttons = list(payload) if isinstance(payload, (list, tuple)) else [0] * button_size
        while len(buttons) < button_size:
            buttons.append(0)
        flip = random.choice(list(button_flip_indices))
        if flip < len(buttons):
            buttons[flip] = 1 - int(buttons[flip])
        new_p = tuple(int(b) for b in buttons[:button_size])
    out[i] = (new_p, d)  # type: ignore[assignment]
    return rle_normalize(out)


def mutate_split(runs: RleSeq[T]) -> RleSeq[T]:
    """Split one run with duration ≥ 2 into two adjacent runs (same payload).

    Does **not** re-merge equal neighbors so a follow-up payload/duration
    mutation can edit each half independently.
    """
    candidates = [i for i, (_, d) in enumerate(runs) if d >= 2]
    if not candidates:
        return list(runs)
    i = random.choice(candidates)
    payload, d = runs[i]
    left = random.randint(1, d - 1)
    out = list(runs)
    out[i : i + 1] = [(payload, left), (payload, d - left)]
    return out


def mutate_merge(runs: RleSeq[T]) -> RleSeq[T]:
    """Merge two adjacent runs (prefer equal payloads; else take first payload)."""
    if len(runs) < 2:
        return list(runs)
    i = random.randrange(len(runs) - 1)
    p0, d0 = runs[i]
    p1, d1 = runs[i + 1]
    payload = p0 if p0 == p1 or random.random() < 0.7 else p1
    out = list(runs)
    out[i : i + 2] = [(payload, d0 + d1)]
    return rle_normalize(out)


def mutate_insert(
    runs: RleSeq[T],
    *,
    max_duration: int = 12,
    atomic_patterns: Sequence[Sequence[object]] = (),
    num_actions: int = 14,
) -> RleSeq[T]:
    """Insert a short run (atomic pattern or random / neighbor payload)."""
    out = list(runs)
    pos = random.randint(0, len(out))
    duration = random.randint(1, max_duration)
    if atomic_patterns and random.random() < 0.5:
        pattern = random.choice(list(atomic_patterns))
        # pattern is a sequence of payloads each held 1f, or single payload
        chunk: RleSeq[T] = []
        for p in pattern:
            key = _as_key(p)  # type: ignore[arg-type]
            if chunk and chunk[-1][0] == key:
                chunk[-1] = (key, chunk[-1][1] + 1)  # type: ignore[assignment]
            else:
                chunk.append((key, 1))  # type: ignore[arg-type]
        # scale to ~duration by repeating last
        while rle_total_frames(chunk) < duration and chunk:
            last_p, last_d = chunk[-1]
            chunk[-1] = (last_p, last_d + 1)
        out[pos:pos] = chunk
    else:
        if out:
            payload = out[random.randrange(len(out))][0]
            if isinstance(payload, int) and random.random() < 0.3:
                payload = random.randint(0, max(0, num_actions - 1))  # type: ignore[assignment]
        else:
            payload = 0  # type: ignore[assignment]
        out.insert(pos, (payload, duration))
    return rle_normalize(out)


def mutate_delete_run(runs: RleSeq[T]) -> RleSeq[T]:
    """Delete one short run (duration ≤ 8 preferred)."""
    if len(runs) < 2:
        return list(runs)
    short = [i for i, (_, d) in enumerate(runs) if d <= 8]
    i = random.choice(short) if short and random.random() < 0.7 else random.randrange(len(runs))
    out = list(runs)
    del out[i]
    return rle_normalize(out)


def mutate_swap_runs(runs: RleSeq[T]) -> RleSeq[T]:
    """Swap two nearby runs."""
    if len(runs) < 2:
        return list(runs)
    i = random.randrange(len(runs) - 1)
    j = min(len(runs) - 1, i + random.randint(1, min(5, len(runs) - 1 - i)))
    out = list(runs)
    out[i], out[j] = out[j], out[i]
    return rle_normalize(out)


def mutate_rle(
    runs: Sequence[RleRun[T]],
    *,
    config: RleMutateConfig | None = None,
    n_ops: int = 1,
) -> RleSeq[T]:
    """Apply one or more weighted RLE mutations."""
    cfg = config or RleMutateConfig()
    out: RleSeq[T] = rle_normalize(runs)
    ops = [
        ("duration", 30),
        ("payload", 15),
        ("split", 10),
        ("merge", 10),
        ("insert", 15),
        ("delete", 12),
        ("swap", 8),
    ]
    names = [n for n, _ in ops]
    weights = [w for _, w in ops]
    for _ in range(max(1, n_ops)):
        if not out:
            out = mutate_insert(
                out,
                max_duration=cfg.max_insert_duration,
                atomic_patterns=cfg.atomic_patterns,  # type: ignore[arg-type]
                num_actions=cfg.num_actions,
            )
            continue
        op = random.choices(names, weights=weights, k=1)[0]
        if op == "duration":
            out = mutate_duration(
                out, max_delta=cfg.max_duration_delta, min_duration=cfg.min_run_duration
            )
        elif op == "payload":
            out = mutate_payload(
                out,
                num_actions=cfg.num_actions,
                button_size=cfg.button_size,
                button_flip_indices=cfg.button_flip_indices,
            )
        elif op == "split":
            out = mutate_split(out)
        elif op == "merge":
            out = mutate_merge(out)
        elif op == "insert":
            out = mutate_insert(
                out,
                max_duration=cfg.max_insert_duration,
                atomic_patterns=cfg.atomic_patterns,  # type: ignore[arg-type]
                num_actions=cfg.num_actions,
            )
        elif op == "delete":
            out = mutate_delete_run(out)
        elif op == "swap":
            out = mutate_swap_runs(out)
    return out


def crossover_rle_single(
    parent1: Sequence[RleRun[T]],
    parent2: Sequence[RleRun[T]],
) -> RleSeq[T]:
    """Single-point crossover on run index (not frame index)."""
    if not parent1:
        return rle_normalize(parent2)
    if not parent2:
        return rle_normalize(parent1)
    cut1 = random.randint(0, len(parent1))
    # Proportional cut on parent2 by frame fraction
    frac = cut1 / max(1, len(parent1))
    cut2 = int(round(frac * len(parent2)))
    cut2 = max(0, min(cut2, len(parent2)))
    return rle_normalize([*parent1[:cut1], *parent2[cut2:]])


def crossover_rle_frame_aligned(
    parent1: Sequence[RleRun[T]],
    parent2: Sequence[RleRun[T]],
) -> RleSeq[T]:
    """Crossover at a random absolute frame (aligned on progress proxies)."""
    t1 = rle_total_frames(parent1)
    t2 = rle_total_frames(parent2)
    if t1 < 2 or t2 < 2:
        return crossover_rle_single(parent1, parent2)
    # Pick a fraction along the shorter timeline
    frac = random.uniform(0.15, 0.85)
    f1 = int(frac * t1)
    f2 = int(frac * t2)
    head = rle_slice_frames(parent1, 0, f1)
    tail = rle_slice_frames(parent2, f2, t2)
    return rle_normalize([*head, *tail])


# SMB-oriented atomic patterns (NES-9 button tuples padded conceptually).
# Indices: B=0, SELECT=2, START=3, UP=4, DOWN=5, LEFT=6, RIGHT=7, A=8

def _nes9(*pressed: int) -> tuple[int, ...]:
    b = [0] * 9
    for i in pressed:
        if 0 <= i < 9:
            b[i] = 1
    return tuple(b)


SMB_ATOMIC_PATTERNS: tuple[tuple[tuple[int, ...], ...], ...] = (
    # short hop (A pulse) while running right
    (_nes9(7, 0, 8), _nes9(7, 0), _nes9(7, 0)),
    # run right
    (_nes9(7, 0),),
    # brake / left briefly
    (_nes9(6), _nes9(6), _nes9()),
    # pipe down
    (_nes9(5), _nes9(5), _nes9(5), _nes9(5)),
    # walk right jump
    (_nes9(7, 8), _nes9(7), _nes9(7)),
    # idle tick
    (_nes9(),),
)

# Action-index atoms for SMB_ACTIONS table (see retro_harness.platformer.levels.smb)
SMB_ACTION_ATOMIC_PATTERNS: tuple[tuple[int, ...], ...] = (
    (3, 2, 2),  # run+jump then run
    (2,),  # run
    (10, 10, 10, 10),  # down pipe
    (4, 1, 1),  # walk jump
    (0,),  # idle
    (6, 6),  # left
)
