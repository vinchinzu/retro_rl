"""Deterministic frame-saving helpers for raw-button TAS seeds.

These tools are game-agnostic: they operate on lists of button arrays and an
optional evaluate callable (or :class:`retro_harness.platformer.evaluator.Evaluator`).

Typical pipeline for a completed seed::

    1. analyze_seed / find_stalls     — locate waste
    2. trim_leading_idle              — drop phase-safe startup idle
    3. trim_after_completion          — drop post-clear pad
    4. cleanup_auto_inputs            — zero buttons during forced states
    5. segment_hillclimb / hillclimb_raw — local search in a window
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EvalFn = Callable[[list[list[int]]], Any]


def is_idle_frame(buttons: Sequence[int]) -> bool:
    """True when no button is held."""
    return not any(int(b) for b in buttons)


def count_leading_idle(frames: Sequence[Sequence[int]]) -> int:
    """Count consecutive idle frames from the start of *frames*."""
    n = 0
    for frame in frames:
        if not is_idle_frame(frame):
            break
        n += 1
    return n


def count_trailing_idle(frames: Sequence[Sequence[int]]) -> int:
    """Count consecutive idle frames at the end of *frames*."""
    n = 0
    for frame in reversed(frames):
        if not is_idle_frame(frame):
            break
        n += 1
    return n


def clone_frames(frames: Sequence[Sequence[int]]) -> list[list[int]]:
    """Deep-copy button frames to mutable lists."""
    return [list(map(int, f)) for f in frames]


def pad_buttons(frame: Sequence[int], size: int = 12) -> list[int]:
    """Pad/truncate a button vector to *size* (default SNES 12)."""
    buttons = [int(b) for b in frame[:size]]
    if len(buttons) < size:
        buttons.extend([0] * (size - len(buttons)))
    return buttons


def _result_completed(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, dict):
        return bool(result.get("completed") or result.get("success"))
    return bool(getattr(result, "completed", False) or getattr(result, "success", False))


def _result_frames(result: Any, fallback: int) -> int:
    if result is None:
        return fallback
    if isinstance(result, dict):
        for key in ("total_frames", "frames", "clear_frames"):
            if key in result and result[key] is not None:
                return int(result[key])
        return fallback
    for attr in ("total_frames", "frames", "clear_frames"):
        val = getattr(result, attr, None)
        if val is not None:
            return int(val)
    return fallback


def _result_fitness(result: Any) -> float:
    if result is None:
        return float("-inf")
    if isinstance(result, dict):
        return float(result.get("fitness", 0.0))
    return float(getattr(result, "fitness", 0.0))


@dataclass
class TrimResult:
    """Outcome of a leading-idle or post-completion trim search."""

    frames: list[list[int]]
    trim: int
    clear_frames: int | None
    completed: bool
    original_frames: int
    notes: str = ""
    tried: list[dict[str, Any]] = field(default_factory=list)

    @property
    def frames_saved(self) -> int:
        if self.clear_frames is None:
            return 0
        # Prefer clear-frame delta when known; else seed length delta.
        return max(0, self.original_frames - len(self.frames))


def trim_leading_idle(
    frames: Sequence[Sequence[int]],
    evaluate: EvalFn,
    *,
    max_trim: int | None = None,
    step: int = 1,
    parity: str = "any",
    require_completion: bool = True,
    verbose: bool = False,
) -> TrimResult:
    """Search leading-idle trims that still complete (or keep max fitness).

    Parameters
    ----------
    parity:
        ``\"any\"`` try every candidate trim;
        ``\"even\"`` only even trims (SMB 1-1 phase trap: odd trims die);
        ``\"odd\"`` only odd trims.
    step:
        Stride between candidate trim values (after parity filter).
    """
    base = clone_frames(frames)
    lead = count_leading_idle(base)
    limit = lead if max_trim is None else min(lead, max_trim)
    original_len = len(base)

    def allowed(t: int) -> bool:
        if parity == "even":
            return t % 2 == 0
        if parity == "odd":
            return t % 2 == 1
        return True

    candidates = [t for t in range(0, limit + 1, max(1, step)) if allowed(t)]
    if 0 not in candidates:
        candidates.insert(0, 0)

    best_trim = 0
    best_frames = base
    best_clear: int | None = None
    best_completed = False
    best_score = float("-inf")  # higher is better: completed first, then fewer clear frames
    tried: list[dict[str, Any]] = []

    for trim in candidates:
        candidate = base[trim:]
        result = evaluate(candidate)
        completed = _result_completed(result)
        clear = _result_frames(result, len(candidate)) if completed else None
        fitness = _result_fitness(result)
        tried.append(
            {
                "trim": trim,
                "completed": completed,
                "clear_frames": clear,
                "fitness": fitness,
                "seed_len": len(candidate),
            }
        )
        if verbose:
            print(
                f"[TRIM] lead={trim:3d} completed={completed} "
                f"clear={clear} fitness={fitness:.1f}"
            )

        if require_completion and not completed:
            continue

        # Prefer completion, then fewer clear frames, then higher fitness, then larger trim.
        if completed:
            score = (
                1_000_000_000
                - (clear if clear is not None else len(candidate)) * 1000
                + fitness
                + trim
            )
        else:
            score = fitness

        if score > best_score:
            best_score = score
            best_trim = trim
            best_frames = candidate
            best_clear = clear
            best_completed = completed

    return TrimResult(
        frames=best_frames,
        trim=best_trim,
        clear_frames=best_clear,
        completed=best_completed,
        original_frames=original_len,
        notes=f"leading idle trim={best_trim} (lead_available={lead}, parity={parity})",
        tried=tried,
    )


def trim_after_completion(
    frames: Sequence[Sequence[int]],
    evaluate: EvalFn,
    *,
    pad: int = 30,
    verbose: bool = False,
) -> TrimResult:
    """Keep frames through first completion + *pad* idle frames, drop the rest."""
    base = clone_frames(frames)
    result = evaluate(base)
    if not _result_completed(result):
        return TrimResult(
            frames=base,
            trim=0,
            clear_frames=None,
            completed=False,
            original_frames=len(base),
            notes="seed does not complete; left unchanged",
        )
    clear = _result_frames(result, len(base))
    keep = min(len(base), clear + max(0, pad))
    trimmed = base[:keep]
    # Force trailing pad to idle so leftover junk inputs cannot desync.
    idle = [0] * max(len(base[0]) if base else 12, 9)
    for i in range(clear, keep):
        trimmed[i] = list(idle[: len(trimmed[i])])
    if verbose:
        print(f"[TRIM] after_completion clear={clear} keep={keep} pad={pad}")
    # Re-verify
    result2 = evaluate(trimmed)
    return TrimResult(
        frames=trimmed,
        trim=len(base) - keep,
        clear_frames=_result_frames(result2, clear) if _result_completed(result2) else clear,
        completed=_result_completed(result2),
        original_frames=len(base),
        notes=f"post-clear pad={pad}; dropped {len(base) - keep} trailing frames",
    )


@dataclass
class StallRegion:
    """A contiguous stretch with no progress (or zero speed)."""

    start: int
    length: int
    x: float | int = 0
    y: float | int = 0
    reason: str = "stall"

    @property
    def end(self) -> int:
        return self.start + self.length


def find_stalls(
    progress: Sequence[float],
    *,
    min_length: int = 20,
    positions: Sequence[tuple[float, float]] | None = None,
) -> list[StallRegion]:
    """Find runs where progress does not increase for *min_length* frames.

    *progress[i]* is the max-progress observed through frame *i*.
    """
    stalls: list[StallRegion] = []
    if not progress:
        return stalls
    run = 0
    start = 0
    for i in range(1, len(progress)):
        if progress[i] <= progress[i - 1]:
            if run == 0:
                start = i
            run += 1
        else:
            if run >= min_length:
                x = y = 0.0
                if positions is not None and start < len(positions):
                    x, y = positions[start]
                stalls.append(StallRegion(start=start, length=run, x=x, y=y))
            run = 0
    if run >= min_length:
        x = y = 0.0
        if positions is not None and start < len(positions):
            x, y = positions[start]
        stalls.append(StallRegion(start=start, length=run, x=x, y=y))
    return stalls


def find_button_hold_stalls(
    frames: Sequence[Sequence[int]],
    *,
    min_length: int = 20,
) -> list[StallRegion]:
    """Find long runs of identical button vectors (held inputs / waits)."""
    stalls: list[StallRegion] = []
    if not frames:
        return stalls
    start = 0
    for i in range(1, len(frames)):
        if list(frames[i]) != list(frames[start]):
            length = i - start
            if length >= min_length:
                stalls.append(
                    StallRegion(start=start, length=length, reason="hold")
                )
            start = i
    length = len(frames) - start
    if length >= min_length:
        stalls.append(StallRegion(start=start, length=length, reason="hold"))
    return stalls


def cleanup_auto_inputs(
    frames: Sequence[Sequence[int]],
    player_states: Sequence[int],
    *,
    auto_states: Sequence[int] = (3, 4, 5),
) -> tuple[list[list[int]], int]:
    """Zero buttons on frames whose *player_states[i]* is automated.

    Returns ``(cleaned_frames, zeroed_count)``. Lengths must match.
    SMB uses player_state 3/4/5 for area-enter / flagpole / castle walk;
    other games can pass their own *auto_states*.
    """
    if len(frames) != len(player_states):
        raise ValueError(
            f"frames ({len(frames)}) and player_states ({len(player_states)}) length mismatch"
        )
    auto = set(int(s) for s in auto_states)
    cleaned = clone_frames(frames)
    zeroed = 0
    for i, state in enumerate(player_states):
        if int(state) in auto and any(cleaned[i]):
            cleaned[i] = [0] * len(cleaned[i])
            zeroed += 1
    return cleaned, zeroed


def compress_hold_window(
    frames: Sequence[Sequence[int]],
    start: int,
    end: int,
    new_length: int,
) -> list[list[int]]:
    """Shrink frames[start:end] to *new_length* copies of the first hold frame.

    If the window is not a pure hold, keeps the first *new_length* frames of the
    window (prefix truncate). Used by searchers that probe hold shortening.
    """
    base = clone_frames(frames)
    if start < 0 or end > len(base) or start >= end:
        raise ValueError(f"bad window [{start}:{end}] for len={len(base)}")
    if new_length < 0:
        raise ValueError("new_length must be >= 0")
    window = base[start:end]
    if new_length == 0:
        replacement: list[list[int]] = []
    elif all(w == window[0] for w in window):
        replacement = [list(window[0]) for _ in range(new_length)]
    else:
        replacement = [list(f) for f in window[:new_length]]
    return base[:start] + replacement + base[end:]


def search_hold_compressions(
    frames: Sequence[Sequence[int]],
    evaluate: EvalFn,
    *,
    min_hold: int = 30,
    min_keep: int = 1,
    max_trials_per_hold: int = 8,
    require_completion: bool = True,
    verbose: bool = False,
) -> TrimResult:
    """Try shortening long identical-button holds while preserving completion.

    Greedy left-to-right: for each hold, binary-search the shortest length that
    still completes (or improves fitness).
    """
    current = clone_frames(frames)
    original_len = len(current)
    base_result = evaluate(current)
    if require_completion and not _result_completed(base_result):
        return TrimResult(
            frames=current,
            trim=0,
            clear_frames=None,
            completed=False,
            original_frames=original_len,
            notes="seed does not complete; hold search skipped",
        )

    best_clear = _result_frames(base_result, len(current))
    tried: list[dict[str, Any]] = []
    # Snapshot hold starts on the original sequence; re-scan after each success.
    pending_starts = [
        h.start for h in find_button_hold_stalls(current, min_length=min_hold)
    ]
    holds_scanned = 0

    for _origin in pending_starts:
        holds_now = find_button_hold_stalls(current, min_length=min_hold)
        # Pick the longest remaining hold (greedy waste removal).
        if not holds_now:
            break
        match = max(holds_now, key=lambda h: h.length)
        holds_scanned += 1

        lo, hi = min_keep, match.length
        best_keep = match.length
        best_local = current
        trials_left = max_trials_per_hold
        # Binary search shortest keep that still completes with clear <= best_clear
        while lo <= hi and trials_left > 0:
            mid = (lo + hi) // 2
            candidate = compress_hold_window(current, match.start, match.end, mid)
            result = evaluate(candidate)
            completed = _result_completed(result)
            clear = _result_frames(result, len(candidate)) if completed else None
            tried.append(
                {
                    "hold_start": match.start,
                    "hold_len": match.length,
                    "keep": mid,
                    "completed": completed,
                    "clear_frames": clear,
                }
            )
            if verbose:
                print(
                    f"[HOLD] @{match.start} {match.length}->{mid} "
                    f"completed={completed} clear={clear}"
                )
            if completed and clear is not None and clear <= best_clear:
                best_keep = mid
                best_clear = clear
                best_local = candidate
                hi = mid - 1
            else:
                lo = mid + 1
            trials_left -= 1

        if best_keep < match.length:
            current = best_local
            if verbose:
                print(f"[HOLD] compressed @{match.start} {match.length}->{best_keep}")

    final = evaluate(current)
    return TrimResult(
        frames=current,
        trim=original_len - len(current),
        clear_frames=_result_frames(final, len(current))
        if _result_completed(final)
        else None,
        completed=_result_completed(final),
        original_frames=original_len,
        notes=(
            f"hold compression trials={len(tried)} "
            f"holds_scanned={holds_scanned}"
        ),
        tried=tried,
    )


def save_raw_seed(
    path: Path | str,
    frames: Sequence[Sequence[int]],
    *,
    metadata: dict[str, Any] | None = None,
    button_size: int = 12,
) -> Path:
    """Write a ``raw_buttons`` JSON seed compatible with retro_harness.platformer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = [pad_buttons(f, button_size) for f in frames]
    data: dict[str, Any] = {
        "raw_buttons": raw,
        "num_frames": len(raw),
    }
    if metadata:
        data.update(metadata)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def load_raw_frames(path: Path | str) -> list[list[int]]:
    """Load raw button frames from a recording / hillclimb JSON."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "raw_buttons" in data:
        return clone_frames(data["raw_buttons"])
    if "segments" in data and data.get("format") == "nes9_rle":
        frames: list[list[int]] = []
        for seg in data["segments"]:
            buttons = pad_buttons(seg["b"], 12)
            frames.extend([list(buttons) for _ in range(int(seg["n"]))])
        return frames
    if "actions" in data:
        raise ValueError(
            f"{path} has action indices only; convert with action_index_to_buttons first"
        )
    raise ValueError(f"no raw_buttons/segments in {path}")


def analyze_seed_static(frames: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Cheap analysis that needs no emulator."""
    holds = find_button_hold_stalls(frames, min_length=20)
    return {
        "num_frames": len(frames),
        "leading_idle": count_leading_idle(frames),
        "trailing_idle": count_trailing_idle(frames),
        "hold_stalls": [asdict(h) for h in holds[:50]],
        "hold_stall_count": len(holds),
        "longest_hold": max((h.length for h in holds), default=0),
        "total_hold_waste_estimate": sum(max(0, h.length - 5) for h in holds),
    }
