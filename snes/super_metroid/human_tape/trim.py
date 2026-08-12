"""Offline idle + retry trim for guided_human room hops.

Pure offline: frames (SNES-12) + optional per-frame trace. No emulator.
Trimmed seeds stay hop-replay-validatable later by keeping absolute
``kept_ranges`` into the source task.

Modes
-----
* ``traversal`` (path rooms): leading idle, trailing after leave, mid-idle
  (heuristic), retry-loop high-water-mark cuts.
* ``combat`` (bosses / metroids): leading idle + optional trailing only.
  No mid-idle, no retry HWM cuts.
* ``safe``: leading + trailing only (contiguous keep). Prefer this for
  **open-loop hop-replay** seeds — mid/retry cuts skip frames that still
  tick enemy RNG and are **not** open-loop-safe unless dual-green validated.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.anchors import parse_room_id

# SNES-12 button order (gym-retro / libretro convention)
BTN_B, BTN_Y, BTN_SELECT, BTN_START = 0, 1, 2, 3
BTN_UP, BTN_DOWN, BTN_LEFT, BTN_RIGHT = 4, 5, 6, 7
BTN_A, BTN_X, BTN_L, BTN_R = 8, 9, 10, 11

# Boss / metroid rooms: combat trim mode (no mid-idle / retry cuts).
COMBAT_ROOM_IDS: frozenset[int] = frozenset(
    {
        0xDD58,  # Mother Brain
        0xB32E,  # Ridley
        0xDA60,  # Draygon
        0xCD13,  # Phantoon
        0xB283,  # Golden Torizo
        0xB62B,  # Metal Pirates
        0xDAE1,  # Metroid room 1
        0xDB31,  # Metroid room 2
        0xDB7D,  # Metroid room 3
        0xDBCD,  # Metroid room 4
        0xDCB1,  # Big Boy (combat-like thrash)
    }
)

_VEL_EPS = 1.0  # |vx|+|vy| below this ≈ stopped
_PROGRESS_FLAT_PX = 2.0


@dataclass
class TrimReport:
    mode: str
    frames_before: int
    frames_after: int
    leading_idle_cut: int
    trailing_cut: int
    mid_idle_cut: int
    retry_loops_cut: int
    retry_frames_cut: int
    kept_ranges: list[tuple[int, int]]  # absolute [lo, hi) into original frames
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kept_ranges"] = [[a, b] for a, b in self.kept_ranges]
        return d


def is_combat_room(room_id: int | None) -> bool:
    if room_id is None:
        return False
    return int(room_id) in COMBAT_ROOM_IDS


def infer_mode(room_id: int | None) -> str:
    return "combat" if is_combat_room(room_id) else "traversal"


def _frame_vec(frame: Sequence[int] | None) -> list[int]:
    if frame is None:
        return [0] * 12
    out = [int(b) for b in frame[:12]]
    if len(out) < 12:
        out.extend([0] * (12 - len(out)))
    return out


def is_idle_frame(frame: Sequence[int] | None) -> bool:
    """Idle = all-zero SNES-12 buttons."""
    return all(b == 0 for b in _frame_vec(frame))


def holds_charge_safe_buttons(frame: Sequence[int] | None) -> bool:
    """True if B / Y / X held — never mid-idle-cut (dash / charge / shoot)."""
    v = _frame_vec(frame)
    return bool(v[BTN_B] or v[BTN_Y] or v[BTN_X])


def _trace_row(
    trace: Sequence[Mapping[str, Any]] | None, index: int
) -> Mapping[str, Any] | None:
    if not trace or index < 0 or index >= len(trace):
        return None
    return trace[index]


def _xy(row: Mapping[str, Any] | None, fallback: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if row is None:
        return fallback
    return float(row.get("x", fallback[0])), float(row.get("y", fallback[1]))


def _room_id(row: Mapping[str, Any] | None) -> int | None:
    """Room id from a trace row; reuses anchors.parse_room_id for int/hex forms."""
    if row is None:
        return None
    if row.get("room") is not None:
        return parse_room_id(row["room"])
    return parse_room_id(row.get("room_hex") or row.get("room_id"))


def _vel_sum(row: Mapping[str, Any] | None) -> float | None:
    if row is None:
        return None
    if "vx" not in row and "vy" not in row:
        return None
    return abs(float(row.get("vx") or 0)) + abs(float(row.get("vy") or 0))


def progress_along_leave(
    trace: Sequence[Mapping[str, Any]],
    start_index: int,
    end_index: int,
    *,
    start_xy: Sequence[float] | None = None,
    end_xy: Sequence[float] | None = None,
) -> list[float]:
    """Scalar progress along unit(leave - enter) for each frame in [start, end]."""
    if end_index < start_index:
        return []
    if start_xy is not None:
        x0, y0 = float(start_xy[0]), float(start_xy[1])
    else:
        x0, y0 = _xy(_trace_row(trace, start_index))
    if end_xy is not None:
        x1, y1 = float(end_xy[0]), float(end_xy[1])
    else:
        x1, y1 = _xy(_trace_row(trace, end_index), (x0, y0))
    dx, dy = x1 - x0, y1 - y0
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        # Degenerate leave vector: fall back to path length from enter.
        out: list[float] = []
        for i in range(start_index, end_index + 1):
            x, y = _xy(_trace_row(trace, i), (x0, y0))
            out.append(math.hypot(x - x0, y - y0))
        return out
    ux, uy = dx / mag, dy / mag
    out = []
    for i in range(start_index, end_index + 1):
        x, y = _xy(_trace_row(trace, i), (x0, y0))
        out.append((x - x0) * ux + (y - y0) * uy)
    return out


def find_retry_loop_cuts(
    progress: Sequence[float],
    *,
    drop_px: float = 48.0,
    min_loop_frames: int = 45,
) -> list[tuple[int, int]]:
    """Local half-open [lo, hi) ranges for retry loop bodies.

    When progress falls by more than ``drop_px`` from the high-water mark and
    later recovers to ≥ that HWM, the span between first HWM touch and recovery
    is a retry loop. We cut the body ``(hwm_idx, recovery)`` — keep the first
    approach peak and the successful continuation from recovery. Never cuts a
    drop that does not recover (last successful segment to leave is kept).
    """
    n = len(progress)
    if n < min_loop_frames + 2:
        return []
    cuts: list[tuple[int, int]] = []
    hwm = float(progress[0])
    hwm_idx = 0
    i = 1
    while i < n:
        p = float(progress[i])
        if p > hwm + 1e-9:
            hwm = p
            hwm_idx = i
            i += 1
            continue
        if p < hwm - drop_px:
            # Seek recovery to ≥ previous HWM.
            j = i + 1
            while j < n and float(progress[j]) < hwm - 1e-9:
                j += 1
            if j < n and float(progress[j]) >= hwm - 1e-9:
                lo, hi = hwm_idx + 1, j  # exclusive of HWM peak and recovery
                if hi - lo >= min_loop_frames:
                    cuts.append((lo, hi))
                    # Resume after recovery; HWM stands at recovery frame.
                    hwm = float(progress[j])
                    hwm_idx = j
                    i = j + 1
                    continue
            # No usable recovery: do not cut; advance past the drop search.
            i = j if j > i else i + 1
            continue
        i += 1
    return cuts


def find_mid_idle_cuts(
    frames: Sequence[Sequence[int]],
    progress: Sequence[float],
    trace: Sequence[Mapping[str, Any]] | None,
    start_index: int,
    *,
    min_idle: int = 40,
    progress_flat_px: float = _PROGRESS_FLAT_PX,
) -> list[tuple[int, int]]:
    """Local half-open mid-idle runs safe to cut (traversal only)."""
    n = len(progress)
    if n < min_idle or len(frames) < start_index + n:
        return []

    def cuttable(local_i: int) -> bool:
        abs_i = start_index + local_i
        fr = frames[abs_i]
        if holds_charge_safe_buttons(fr):
            return False
        if not is_idle_frame(fr):
            return False
        row = _trace_row(trace, abs_i)
        vsum = _vel_sum(row)
        if vsum is not None and vsum > _VEL_EPS:
            return False
        return True

    cuts: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not cuttable(i):
            i += 1
            continue
        j = i + 1
        while j < n and cuttable(j):
            j += 1
        run_lo, run_hi = i, j  # [run_lo, run_hi)
        run_len = run_hi - run_lo
        if run_len >= min_idle:
            # Progress flat across the run (± progress_flat_px of run median/start).
            seg = [float(progress[k]) for k in range(run_lo, run_hi)]
            p0 = seg[0]
            if max(abs(p - p0) for p in seg) <= progress_flat_px:
                cuts.append((run_lo, run_hi))
            else:
                # Try longest flat sub-runs inside this idle span.
                sub_a = run_lo
                while sub_a < run_hi:
                    base = float(progress[sub_a])
                    sub_b = sub_a + 1
                    while (
                        sub_b < run_hi
                        and abs(float(progress[sub_b]) - base) <= progress_flat_px
                    ):
                        sub_b += 1
                    if sub_b - sub_a >= min_idle:
                        cuts.append((sub_a, sub_b))
                    sub_a = sub_b if sub_b > sub_a else sub_a + 1
        i = j
    return cuts


def _mask_to_ranges(keep: Sequence[bool], absolute_base: int) -> list[tuple[int, int]]:
    """Convert keep mask to absolute half-open [lo, hi) ranges."""
    ranges: list[tuple[int, int]] = []
    n = len(keep)
    i = 0
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i + 1
        while j < n and keep[j]:
            j += 1
        ranges.append((absolute_base + i, absolute_base + j))
        i = j
    return ranges


def _apply_local_cuts(keep: list[bool], cuts: Sequence[tuple[int, int]]) -> int:
    """Mark local [lo, hi) as removed. Returns frames newly cut."""
    cut_n = 0
    n = len(keep)
    for lo, hi in cuts:
        lo2 = max(0, int(lo))
        hi2 = min(n, int(hi))
        for i in range(lo2, hi2):
            if keep[i]:
                keep[i] = False
                cut_n += 1
    return cut_n


def trim_hop(
    frames: Sequence[Sequence[int]],
    trace: Sequence[Mapping[str, Any]] | None,
    start_index: int,
    end_index: int,
    *,
    mode: str = "traversal",
    leave_room: int | None = None,
    drop_px: float = 48.0,
    min_loop_frames: int = 45,
    min_idle: int = 40,
    pad_after: int = 8,
    keep_leading_idle: int = 0,
    start_xy: Sequence[float] | None = None,
    end_xy: Sequence[float] | None = None,
) -> tuple[list[list[int]], TrimReport]:
    """Trim one hop slice. Return concatenated kept frames + report.

    ``kept_ranges`` are absolute half-open indices into the original ``frames``.
    """
    mode_l = (mode or "traversal").strip().lower()
    if mode_l not in ("traversal", "combat", "safe"):
        raise ValueError(
            f"mode must be 'traversal', 'combat', or 'safe', got {mode!r}"
        )

    if end_index < start_index:
        report = TrimReport(
            mode=mode_l,
            frames_before=0,
            frames_after=0,
            leading_idle_cut=0,
            trailing_cut=0,
            mid_idle_cut=0,
            retry_loops_cut=0,
            retry_frames_cut=0,
            kept_ranges=[],
            notes=["empty range"],
        )
        return [], report

    start_index = int(start_index)
    end_index = int(end_index)
    n = end_index - start_index + 1
    if start_index < 0 or end_index >= len(frames):
        raise IndexError(
            f"hop range [{start_index}, {end_index}] outside frames len={len(frames)}"
        )

    keep = [True] * n
    notes: list[str] = []

    # --- Leading idle ---
    lead = 0
    while lead < n and is_idle_frame(frames[start_index + lead]):
        lead += 1
    keep_lead = max(0, min(int(keep_leading_idle), lead))
    leading_cut = lead - keep_lead
    for i in range(leading_cut):
        keep[i] = False
    if leading_cut:
        notes.append(f"leading_idle cut {leading_cut} (kept {keep_lead})")

    # --- Trailing after leave_room / phase end / trailing idle pad ---
    trailing_cut = 0
    trail_from: int | None = None  # local index: cut [trail_from, n)

    if leave_room is not None and trace is not None:
        for i in range(n):
            rid = _room_id(_trace_row(trace, start_index + i))
            if rid is not None and int(rid) == int(leave_room):
                trail_from = min(n, i + 1 + max(0, int(pad_after)))
                notes.append(
                    f"leave_room 0x{int(leave_room):04X} at local {i}; "
                    f"pad_after={pad_after}"
                )
                break

    if trail_from is None and trace is not None:
        # Phase-ending heuristic: first door_transition / non-ordinary after
        # we have left ordinary gameplay mid-hop (optional soft signal).
        for i in range(n):
            row = _trace_row(trace, start_index + i)
            if row is None:
                continue
            phase = str(row.get("phase") or "")
            if phase in ("door_transition", "DoorTransition", "DOOR_TRANSITION"):
                # Only treat as leave if room already changed or door flag set.
                if int(row.get("door_transition") or 0) or (
                    leave_room is not None
                    and _room_id(row) == int(leave_room)
                ):
                    trail_from = min(n, i + 1 + max(0, int(pad_after)))
                    notes.append(f"phase door_transition at local {i}")
                    break

    if trail_from is None:
        # Trailing idle pad: after last non-idle kept candidate, keep pad_after zeros.
        last_active = -1
        for i in range(n - 1, -1, -1):
            if is_idle_frame(frames[start_index + i]):
                continue
            last_active = i
            break
        if last_active >= 0:
            trail_from = last_active + 1 + max(0, int(pad_after))
            if trail_from < n:
                notes.append(
                    f"trailing idle after last active local {last_active} "
                    f"+ pad_after={pad_after}"
                )
        elif leading_cut < n:
            # Entire hop idle — leave pad_after after leading keep.
            trail_from = leading_cut + keep_lead + max(0, int(pad_after))
            notes.append("all-idle hop trailing pad")

    if trail_from is not None and trail_from < n:
        for i in range(max(0, trail_from), n):
            if keep[i]:
                keep[i] = False
                trailing_cut += 1

    # Progress signal (full hop; cuts only remove marked frames later).
    progress: list[float] = []
    if trace is not None and len(trace) > end_index:
        progress = progress_along_leave(
            trace,
            start_index,
            end_index,
            start_xy=start_xy,
            end_xy=end_xy,
        )
    else:
        progress = [0.0] * n
        if trace is None:
            notes.append("no trace: skip progress-based cuts")

    retry_loops_cut = 0
    retry_frames_cut = 0
    mid_idle_cut = 0

    if mode_l == "safe":
        notes.append(
            "safe mode: leading+trailing only (open-loop contiguous; "
            "no mid-idle/retry — those need dual-green validation)"
        )
    elif mode_l == "traversal" and progress:
        # --- Retry HWM loops (heuristic; not open-loop-safe without dual green) ---
        retry_cuts = find_retry_loop_cuts(
            progress, drop_px=float(drop_px), min_loop_frames=int(min_loop_frames)
        )
        # Do not re-cut leading/trailing already removed; still count only new.
        for lo, hi in retry_cuts:
            newly = _apply_local_cuts(keep, [(lo, hi)])
            if newly:
                retry_loops_cut += 1
                retry_frames_cut += newly
        if retry_loops_cut:
            notes.append(
                f"retry loops cut={retry_loops_cut} frames={retry_frames_cut} "
                f"drop_px={drop_px} min_loop={min_loop_frames}"
            )

        # --- Mid idle (heuristic; skips frames that still tick enemy RNG) ---
        # Avoid cutting into leading pad zone / trailing already removed.
        mid_cuts = find_mid_idle_cuts(
            frames,
            progress,
            trace,
            start_index,
            min_idle=int(min_idle),
        )
        # Restrict mid cuts to interior that is still kept and not leading/trail edge noise.
        interior_mid: list[tuple[int, int]] = []
        for lo, hi in mid_cuts:
            # Clip to still-relevant interior (skip pure leading already handled).
            lo2 = max(lo, leading_cut)
            hi2 = hi
            if trail_from is not None:
                hi2 = min(hi2, trail_from)
            if hi2 - lo2 >= int(min_idle):
                interior_mid.append((lo2, hi2))
        mid_idle_cut = _apply_local_cuts(keep, interior_mid)
        if mid_idle_cut:
            notes.append(f"mid_idle cut {mid_idle_cut} (min_idle={min_idle})")
    elif mode_l == "combat":
        notes.append("combat mode: no mid-idle / retry cuts")

    kept_ranges = _mask_to_ranges(keep, start_index)
    trimmed: list[list[int]] = []
    for lo, hi in kept_ranges:
        for i in range(lo, hi):
            trimmed.append(_frame_vec(frames[i]))

    report = TrimReport(
        mode=mode_l,
        frames_before=n,
        frames_after=len(trimmed),
        leading_idle_cut=leading_cut,
        trailing_cut=trailing_cut,
        mid_idle_cut=mid_idle_cut,
        retry_loops_cut=retry_loops_cut,
        retry_frames_cut=retry_frames_cut,
        kept_ranges=kept_ranges,
        notes=notes,
    )
    return trimmed, report


def export_trimmed_seed(
    path: Path | str,
    frames: Sequence[Sequence[int]],
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write JSON seed: raw SNES-12 frames + meta (source task, hop, trim)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "frames": [_frame_vec(f) for f in frames],
        "frame_count": len(frames),
        "meta": dict(meta or {}),
    }
    out.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return out


def trim_task_hop(
    task: Mapping[str, Any],
    hop_index: int,
    *,
    hops: Sequence[Mapping[str, Any]] | None = None,
    mode: str | None = None,
    **trim_kwargs: Any,
) -> tuple[list[list[int]], TrimReport, dict[str, Any]]:
    """Convenience: trim hop ``hop_index`` from a loaded guided_human task dict.

    Returns trimmed frames, report, and hop record used.
    """
    from super_metroid.human_tape.hops import load_room_hops  # local import

    frames = list(task.get("frames") or [])
    if hops is None:
        hop_list = load_room_hops(task_data=task, settle=True)
    else:
        hop_list = list(hops)
    if hop_index < 0 or hop_index >= len(hop_list):
        raise IndexError(f"hop {hop_index} out of range (n={len(hop_list)})")
    hop = dict(hop_list[hop_index])
    si = int(hop["start_index"])
    ei = int(hop["end_index"])
    room_id = int(hop.get("room_id") or 0)
    mode_use = mode or infer_mode(room_id)

    leave_room = trim_kwargs.pop("leave_room", None)
    if leave_room is None and hop_index + 1 < len(hop_list):
        leave_room = int(hop_list[hop_index + 1].get("room_id") or 0)

    # Expand end to include leave room pad frames when next hop exists so
    # trailing leave_room cut can fire (hop end is last frame of current room).
    end_for_trim = ei
    pad_after = int(trim_kwargs.get("pad_after", 8))
    if leave_room is not None and hop_index + 1 < len(hop_list):
        next_si = int(hop_list[hop_index + 1]["start_index"])
        # Include next-room frames up to pad_after + small margin for leave detect.
        end_for_trim = min(len(frames) - 1, max(ei, next_si + pad_after))

    start_xy = hop.get("xy")
    end_xy = hop.get("end_xy")
    trimmed, report = trim_hop(
        frames,
        trace,
        si,
        end_for_trim,
        mode=mode_use,
        leave_room=leave_room,
        start_xy=start_xy,
        end_xy=end_xy,
        **trim_kwargs,
    )
    return trimmed, report, hop


__all__ = [
    "BTN_A",
    "BTN_B",
    "BTN_X",
    "BTN_Y",
    "COMBAT_ROOM_IDS",
    "TrimReport",
    "export_trimmed_seed",
    "find_mid_idle_cuts",
    "find_retry_loop_cuts",
    "holds_charge_safe_buttons",
    "infer_mode",
    "is_combat_room",
    "is_idle_frame",
    "progress_along_leave",
    "trim_hop",
    "trim_task_hop",
]
