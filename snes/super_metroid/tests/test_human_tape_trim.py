"""Unit tests for offline human hop idle/retry trim (no emulator)."""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid.human_tape.trim import (
    COMBAT_ROOM_IDS,
    export_trimmed_seed,
    find_retry_loop_cuts,
    is_idle_frame,
    progress_along_leave,
    trim_hop,
)


def _idle() -> list[int]:
    return [0] * 12


def _right() -> list[int]:
    # SNES-12: index 7 = Right
    f = [0] * 12
    f[7] = 1
    return f


def _b_held() -> list[int]:
    f = [0] * 12
    f[0] = 1  # B
    return f


def _trace_row(
    frame: int,
    x: int,
    y: int,
    *,
    room: int = 0xDE4D,
    vx: int = 0,
    vy: int = 0,
    pose: int = 1,
) -> dict:
    return {
        "frame": frame,
        "x": x,
        "y": y,
        "room": room,
        "room_hex": f"0x{room:04X}",
        "pose": pose,
        "vx": vx,
        "vy": vy,
        "buttons": [],
        "door_transition": 0,
        "phase": "ordinary_gameplay",
    }


def test_leading_idle_cut() -> None:
    """Leading idle of 20 zeros → cut."""
    n = 30
    frames = [_idle() for _ in range(20)] + [_right() for _ in range(10)]
    # move right in x after idle
    trace = []
    for i in range(n):
        x = 0 if i < 20 else (i - 20) * 4
        trace.append(_trace_row(i, x, 100, vx=0 if i < 20 else 4))
    trimmed, report = trim_hop(
        frames,
        trace,
        0,
        n - 1,
        mode="traversal",
        min_idle=999,  # disable mid
        min_loop_frames=999,  # disable retry
        pad_after=8,
    )
    assert report.leading_idle_cut == 20
    assert report.frames_before == 30
    assert report.frames_after == 10  # 10 active; no trailing idle beyond pad
    assert len(trimmed) == 10
    assert all(not is_idle_frame(f) for f in trimmed)
    # kept_ranges absolute
    assert report.kept_ranges == [(20, 30)]


def test_synthetic_retry_loop_cut() -> None:
    """Progress advances, retreats, recovers → loop cut, final length shorter."""
    # Path along +x: 0 → 100, retreat to 40, re-climb to 100, then to 150.
    xs: list[int] = []
    # approach to 100 (50 frames, +2px)
    for i in range(51):
        xs.append(i * 2)  # 0..100
    # retreat to 40 (40 frames)
    for i in range(1, 41):
        xs.append(100 - i * 1.5)  # down toward 40
    # re-climb to 100 (40 frames)
    start_re = xs[-1]
    for i in range(1, 41):
        xs.append(start_re + (100 - start_re) * i / 40)
    # continue to 150 (25 frames)
    for i in range(1, 26):
        xs.append(100 + i * 2)

    n = len(xs)
    frames = [_right() for _ in range(n)]
    trace = [_trace_row(i, int(xs[i]), 200, vx=2) for i in range(n)]

    prog = progress_along_leave(trace, 0, n - 1, start_xy=(xs[0], 200), end_xy=(xs[-1], 200))
    assert max(prog) > 100
    cuts = find_retry_loop_cuts(prog, drop_px=48, min_loop_frames=45)
    assert len(cuts) >= 1
    loop_frames = sum(hi - lo for lo, hi in cuts)
    assert loop_frames >= 45

    trimmed, report = trim_hop(
        frames,
        trace,
        0,
        n - 1,
        mode="traversal",
        drop_px=48,
        min_loop_frames=45,
        min_idle=999,  # no mid idle
        pad_after=100,  # don't trail-cut active frames
        keep_leading_idle=0,
    )
    assert report.retry_loops_cut >= 1
    assert report.retry_frames_cut >= 45
    assert report.frames_after < report.frames_before
    assert len(trimmed) == report.frames_after
    # Concat of kept_ranges equals trimmed length
    kept_n = sum(hi - lo for lo, hi in report.kept_ranges)
    assert kept_n == report.frames_after


def test_combat_mode_no_mid_or_retry() -> None:
    """Combat mode does not cut mid idle / retry."""
    # Build: advance, big retreat+recover (retry-like), plus long mid idle.
    xs: list[int] = []
    for i in range(40):
        xs.append(i * 3)  # 0..117
    peak = xs[-1]
    for i in range(50):
        xs.append(peak - i)  # retreat
    for i in range(50):
        xs.append(xs[-1] + 1)  # recover past peak
    # long idle stretch at constant x
    idle_x = xs[-1]
    for _ in range(60):
        xs.append(idle_x)
    # leave
    for i in range(20):
        xs.append(idle_x + i * 2)

    n = len(xs)
    # approach 40 + retreat 50 + recover 50 = 140; then 60 idle; then leave
    idle_lo, idle_hi = 140, 200
    frames: list[list[int]] = []
    trace = []
    for i in range(n):
        if idle_lo <= i < idle_hi:
            frames.append(_idle())
            vx = 0
        else:
            frames.append(_right())
            vx = 2
        trace.append(
            _trace_row(
                i,
                int(xs[i]),
                100,
                room=0xDD58,  # Mother Brain combat room
                vx=vx,
            )
        )

    # Confirm traversal would cut something
    t_trim, t_rep = trim_hop(
        frames,
        trace,
        0,
        n - 1,
        mode="traversal",
        drop_px=48,
        min_loop_frames=45,
        min_idle=40,
        pad_after=8,
    )
    assert t_rep.retry_frames_cut > 0 or t_rep.mid_idle_cut > 0

    trimmed, report = trim_hop(
        frames,
        trace,
        0,
        n - 1,
        mode="combat",
        drop_px=48,
        min_loop_frames=45,
        min_idle=40,
        pad_after=8,
    )
    assert report.mode == "combat"
    assert report.retry_loops_cut == 0
    assert report.retry_frames_cut == 0
    assert report.mid_idle_cut == 0
    # combat may still cut leading idle / trailing
    assert report.frames_after == n - report.leading_idle_cut - report.trailing_cut
    assert 0xDD58 in COMBAT_ROOM_IDS


def test_b_held_not_mid_idle_cut() -> None:
    """B-held frames are not mid-idle-cut even if velocity/progress flat."""
    n = 80
    frames = [_b_held() for _ in range(n)]  # B held whole time, standing
    trace = [
        _trace_row(i, 50, 100, vx=0, vy=0) for i in range(n)
    ]
    # end_xy same-ish so progress flat
    trimmed, report = trim_hop(
        frames,
        trace,
        0,
        n - 1,
        mode="traversal",
        min_idle=40,
        min_loop_frames=999,
        pad_after=8,
        start_xy=(50, 100),
        end_xy=(50, 100),
    )
    assert report.mid_idle_cut == 0
    # B-held is not "idle" (all zeros) so leading/trailing idle also 0;
    # trailing pad logic: last non-idle is last frame → no trailing cut
    assert report.leading_idle_cut == 0
    assert report.frames_after == n
    assert len(trimmed) == n


def test_export_trimmed_seed(tmp_path: Path) -> None:
    frames = [_right(), _idle(), _right()]
    meta = {"source_task": "demo.json", "hop_index": 1}
    path = export_trimmed_seed(tmp_path / "seed.json", frames, meta)
    data = json.loads(path.read_text())
    assert data["frame_count"] == 3
    assert data["frames"][0][7] == 1
    assert data["meta"]["hop_index"] == 1


def test_keep_leading_idle_param() -> None:
    frames = [_idle()] * 10 + [_right()] * 5
    trace = [_trace_row(i, min(i, 10) * 3, 0) for i in range(15)]
    _, report = trim_hop(
        frames,
        trace,
        0,
        14,
        mode="combat",
        keep_leading_idle=2,
        min_idle=999,
        min_loop_frames=999,
        pad_after=8,
    )
    assert report.leading_idle_cut == 8
    assert report.kept_ranges[0][0] == 8
