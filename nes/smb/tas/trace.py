"""Replay a 1-1 seed and capture a TAS-oriented timeline.

Reports flagpole touch, castle auto-walk, level exit, wall-slams (xs→0 while
grounded with prior speed), and progress stalls. Used by the analyzer and as
input to window discovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from retro_harness.env import make_env
from retro_harness.platformer.frame_tools import clone_frames, find_stalls, is_idle_frame
from smb.paths import GAME_DIR
from smb.ram import (
    LEVEL_ID_1_1,
    left_level_1_1,
    read_snapshot,
)

# player_state values during end-of-level automation (disassembly)
PLAYER_STATE_FLAGPOLE = 4
PLAYER_STATE_CASTLE = 5
# Treat these as "still controllable physics" for wall-slam detection
CONTROLLABLE_GROUNDED = frozenset({0x00, 0x08})


@dataclass(frozen=True)
class TraceEvent:
    """A named milestone or problem frame in a seed."""

    frame: int
    kind: str
    x: int = 0
    y: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SeedTrace:
    """Full timeline + summary metrics for a 1-1 seed replay."""

    num_frames: int
    completed: bool
    flag_frame: int | None = None
    castle_frame: int | None = None
    leave_frame: int | None = None
    max_player_x: int = 0
    max_x_speed: int = 0
    died: bool = False
    death_frame: int | None = None
    events: list[TraceEvent] = field(default_factory=list)
    # Per-frame series (optional; kept short for reports)
    wall_slams: list[TraceEvent] = field(default_factory=list)
    stalls: list[dict[str, Any]] = field(default_factory=list)
    xs_zero_runs: list[dict[str, Any]] = field(default_factory=list)
    player_states: list[int] = field(default_factory=list)
    player_x: list[int] = field(default_factory=list)
    x_speeds: list[int] = field(default_factory=list)

    @property
    def clear_frames(self) -> int | None:
        """Frames until level_id leaves 1-1 (first leave frame)."""
        return self.leave_frame

    @property
    def controllable_frames(self) -> int | None:
        """Frames until flagpole grab (player_state=4)."""
        return self.flag_frame

    def summary(self) -> dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "completed": self.completed,
            "flag_frame": self.flag_frame,
            "castle_frame": self.castle_frame,
            "leave_frame": self.leave_frame,
            "clear_frames": self.clear_frames,
            "controllable_frames": self.controllable_frames,
            "max_player_x": self.max_player_x,
            "max_x_speed": self.max_x_speed,
            "died": self.died,
            "death_frame": self.death_frame,
            "wall_slam_count": len(self.wall_slams),
            "stall_count": len(self.stalls),
            "xs_zero_run_count": len(self.xs_zero_runs),
            "events": [e.to_dict() for e in self.events],
            "wall_slams": [e.to_dict() for e in self.wall_slams[:40]],
            "stalls": self.stalls[:40],
            "xs_zero_runs": self.xs_zero_runs[:40],
        }


def _pad9(frame: Sequence[int]) -> list[int]:
    buttons = [int(b) for b in frame[:9]]
    if len(buttons) < 9:
        buttons.extend([0] * (9 - len(buttons)))
    return buttons


def trace_seed(
    frames: Sequence[Sequence[int]],
    *,
    state_name: str = "Level1_1",
    game_dir: Path | None = None,
    pad_idle_after: int = 0,
    min_stall_len: int = 12,
    keep_series: bool = True,
) -> SeedTrace:
    """Replay *frames* from ``Level1_1`` and collect TAS metrics.

    Parameters
    ----------
    pad_idle_after:
        Extra idle frames after the seed (useful when a seed flag-grabs but
        ends a few frames before the level-load detect).
    """
    game_dir = game_dir or GAME_DIR
    buttons = [_pad9(f) for f in frames]
    if pad_idle_after > 0:
        idle = [0] * 9
        buttons = buttons + [list(idle) for _ in range(pad_idle_after)]

    env = make_env(
        game="SuperMarioBros-Nes-v0",
        state=state_name,
        game_dir=game_dir,
        render_mode="rgb_array",
    )
    try:
        env.reset()
        start_lives = int(env.get_ram()[0x075A])
        snap0 = read_snapshot(env.get_ram(), 0)

        flag_frame: int | None = None
        castle_frame: int | None = None
        leave_frame: int | None = None
        death_frame: int | None = None
        died = False
        max_x = snap0.player_x
        max_xs = 0
        events: list[TraceEvent] = []
        wall_slams: list[TraceEvent] = []
        player_states: list[int] = []
        player_x_series: list[int] = []
        x_speeds: list[int] = []
        progress_series: list[float] = []
        prev_xs = 0
        xs_zero_start: int | None = None
        xs_zero_runs: list[dict[str, Any]] = []

        for i, frame in enumerate(buttons):
            env.step(np.array(frame, dtype=np.int8))
            ram = env.get_ram()
            snap = read_snapshot(ram, i + 1)
            max_x = max(max_x, snap.player_x)
            max_xs = max(max_xs, abs(snap.x_speed))
            progress_series.append(float(max_x))
            if keep_series:
                player_states.append(snap.player_state)
                player_x_series.append(snap.player_x)
                x_speeds.append(snap.x_speed)

            if not died and (snap.dying or int(ram[0x075A]) < start_lives):
                died = True
                death_frame = i + 1
                events.append(
                    TraceEvent(i + 1, "death", snap.player_x, snap.player_y, "lives/dying")
                )

            if flag_frame is None and snap.player_state == PLAYER_STATE_FLAGPOLE:
                flag_frame = i + 1
                events.append(
                    TraceEvent(i + 1, "flag", snap.player_x, snap.player_y)
                )
            if castle_frame is None and snap.player_state == PLAYER_STATE_CASTLE:
                castle_frame = i + 1
                events.append(
                    TraceEvent(i + 1, "castle", snap.player_x, snap.player_y)
                )
            if (
                leave_frame is None
                and max_x >= 2500
                and left_level_1_1(ram, start_level_id=LEVEL_ID_1_1)
            ):
                leave_frame = i + 1
                events.append(
                    TraceEvent(
                        i + 1,
                        "leave",
                        snap.player_x,
                        snap.player_y,
                        f"level_id={snap.level_id}",
                    )
                )

            # Wall-slam: grounded controllable, xs collapses from run speed
            grounded = (
                snap.player_state in CONTROLLABLE_GROUNDED and not snap.in_air
            )
            if (
                grounded
                and snap.x_speed == 0
                and prev_xs >= 16
                and snap.player_x > 80
                and flag_frame is None
            ):
                ev = TraceEvent(
                    i + 1,
                    "wall_slam",
                    snap.player_x,
                    snap.player_y,
                    f"prev_xs={prev_xs}",
                )
                wall_slams.append(ev)
                events.append(ev)

            # Track zero-speed runs while still pre-flag and grounded
            if (
                flag_frame is None
                and grounded
                and snap.x_speed == 0
                and snap.player_x > 80
            ):
                if xs_zero_start is None:
                    xs_zero_start = i + 1
            else:
                if xs_zero_start is not None:
                    length = (i + 1) - xs_zero_start
                    if length >= 4:
                        xs_zero_runs.append(
                            {
                                "start": xs_zero_start,
                                "length": length,
                                "x": player_x_series[xs_zero_start - 1]
                                if player_x_series
                                else 0,
                            }
                        )
                    xs_zero_start = None

            prev_xs = snap.x_speed

            # Stop a bit after leave to keep traces short
            if leave_frame is not None and i + 1 >= leave_frame + 5:
                break

        if xs_zero_start is not None and flag_frame is None:
            length = len(buttons) - xs_zero_start + 1
            if length >= 4:
                xs_zero_runs.append(
                    {
                        "start": xs_zero_start,
                        "length": length,
                        "x": player_x_series[xs_zero_start - 1]
                        if player_x_series
                        else 0,
                    }
                )

        stalls = find_stalls(progress_series, min_length=min_stall_len)
        stall_dicts = [
            {
                "start": s.start,
                "length": s.length,
                "x": int(progress_series[s.start]) if s.start < len(progress_series) else 0,
                "reason": "no_progress",
            }
            for s in stalls
            # Ignore post-flag castle idle as "stall waste" for windowing
            if flag_frame is None or s.start < flag_frame
        ]

        return SeedTrace(
            num_frames=len(buttons),
            completed=leave_frame is not None and not died,
            flag_frame=flag_frame,
            castle_frame=castle_frame,
            leave_frame=leave_frame,
            max_player_x=max_x,
            max_x_speed=max_xs,
            died=died,
            death_frame=death_frame,
            events=events,
            wall_slams=wall_slams,
            stalls=stall_dicts,
            xs_zero_runs=xs_zero_runs,
            player_states=player_states if keep_series else [],
            player_x=player_x_series if keep_series else [],
            x_speeds=x_speeds if keep_series else [],
        )
    finally:
        env.close()


def pad_to_completion(
    frames: Sequence[Sequence[int]],
    *,
    max_pad: int = 900,
    state_name: str = "Level1_1",
    game_dir: Path | None = None,
) -> tuple[list[list[int]], SeedTrace]:
    """Append idle frames until the seed leaves 1-1 (or *max_pad* exhausted).

    Returns ``(padded_frames, final_trace)``. If already complete, returns a
    clone without extra pad.
    """
    base = clone_frames(frames)
    # Normalize to 9-button NES vectors for seeds
    base = [[int(b) for b in f[:9]] + [0] * max(0, 9 - len(f[:9])) for f in base]

    probe = trace_seed(base, state_name=state_name, game_dir=game_dir, pad_idle_after=0)
    if probe.completed:
        return base, probe

    # Binary-ish: try increasing pads
    for pad in (50, 100, 200, 400, max_pad):
        pad = min(pad, max_pad)
        tr = trace_seed(
            base, state_name=state_name, game_dir=game_dir, pad_idle_after=pad
        )
        if tr.completed and tr.leave_frame is not None:
            need = tr.leave_frame - len(base)
            # keep a small post-leave pad for evaluator debounce safety
            keep_extra = max(0, need) + 20
            padded = base + [[0] * 9 for _ in range(keep_extra)]
            final = trace_seed(
                padded, state_name=state_name, game_dir=game_dir, pad_idle_after=0
            )
            return padded, final

    return base, probe
