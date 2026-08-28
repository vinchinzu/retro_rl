"""Isolated 1-3 autobot (32-exit). Replay a clear seed; scout inserts jumps.

Start is ``Level1_3`` (dash_level=2). Completion is 1-4 control
(``dash_level==3``), never AreaNumber. Warp any% is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import NES_ACTION_SIZE, nes_idle_action
from smb.paths import MODELS_DIR
from smb.policy import expand_nes9_rle, load_nes9_rle_seed
from smb.ram import (
    PLAYER_STATE_AUTO_WALK,
    PLAYER_STATE_FLAGPOLE,
    SmbSnapshot,
    player_on_ground,
    read_snapshot,
)
from smb.tas.stages import is_1_4_control

DEFAULT_1_3_SEED = MODELS_DIR / "smb_1_3_clear.json"

_IDLE = [0, 0, 0, 0, 0, 0, 0, 0, 0]
_RB = [1, 0, 0, 0, 0, 0, 0, 1, 0]
_RBA = [1, 0, 0, 0, 0, 0, 0, 1, 1]

JUMP_HOLD = 16
JUMP_AHEAD = 40
DEFAULT_MAX_FRAMES = 5000


def _nes9(buttons: list[int], action_size: int = NES_ACTION_SIZE) -> np.ndarray:
    b = list(buttons[:action_size])
    if len(b) < action_size:
        b.extend([0] * (action_size - len(b)))
    return np.array(b, dtype=np.int8)


def _idle(action_size: int = NES_ACTION_SIZE) -> np.ndarray:
    idle = np.asarray(nes_idle_action(), dtype=np.int8)
    if idle.shape[0] != action_size:
        idle = np.zeros(action_size, dtype=np.int8)
    return idle


@dataclass
class BunnyHopPolicy:
    """Run-right; tap A on every physics-grounded frame (athletic 1-3)."""

    hold_a: int = JUMP_HOLD
    action_size: int = NES_ACTION_SIZE
    jump_held: int = field(default=0, init=False)
    index: int = field(default=0, init=False)
    _recorded: list[list[int]] = field(default_factory=list, repr=False)

    def reset(self) -> None:
        self.jump_held = 0
        self.index = 0
        self._recorded.clear()

    def recorded_frames(self) -> list[list[int]]:
        return [list(row) for row in self._recorded]

    def step(self, snap: SmbSnapshot, *, on_ground: bool) -> FrameAction:
        if int(snap.player_state) in (PLAYER_STATE_FLAGPOLE, PLAYER_STATE_AUTO_WALK):
            action = _idle(self.action_size)
            reason = "1_3_pole"
        elif is_1_4_control(snap):
            action = _idle(self.action_size)
            reason = "1_4_control"
        elif self.jump_held > 0:
            self.jump_held -= 1
            action = _nes9(_RBA, self.action_size)
            reason = "1_3_air"
        elif on_ground:
            self.jump_held = max(0, self.hold_a - 1)
            action = _nes9(_RBA, self.action_size)
            reason = "1_3_bunny"
        else:
            action = _nes9(_RB, self.action_size)
            reason = "1_3_run"
        self._recorded.append([int(b) for b in action[:9]])
        self.index += 1
        return FrameAction(action, reason)


@dataclass
class JumpWindowPolicy:
    """Run-right 1-3 with A held at listed world-x windows (control-relative)."""

    jump_xs: list[int] = field(default_factory=list)
    hold_a: int = JUMP_HOLD
    ahead: int = JUMP_AHEAD
    action_size: int = NES_ACTION_SIZE
    jump_held: int = field(default=0, init=False)
    used: set[int] = field(default_factory=set, init=False)
    index: int = field(default=0, init=False)
    _recorded: list[list[int]] = field(default_factory=list, repr=False)

    def reset(self) -> None:
        self.jump_held = 0
        self.used.clear()
        self.index = 0
        self._recorded.clear()

    def recorded_frames(self) -> list[list[int]]:
        return [list(row) for row in self._recorded]

    def step(self, snap: SmbSnapshot, *, on_ground: bool) -> FrameAction:
        if int(snap.player_state) in (PLAYER_STATE_FLAGPOLE, PLAYER_STATE_AUTO_WALK):
            action = _idle(self.action_size)
            reason = "1_3_pole"
        elif is_1_4_control(snap):
            action = _idle(self.action_size)
            reason = "1_4_control"
        elif self.jump_held > 0:
            self.jump_held -= 1
            action = _nes9(_RBA, self.action_size)
            reason = "1_3_jump"
        else:
            x = int(snap.player_x)
            fire: int | None = None
            for jx in self.jump_xs:
                if jx in self.used:
                    continue
                if jx - self.ahead <= x <= jx + 8:
                    fire = jx
                    break
            if fire is not None and on_ground:
                self.used.add(fire)
                self.jump_held = max(0, self.hold_a - 1)
                action = _nes9(_RBA, self.action_size)
                reason = f"1_3_jump_{fire}"
            else:
                action = _nes9(_RB, self.action_size)
                reason = "1_3_run"
        self._recorded.append([int(b) for b in action[:9]])
        self.index += 1
        return FrameAction(action, reason)


@dataclass
class Level13ReplayPolicy:
    """Frame-perfect 1-3 clear from ``smb_1_3_clear.json`` (isolated Level1_3)."""

    seed_path: Path = DEFAULT_1_3_SEED
    action_size: int = NES_ACTION_SIZE
    frames: list[list[int]] = field(default_factory=list, repr=False)
    index: int = field(default=0, init=False)
    exhausted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.frames and self.seed_path.is_file():
            self.frames = expand_nes9_rle(load_nes9_rle_seed(self.seed_path))

    def reset(self) -> None:
        self.index = 0
        self.exhausted = False

    def step(self, snap: SmbSnapshot | None = None) -> FrameAction:
        del snap
        if self.index >= len(self.frames):
            self.exhausted = True
            return FrameAction(_idle(self.action_size), "1_3_exhausted")
        buttons = list(self.frames[self.index][:9])
        self.index += 1
        return FrameAction(_nes9(buttons, self.action_size), f"1_3_{self.index}")


def play_1_3(
    env: Any,
    *,
    policy: JumpWindowPolicy | BunnyHopPolicy | Level13ReplayPolicy,
    max_frames: int = DEFAULT_MAX_FRAMES,
    start_lives: int | None = None,
) -> dict[str, Any]:
    """Run one 1-3 attempt from the current env. Success = 1-4 control."""
    if hasattr(policy, "reset"):
        policy.reset()
    snap = read_snapshot(env.get_ram())
    lives0 = start_lives if start_lives is not None else snap.lives
    max_x = int(snap.player_x)
    outcome = "timeout"
    end_frame = 0
    for frame in range(1, max_frames + 1):
        snap = read_snapshot(env.get_ram(), frame=frame - 1)
        if is_1_4_control(snap):
            outcome = "success"
            end_frame = frame - 1
            break
        on_ground = player_on_ground(env.get_ram())
        if isinstance(policy, (JumpWindowPolicy, BunnyHopPolicy)):
            tick = policy.step(snap, on_ground=on_ground)
        else:
            tick = policy.step(snap)
        env.step(tick.action)
        end_frame = frame
        snap = read_snapshot(env.get_ram(), frame=frame)
        max_x = max(max_x, int(snap.player_x))
        if snap.lives < lives0 or snap.dying:
            outcome = "death"
            break
        if is_1_4_control(snap):
            outcome = "success"
            break
    recorded = policy.recorded_frames() if hasattr(policy, "recorded_frames") else []
    return {
        "ok": outcome == "success",
        "outcome": outcome,
        "frames": end_frame,
        "max_x": max_x,
        "death_x": max_x if outcome == "death" else None,
        "snap": {
            "world": int(snap.world),
            "dash_level": int(snap.dash_level),
            "x": int(snap.player_x),
            "y": int(snap.player_y),
            "ps": int(snap.player_state),
            "timer": int(snap.timer),
        },
        "recorded": recorded,
    }
