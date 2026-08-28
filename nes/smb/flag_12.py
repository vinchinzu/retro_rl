"""1-2 flag-exit body for the 32-exit track (not the W4 warp).

HL underground prefix until the end-of-UG lift, then the measured lift/pipe
tail, then outdoor stairs/flagpole into 1-3 control. Warp any% fragments
and ``smb_1_2_reactive_fragments.json`` stay untouched.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.run_1_2_flag --record
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
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
from smb.reactive_12 import SURFACE_MAX_WAIT, is_surface_control
from smb.reactive_route import GateWaiter, StateGate
from smb.tas.stages import CONTROL_X_MAX, is_1_3_control

DEFAULT_1_2_FLAG_SEED = MODELS_DIR / "smb_1_2_flag.json"

# NES-9: B=0, D=5, L=6, R=7, A=8
_IDLE = [0, 0, 0, 0, 0, 0, 0, 0, 0]
_RIGHT = [0, 0, 0, 0, 0, 0, 0, 1, 0]
_RB = [1, 0, 0, 0, 0, 0, 0, 1, 0]
_RBA = [1, 0, 0, 0, 0, 0, 0, 1, 1]
_RD = [0, 0, 0, 0, 0, 1, 0, 1, 0]
_DOWN = [0, 0, 0, 0, 0, 1, 0, 0, 0]
_A = [0, 0, 0, 0, 0, 0, 0, 0, 1]

HL_LIFT_A_HOLD = 19
LIFT_X_MIN = 2480
LIFT_X_MAX = 2560
LIFT_Y_MIN = 140
LIFT_Y_MAX = 160
PIPE_WALK_X = 2646
OUTDOOR_AREA_POINTER = 194
PLAYER_STATE_PIPE_WALK = 2
PLAYER_STATE_PIPE_ENTER = 3
DEFAULT_MAX_FRAMES = 4500
OUTDOOR_MAX_FRAMES = 1600


class Phase(Enum):
    WAIT_SURFACE = auto()
    BODY = auto()
    DONE = auto()
    FAILED = auto()


class TailPhase(Enum):
    JUMP = auto()
    COAST = auto()
    WALK = auto()
    PIPE = auto()
    OUTDOOR = auto()
    DONE = auto()
    FAILED = auto()


def is_lift_pose(snap: SmbSnapshot) -> bool:
    """Last physics-grounded HL pose on the end-of-UG lift (x≈2520 y≈148)."""
    return (
        int(snap.world) == 0
        and int(snap.dash_level) == 1
        and LIFT_X_MIN <= int(snap.player_x) <= LIFT_X_MAX
        and LIFT_Y_MIN <= int(snap.player_y) <= LIFT_Y_MAX
        and int(snap.x_speed) >= 24
        and not snap.dying
    )


def is_outdoor_flag_area(snap: SmbSnapshot) -> bool:
    """World-0 flag courtyard after the UG exit pipe (area 194)."""
    if int(snap.world) != 0 or int(snap.dash_level) != 1 or snap.dying:
        return False
    if int(snap.player_state) in (PLAYER_STATE_FLAGPOLE, PLAYER_STATE_AUTO_WALK):
        return True
    if int(snap.area_pointer) == OUTDOOR_AREA_POINTER:
        return int(snap.player_x) > 0 and int(snap.player_y) > 0
    return False


def is_pipe_transition(snap: SmbSnapshot) -> bool:
    ps = int(snap.player_state)
    if ps in (0, PLAYER_STATE_PIPE_WALK, PLAYER_STATE_PIPE_ENTER, 7):
        return True
    return int(snap.player_x) == 0 and int(snap.player_y) == 0


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
class FlagTailController:
    """From the lift: A-only 19f, idle coast, walk onto the short pipe, outdoor.

    Standing DOWN does not enter; walking onto the lip (x≈2646, ps=2) does.
    Jumps only take if ``on_ground`` ($001D==0) — ``snap.grounded`` lags.
    """

    hold_a: int = HL_LIFT_A_HOLD
    action_size: int = NES_ACTION_SIZE
    phase: TailPhase = field(default=TailPhase.JUMP, init=False)
    jump_held: int = field(default=0, init=False)
    stall_x: int = field(default=-1, init=False)
    stall_n: int = field(default=0, init=False)
    outdoor_frames: int = field(default=0, init=False)

    def reset(self) -> None:
        self.phase = TailPhase.JUMP
        self.jump_held = 0
        self.stall_x = -1
        self.stall_n = 0
        self.outdoor_frames = 0

    @property
    def done(self) -> bool:
        return self.phase in (TailPhase.DONE, TailPhase.FAILED)

    def step(self, snap: SmbSnapshot, *, on_ground: bool) -> FrameAction:
        if is_1_3_control(snap):
            self.phase = TailPhase.DONE
            return FrameAction(_idle(self.action_size), "flag_1_3")
        if snap.dying or int(snap.world) != 0:
            self.phase = TailPhase.FAILED
            return FrameAction(_idle(self.action_size), "flag_tail_fail")

        if self.phase is TailPhase.JUMP:
            if self.jump_held == 0 and not on_ground:
                return FrameAction(_idle(self.action_size), "flag_wait_ground")
            self.jump_held += 1
            if self.jump_held >= self.hold_a:
                self.phase = TailPhase.COAST
            return FrameAction(_nes9(_A, self.action_size), f"flag_a_{self.jump_held}")

        if self.phase is TailPhase.COAST:
            if on_ground:
                self.phase = TailPhase.WALK
                return self._walk(snap)
            return FrameAction(_idle(self.action_size), "flag_coast")

        if self.phase is TailPhase.WALK:
            if is_pipe_transition(snap) or int(snap.player_state) in (
                PLAYER_STATE_PIPE_WALK,
                PLAYER_STATE_PIPE_ENTER,
            ):
                self.phase = TailPhase.PIPE
                return FrameAction(_nes9(_DOWN, self.action_size), "flag_pipe")
            return self._walk(snap)

        if self.phase is TailPhase.PIPE:
            if is_outdoor_flag_area(snap):
                self.phase = TailPhase.OUTDOOR
                return self._outdoor(snap, on_ground=on_ground)
            return FrameAction(_nes9(_DOWN, self.action_size), "flag_pipe")

        if self.phase is TailPhase.OUTDOOR:
            self.outdoor_frames += 1
            if self.outdoor_frames > OUTDOOR_MAX_FRAMES:
                self.phase = TailPhase.FAILED
                return FrameAction(_idle(self.action_size), "flag_outdoor_timeout")
            return self._outdoor(snap, on_ground=on_ground)

        return FrameAction(_idle(self.action_size), "flag_tail_idle")

    def _walk(self, snap: SmbSnapshot) -> FrameAction:
        if int(snap.player_x) >= PIPE_WALK_X:
            return FrameAction(_nes9(_RD, self.action_size), "flag_lip")
        return FrameAction(_nes9(_RIGHT, self.action_size), "flag_walk")

    def _outdoor(self, snap: SmbSnapshot, *, on_ground: bool) -> FrameAction:
        x = int(snap.player_x)
        if x == self.stall_x:
            self.stall_n += 1
        else:
            self.stall_x = x
            self.stall_n = 0
        if int(snap.player_state) in (PLAYER_STATE_FLAGPOLE, PLAYER_STATE_AUTO_WALK):
            return FrameAction(_idle(self.action_size), "flag_pole")
        if self.stall_n >= 8 and on_ground:
            return FrameAction(_nes9(_RBA, self.action_size), "flag_stall_hop")
        return FrameAction(_nes9(_RB, self.action_size), "flag_run")


@dataclass
class Flag12Policy:
    """State-gated 1-2 flag body: wait surface control, replay HL+tail seed."""

    seed_path: Path = DEFAULT_1_2_FLAG_SEED
    action_size: int = NES_ACTION_SIZE
    max_surface_wait: int = SURFACE_MAX_WAIT
    frames: list[list[int]] = field(default_factory=list, repr=False)
    phase: Phase = field(default=Phase.WAIT_SURFACE, init=False)
    index: int = field(default=0, init=False)
    total_steps: int = field(default=0, init=False)
    _recorded: list[list[int]] = field(default_factory=list, repr=False)
    _surface_waiter: GateWaiter = field(init=False, repr=False)
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.frames and self.seed_path.is_file():
            data = load_nes9_rle_seed(self.seed_path)
            self.frames = expand_nes9_rle(data)
        self._surface_waiter = GateWaiter(
            StateGate(
                "1-2:surface_control",
                is_surface_control,
                "controllable 1-2 surface after the 1-1 predecessor",
            ),
            max_frames=self.max_surface_wait,
        )

    def reset(self) -> None:
        self.phase = Phase.WAIT_SURFACE
        self.index = 0
        self.total_steps = 0
        self._recorded.clear()
        self.log.clear()
        self._surface_waiter.reset()

    @property
    def done(self) -> bool:
        return self.phase in (Phase.DONE, Phase.FAILED)

    @property
    def success(self) -> bool:
        return self.phase is Phase.DONE

    def recorded_frames(self) -> list[list[int]]:
        return [list(row) for row in self._recorded]

    def step(self, snap: SmbSnapshot) -> FrameAction:
        if self.phase is Phase.DONE:
            return FrameAction(_idle(self.action_size), "flag12_done")
        if self.phase is Phase.FAILED:
            return FrameAction(_idle(self.action_size), "flag12_failed")
        if is_1_3_control(snap):
            self.phase = Phase.DONE
            self.log.append({"event": "1_3_control", "step": self.total_steps})
            return FrameAction(_idle(self.action_size), "flag12_1_3")
        if snap.dying or int(snap.world) != 0:
            self.phase = Phase.FAILED
            self.log.append({"event": "death_or_warp", "step": self.total_steps})
            return FrameAction(_idle(self.action_size), "flag12_fail")

        if self.phase is Phase.WAIT_SURFACE:
            if self._surface_waiter.observe(snap):
                self.phase = Phase.BODY
                self.log.append(
                    {
                        "event": "surface_control",
                        "wait": self._surface_waiter.frames_waited,
                        "x": snap.player_x,
                    }
                )
                return self._body_action()
            if self._surface_waiter.timed_out:
                self.phase = Phase.FAILED
                self.log.append({"event": "timeout_surface_wait"})
                return FrameAction(_idle(self.action_size), "flag12_timeout_surface")
            return FrameAction(_idle(self.action_size), "flag12_wait_surface")

        return self._body_action()

    def _body_action(self) -> FrameAction:
        if self.index >= len(self.frames):
            self.phase = Phase.FAILED
            self.log.append({"event": "body_exhausted", "index": self.index})
            return FrameAction(_idle(self.action_size), "flag12_exhausted")
        buttons = list(self.frames[self.index][:9])
        self.index += 1
        action = _nes9(buttons, self.action_size)
        self._recorded.append([int(x) for x in action[:9]])
        self.total_steps += 1
        return FrameAction(action, f"flag12_{self.index}")

    def report(self) -> dict[str, Any]:
        return {
            "phase": self.phase.name,
            "success": self.success,
            "total_steps": self.total_steps,
            "index": self.index,
            "body_frames": len(self.frames),
            "seed_path": str(self.seed_path),
            "gates": {"surface": self._surface_waiter.report()},
            "log": list(self.log),
            "recorded_frames": len(self._recorded),
        }


def play_flag_12(
    env: Any,
    *,
    policy: Flag12Policy | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    start_lives: int | None = None,
) -> dict[str, Any]:
    """Play the 1-2 flag body from the current env until 1-3 control / fail."""
    pol = policy or Flag12Policy(action_size=int(env.action_space.shape[0]))
    pol.reset()
    snap0 = read_snapshot(env.get_ram())
    lives0 = start_lives if start_lives is not None else snap0.lives
    max_x = snap0.player_x
    outcome = "timeout"
    end_frame = 0
    snap = snap0
    for frame in range(1, max_frames + 1):
        snap = read_snapshot(env.get_ram(), frame=frame - 1)
        if is_1_3_control(snap):
            outcome = "success"
            pol.phase = Phase.DONE
            end_frame = frame - 1
            break
        tick = pol.step(snap)
        env.step(tick.action)
        end_frame = frame
        snap = read_snapshot(env.get_ram(), frame=frame)
        max_x = max(max_x, snap.player_x)
        if snap.lives < lives0 or snap.dying:
            outcome = "death"
            pol.phase = Phase.FAILED
            break
        if int(snap.world) != 0:
            outcome = "warp"
            pol.phase = Phase.FAILED
            break
        if is_1_3_control(snap):
            outcome = "success"
            pol.phase = Phase.DONE
            pol.log.append({"event": "1_3_control", "step": pol.total_steps})
            break
        if pol.done and not pol.success:
            outcome = "failed"
            break
    return {
        "ok": outcome == "success" and is_1_3_control(snap),
        "outcome": outcome,
        "frames": end_frame,
        "max_x": max_x,
        "snap": {
            "world": int(snap.world),
            "dash_level": int(snap.dash_level),
            "x": int(snap.player_x),
            "y": int(snap.player_y),
            "ps": int(snap.player_state),
            "timer": int(snap.timer),
            "area_pointer": int(snap.area_pointer),
        },
        "policy": pol.report(),
        "recorded": pol.recorded_frames(),
        "control_x_max": CONTROL_X_MAX,
        "on_ground": bool(player_on_ground(env.get_ram())),
    }
