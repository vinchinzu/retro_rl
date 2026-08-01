"""Reactive 1-2 secret-warp controller (state-gated, not absolute-frame).

Phase-insensitive: after any natural 1-1 clear, wait for control gates then
play short control-relative fragments (or pure reactive surface). No mid-level
state loads and no absolute-frame stitches from the continuous seed.

Phases:

1. ``wait_surface`` — idle until 1-2 surface control
2. ``surface`` — walk to the entry pipe + enter underground (reactive)
3. ``wait_underground`` — idle until underground control (timer live)
4. ``underground`` — control-relative RLE through secret exit → World 4

Fragments live in ``models/smb_1_2_reactive_fragments.json`` (extracted from
the verified continuous seed relative to control anchors, not absolute frames).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.nes import NES_ACTION_SIZE, nes_idle_action
from smb.paths import GAME_DIR
from smb.policy import expand_nes9_rle
from smb.ram import SmbSnapshot, read_snapshot, segment_1_2_warp_success
from smb.reactive_route import GateWaiter, StateGate
from snes_oneshot.primitives import FrameAction

DEFAULT_FRAGMENTS = GAME_DIR / "models" / "smb_1_2_reactive_fragments.json"

# NES-9 button layouts (indices match retro NES).
_IDLE = [0, 0, 0, 0, 0, 0, 0, 0, 0]
_RIGHT = [0, 0, 0, 0, 0, 0, 0, 1, 0]
_DOWN = [0, 0, 0, 0, 0, 1, 0, 0, 0]

# Surface pipe sits near x=160 on area 41.
SURFACE_PIPE_X = 158
SURFACE_MAX_WAIT = 600
UNDERGROUND_MAX_WAIT = 600
DEFAULT_MAX_FRAMES = 4000


class Phase(Enum):
    WAIT_SURFACE = auto()
    SURFACE = auto()
    WAIT_UNDERGROUND = auto()
    UNDERGROUND = auto()
    DONE = auto()
    FAILED = auto()


def is_surface_control(snap: SmbSnapshot) -> bool:
    """True on controllable 1-2 surface (pre-pipe)."""
    return (
        snap.world == 0
        and snap.level == 1
        and snap.area_pointer == 41
        and snap.oper_mode == 1
        and snap.player_state in (7, 8)
        and 20 <= snap.player_x <= 100
        and snap.player_y >= 160
        and not snap.dying
    )


def is_underground_control(snap: SmbSnapshot) -> bool:
    """True on controllable 1-2 underground start (post-pipe load)."""
    return (
        snap.world == 0
        and snap.level == 2
        and snap.oper_mode == 1
        and snap.player_state in (7, 8)
        and snap.timer > 0
        and snap.player_x < 100
        and not snap.dying
    )


def is_underground_entered(snap: SmbSnapshot) -> bool:
    """Level-id alias flip when the surface pipe finishes (level becomes 2)."""
    return snap.world == 0 and snap.level == 2


def load_reactive_fragments(path: Path | str = DEFAULT_FRAGMENTS) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "underground_from_control" not in data:
        raise ValueError(f"missing underground_from_control in {path}")
    return data


def underground_frames(path: Path | str = DEFAULT_FRAGMENTS) -> list[list[int]]:
    data = load_reactive_fragments(path)
    ug = data["underground_from_control"]
    return expand_nes9_rle({"format": "nes9_rle", "segments": ug["segments"]})


def surface_frames(path: Path | str = DEFAULT_FRAGMENTS) -> list[list[int]]:
    """Optional macro surface fragment (control-relative); prefer reactive."""
    data = load_reactive_fragments(path)
    surf = data.get("surface_from_control")
    if not surf:
        return []
    return expand_nes9_rle({"format": "nes9_rle", "segments": surf["segments"]})


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
class Reactive12Policy:
    """State-gated 1-2 warp policy.

    Call :meth:`step` with the latest :class:`SmbSnapshot` each frame.
    Does not own the env — pure controller.
    """

    fragments_path: Path = DEFAULT_FRAGMENTS
    action_size: int = NES_ACTION_SIZE
    use_reactive_surface: bool = True
    max_surface_wait: int = SURFACE_MAX_WAIT
    max_underground_wait: int = UNDERGROUND_MAX_WAIT
    phase: Phase = field(default=Phase.WAIT_SURFACE, init=False)
    frames_in_phase: int = field(default=0, init=False)
    ug_index: int = field(default=0, init=False)
    surface_index: int = field(default=0, init=False)
    total_steps: int = field(default=0, init=False)
    _ug_frames: list[list[int]] = field(default_factory=list, repr=False)
    _surf_frames: list[list[int]] = field(default_factory=list, repr=False)
    _recorded: list[list[int]] = field(default_factory=list, repr=False)
    _surface_waiter: GateWaiter = field(init=False, repr=False)
    _underground_waiter: GateWaiter = field(init=False, repr=False)
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._ug_frames = underground_frames(self.fragments_path)
        if not self.use_reactive_surface:
            self._surf_frames = surface_frames(self.fragments_path)
        self._surface_waiter = GateWaiter(
            StateGate(
                "1-2:surface_control",
                is_surface_control,
                "controllable 1-2 surface after the natural 1-1 predecessor",
            ),
            max_frames=self.max_surface_wait,
        )
        self._underground_waiter = GateWaiter(
            StateGate(
                "1-2:underground_control",
                is_underground_control,
                "controllable 1-2 underground after the entry pipe transition",
            ),
            max_frames=self.max_underground_wait,
        )

    def reset(self) -> None:
        self.phase = Phase.WAIT_SURFACE
        self.frames_in_phase = 0
        self.ug_index = 0
        self.surface_index = 0
        self.total_steps = 0
        self._recorded.clear()
        self.log.clear()
        self._surface_waiter.reset()
        self._underground_waiter.reset()

    @property
    def done(self) -> bool:
        return self.phase in (Phase.DONE, Phase.FAILED)

    @property
    def success(self) -> bool:
        return self.phase is Phase.DONE

    def recorded_frames(self) -> list[list[int]]:
        return [list(f) for f in self._recorded]

    def step(self, snap: SmbSnapshot) -> FrameAction:
        """Produce the next action given the *current* pre-step snapshot."""
        if self.phase is Phase.DONE:
            return FrameAction(_idle(self.action_size), "1_2_done")
        if self.phase is Phase.FAILED:
            return FrameAction(_idle(self.action_size), "1_2_failed")

        if snap.dying or (self.total_steps > 0 and snap.oper_mode == 3):
            self.phase = Phase.FAILED
            self.log.append({"event": "death_or_gameover", "step": self.total_steps})
            return FrameAction(_idle(self.action_size), "1_2_fail_death")

        # World 4 = world index 3 (checked again after step in play_reactive_12).
        if snap.world == 3:
            self.phase = Phase.DONE
            self.log.append({"event": "world4", "step": self.total_steps})
            return FrameAction(_idle(self.action_size), "1_2_world4")

        action, reason = self._phase_action(snap)
        self._recorded.append([int(x) for x in action[:9]])
        self.total_steps += 1
        self.frames_in_phase += 1
        return FrameAction(action, reason)

    def _phase_action(self, snap: SmbSnapshot) -> tuple[np.ndarray, str]:
        if self.phase is Phase.WAIT_SURFACE:
            if self._surface_waiter.observe(snap):
                self.log.append(
                    {
                        "event": "surface_control",
                        "wait": self._surface_waiter.frames_waited,
                        "x": snap.player_x,
                    }
                )
                self.phase = Phase.SURFACE
                self.frames_in_phase = 0
                return self._surface_action(snap)
            if self._surface_waiter.timed_out:
                self.phase = Phase.FAILED
                self.log.append({"event": "timeout_surface_wait"})
                return _idle(self.action_size), "1_2_timeout_surface_wait"
            return _idle(self.action_size), "1_2_wait_surface"

        if self.phase is Phase.SURFACE:
            if is_underground_entered(snap):
                self.log.append(
                    {
                        "event": "underground_enter",
                        "frames": self.frames_in_phase,
                        "x": snap.player_x,
                    }
                )
                self.phase = Phase.WAIT_UNDERGROUND
                self.frames_in_phase = 0
                return _idle(self.action_size), "1_2_wait_ug"
            return self._surface_action(snap)

        if self.phase is Phase.WAIT_UNDERGROUND:
            if self._underground_waiter.observe(snap):
                self.log.append(
                    {
                        "event": "underground_control",
                        "wait": self._underground_waiter.frames_waited,
                        "x": snap.player_x,
                        "timer": snap.timer,
                    }
                )
                self.phase = Phase.UNDERGROUND
                self.frames_in_phase = 0
                self.ug_index = 0
                return self._underground_action(snap)
            if self._underground_waiter.timed_out:
                self.phase = Phase.FAILED
                self.log.append({"event": "timeout_ug_wait"})
                return _idle(self.action_size), "1_2_timeout_ug_wait"
            return _idle(self.action_size), "1_2_wait_ug"

        if self.phase is Phase.UNDERGROUND:
            if snap.world == 3:
                self.phase = Phase.DONE
                self.log.append({"event": "world4", "step": self.total_steps})
                return _idle(self.action_size), "1_2_world4"
            if self.ug_index >= len(self._ug_frames):
                self.phase = Phase.FAILED
                self.log.append({"event": "ug_exhausted", "x": snap.player_x})
                return _idle(self.action_size), "1_2_ug_exhausted"
            return self._underground_action(snap)

        return _idle(self.action_size), "1_2_idle"

    def _surface_action(self, snap: SmbSnapshot) -> tuple[np.ndarray, str]:
        if self.use_reactive_surface:
            if snap.player_x < SURFACE_PIPE_X:
                return _nes9(_RIGHT, self.action_size), "1_2_surf_right"
            # On / past pipe: hold DOWN (idle also works once seated; DOWN is safer).
            return _nes9(_DOWN, self.action_size), "1_2_surf_down"
        if self.surface_index >= len(self._surf_frames):
            # Fragment ended without enter — hold DOWN.
            return _nes9(_DOWN, self.action_size), "1_2_surf_down_hold"
        buttons = self._surf_frames[self.surface_index]
        self.surface_index += 1
        return _nes9(buttons, self.action_size), f"1_2_surf_macro_{self.surface_index}"

    def _underground_action(self, snap: SmbSnapshot) -> tuple[np.ndarray, str]:
        buttons = self._ug_frames[self.ug_index]
        self.ug_index += 1
        return _nes9(buttons, self.action_size), f"1_2_ug_{self.ug_index}"

    def report(self) -> dict[str, Any]:
        return {
            "phase": self.phase.name,
            "success": self.success,
            "total_steps": self.total_steps,
            "ug_index": self.ug_index,
            "ug_total": len(self._ug_frames),
            "use_reactive_surface": self.use_reactive_surface,
            "fragments_path": str(self.fragments_path),
            "gates": {
                "surface": self._surface_waiter.report(),
                "underground": self._underground_waiter.report(),
            },
            "log": list(self.log),
            "recorded_frames": len(self._recorded),
        }


def play_reactive_12(
    env: Any,
    *,
    policy: Reactive12Policy | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    start_lives: int | None = None,
) -> dict[str, Any]:
    """Run reactive 1-2 from the current env state until World 4 / fail / timeout.

    Expects the env to already be in (or transitioning into) 1-2 after a
    natural 1-1 clear. Returns a report including recorded controller frames.
    """
    pol = policy or Reactive12Policy(
        action_size=int(env.action_space.shape[0]),
    )
    pol.reset()
    snap0 = read_snapshot(env.get_ram())
    lives0 = start_lives if start_lives is not None else snap0.lives
    max_x = snap0.player_x
    outcome = "timeout"
    end_frame = 0
    obs = None

    for frame in range(1, max_frames + 1):
        snap = read_snapshot(env.get_ram(), frame=frame - 1)
        tick = pol.step(snap)
        obs, *_ = env.step(tick.action)
        end_frame = frame
        ram = env.get_ram()
        snap = read_snapshot(ram, frame=frame)
        max_x = max(max_x, snap.player_x)

        if snap.lives < lives0 or snap.dying:
            outcome = "death"
            pol.phase = Phase.FAILED
            break
        if segment_1_2_warp_success(ram, start_lives=lives0):
            outcome = "success"
            pol.phase = Phase.DONE
            pol.log.append({"event": "world4", "step": pol.total_steps})
            break
        if pol.phase is Phase.FAILED:
            outcome = "failed"
            break

    final = read_snapshot(env.get_ram(), frame=end_frame)
    return {
        "success": outcome == "success",
        "outcome": outcome,
        "frames": end_frame,
        "max_player_x": max_x,
        "start_lives": lives0,
        "final": {
            "player_x": final.player_x,
            "player_y": final.player_y,
            "world": final.world,
            "level": final.level,
            "level_id": final.level_id,
            "lives": final.lives,
            "player_state": final.player_state,
            "area_pointer": final.area_pointer,
            "timer": final.timer,
        },
        "policy": pol.report(),
        "recorded": pol.recorded_frames(),
        "last_obs": obs,
    }
