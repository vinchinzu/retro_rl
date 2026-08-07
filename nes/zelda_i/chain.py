"""Shared live prefix runners for continuous Zelda I segment scripts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zelda_i.level1 import (
    CLEAR_53_MAX_FRAMES,
    CLEAR_63_MAX_FRAMES,
    SEGMENT_MAX_FRAMES as FIRST_KEY_MAX_FRAMES,
)
from zelda_i.level1 import (
    UNLOCK_NORTH_MAX_FRAMES,
    Level1Clear53Controller,
    Level1Clear63Controller,
    Level1FirstKeyController,
    Level1UnlockNorthController,
)
from zelda_i.menus import boot_to_level1_script
from zelda_i.overworld_nav import (
    SEGMENT_MAX_FRAMES as NAV_MAX_FRAMES,
)
from zelda_i.overworld_nav import OverworldToLevel1Controller
from zelda_i.ram import is_level1_ready, read_snapshot
from zelda_i.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX_FRAMES
from zelda_i.sword_cave import SwordCaveController

if TYPE_CHECKING:
    from zelda_i.assist import UnlimitedHealthAssist
    from zelda_i.room_timer import RoomTimer

# Post-step hook: (env, obs, action, global_frame) -> None
FrameCallback = Callable[[Any, Any, Any, int], None]


def _observe_room_timer(
    room_timer: RoomTimer | None,
    env,
    *,
    frame: int,
) -> None:
    """Opt-in hop timing: one observe per emulator frame when a timer is set."""
    if room_timer is not None:
        room_timer.observe(read_snapshot(env.get_ram()), frame=frame)


def _apply_assist(
    assist: UnlimitedHealthAssist | None,
    env,
    *,
    frame: int,
) -> None:
    """Opt-in survival assist after each frame (never default)."""
    if assist is not None:
        assist.apply_env(env, frame=frame)


def _notify_frame(
    on_frame: FrameCallback | None,
    env,
    obs: Any,
    action: Any,
    *,
    frame: int,
) -> None:
    """Opt-in post-step hook (video capture, debug sinks)."""
    if on_frame is not None:
        on_frame(env, obs, action, frame)


def boot_to_ready(
    env,
    *,
    room_timer: RoomTimer | None = None,
    assist: UnlimitedHealthAssist | None = None,
    on_frame: FrameCallback | None = None,
    frame_base: int = 0,
) -> tuple[Any, int]:
    """Drive the power-on menu script to the first playable overworld frame."""
    obs = None
    frame = 0
    for scripted in boot_to_level1_script():
        action = scripted.action
        obs, *_ = env.step(action)
        frame += 1
        gf = frame_base + frame
        _observe_room_timer(room_timer, env, frame=gf)
        _apply_assist(assist, env, frame=gf)
        _notify_frame(on_frame, env, obs, action, frame=gf)
        if is_level1_ready(env.get_ram(), obs_mean=float(obs.mean())):
            return obs, frame
    return obs, frame


def run_natural_to_level1(
    env,
    *,
    require_dungeon: bool = True,
    room_timer: RoomTimer | None = None,
    assist: UnlimitedHealthAssist | None = None,
    on_frame: FrameCallback | None = None,
    frame_base: int = 0,
) -> tuple[
    Any,
    int,
    SwordCaveController,
    OverworldToLevel1Controller,
    int,
]:
    """Run power-on → sword → Level 1 transition in an existing environment.

    Returns ``(obs, boot_frames, sword, nav, end_frame)`` where ``end_frame`` is
    the global emulator frame after this prefix (``frame_base``-relative total).
    """
    obs, boot_frames = boot_to_ready(
        env,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
        frame_base=frame_base,
    )
    global_frame = frame_base + boot_frames
    sword = SwordCaveController()
    for _ in range(SWORD_MAX_FRAMES):
        action = sword.step(read_snapshot(env.get_ram())).action
        obs, *_ = env.step(action)
        global_frame += 1
        _observe_room_timer(room_timer, env, frame=global_frame)
        _apply_assist(assist, env, frame=global_frame)
        _notify_frame(on_frame, env, obs, action, frame=global_frame)
        if sword.success or sword.phase.name == "FAILED":
            break

    # EAST_77 aligns y≈140 then walks right — no fixed post-cave DOWN hold.
    nav = OverworldToLevel1Controller(require_dungeon=require_dungeon)
    if sword.success:
        for _ in range(NAV_MAX_FRAMES):
            action = nav.step(read_snapshot(env.get_ram())).action
            obs, *_ = env.step(action)
            global_frame += 1
            _observe_room_timer(room_timer, env, frame=global_frame)
            _apply_assist(assist, env, frame=global_frame)
            _notify_frame(on_frame, env, obs, action, frame=global_frame)
            if nav.success or nav.phase.name == "FAILED":
                break
    return obs, boot_frames, sword, nav, global_frame


@dataclass
class ControllerStageResult:
    """One reusable controller stage in a live natural-entry chain."""

    name: str
    controller: Any
    max_frames: int
    frames: int = 0
    success: bool = False
    frame_base: int = 0
    end_frame: int = 0

    def report(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "max_frames": self.max_frames,
            "frames": self.frames,
            "success": self.success,
            "controller": self.controller.report(),
        }
        if self.frame_base or self.end_frame:
            payload["frame_base"] = self.frame_base
            payload["end_frame"] = self.end_frame
        return payload


@dataclass
class NaturalMilestoneRun:
    """Structured result from power-on through a named Level 1 milestone."""

    milestone: str
    obs: Any
    boot_frames: int
    sword: SwordCaveController
    nav: OverworldToLevel1Controller
    stages: list[ControllerStageResult] = field(default_factory=list)
    success: bool = False
    end_frame: int = 0

    def report(self) -> dict[str, Any]:
        payload = {
            "milestone": self.milestone,
            "success": self.success,
            "boot_frames": self.boot_frames,
            "sword": self.sword.report(),
            "nav": self.nav.report(),
            "stages": [stage.report() for stage in self.stages],
        }
        if self.end_frame:
            payload["end_frame"] = self.end_frame
        return payload


def run_controller_stage(
    env,
    obs: Any,
    *,
    name: str,
    controller: Any,
    max_frames: int,
    room_timer: RoomTimer | None = None,
    assist: UnlimitedHealthAssist | None = None,
    on_frame: FrameCallback | None = None,
    frame_base: int = 0,
) -> tuple[Any, ControllerStageResult]:
    """Run one controller without duplicating the standard emulator loop.

    When ``room_timer`` is provided, each post-step RAM snapshot is fed to
    :class:`~zelda_i.room_timer.RoomTimer` with a continuous global frame index
    starting at ``frame_base + 1``. Controllers are unchanged when the timer is
    omitted (default).

    When ``assist`` is provided (Survival / infinite-life first pass), health is
    refilled after each frame per ``docs/ASSIST_CONTRACT.md``. Default is no
    assist (Clean).

    When ``on_frame`` is provided, it is called after each step with
    ``(env, obs, action, global_frame)`` (e.g. video capture).
    """
    result = ControllerStageResult(
        name=name,
        controller=controller,
        max_frames=max_frames,
        frame_base=frame_base,
        end_frame=frame_base,
    )
    for frame in range(1, max_frames + 1):
        action = controller.step(read_snapshot(env.get_ram())).action
        obs, *_ = env.step(action)
        result.frames = frame
        result.end_frame = frame_base + frame
        _observe_room_timer(room_timer, env, frame=result.end_frame)
        _apply_assist(assist, env, frame=result.end_frame)
        _notify_frame(on_frame, env, obs, action, frame=result.end_frame)
        if controller.success or controller.phase.name == "FAILED":
            break
    result.success = bool(controller.success)
    return obs, result


_MILESTONE_ORDER = ("level1", "first_key", "north", "clear63", "clear53")


def run_natural_to_milestone(
    env,
    *,
    milestone: str = "clear53",
    room_timer: RoomTimer | None = None,
    assist: UnlimitedHealthAssist | None = None,
    on_frame: FrameCallback | None = None,
    frame_base: int = 0,
) -> NaturalMilestoneRun:
    """Compose the natural power-on Level 1 prefix through one milestone."""
    if milestone not in _MILESTONE_ORDER:
        raise ValueError(
            f"unknown milestone {milestone!r}; expected one of {_MILESTONE_ORDER}"
        )

    obs, boot_frames, sword, nav, end_frame = run_natural_to_level1(
        env,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
        frame_base=frame_base,
    )
    run = NaturalMilestoneRun(
        milestone=milestone,
        obs=obs,
        boot_frames=boot_frames,
        sword=sword,
        nav=nav,
        success=sword.success and nav.success,
        end_frame=end_frame,
    )
    if milestone == "level1" or not run.success:
        return run

    stage_specs = (
        ("first_key", Level1FirstKeyController(), FIRST_KEY_MAX_FRAMES),
        ("north", Level1UnlockNorthController(), UNLOCK_NORTH_MAX_FRAMES),
        ("clear63", Level1Clear63Controller(), CLEAR_63_MAX_FRAMES),
        ("clear53", Level1Clear53Controller(), CLEAR_53_MAX_FRAMES),
    )
    for name, controller, max_frames in stage_specs:
        obs, result = run_controller_stage(
            env,
            obs,
            name=name,
            controller=controller,
            max_frames=max_frames,
            room_timer=room_timer,
            assist=assist,
            on_frame=on_frame,
            frame_base=run.end_frame,
        )
        run.obs = obs
        run.stages.append(result)
        run.end_frame = result.end_frame
        run.success = run.success and result.success
        if name == milestone or not result.success:
            break
    return run
