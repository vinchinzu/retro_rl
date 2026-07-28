"""Shared live prefix runners for continuous Zelda I segment scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.nes import nes_action
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


def boot_to_ready(env) -> tuple[Any, int]:
    """Drive the power-on menu script to the first playable overworld frame."""
    obs = None
    frame = 0
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frame += 1
        if is_level1_ready(env.get_ram(), obs_mean=float(obs.mean())):
            return obs, frame
    return obs, frame


def run_natural_to_level1(
    env,
) -> tuple[
    Any,
    int,
    SwordCaveController,
    OverworldToLevel1Controller,
]:
    """Run power-on → sword → Level 1 transition in an existing environment."""
    obs, boot_frames = boot_to_ready(env)
    sword = SwordCaveController()
    for _ in range(SWORD_MAX_FRAMES):
        obs, *_ = env.step(sword.step(read_snapshot(env.get_ram())).action)
        if sword.success or sword.phase.name == "FAILED":
            break

    if sword.success:
        for _ in range(55):
            obs, *_ = env.step(nes_action("DOWN"))

    nav = OverworldToLevel1Controller()
    if sword.success:
        for _ in range(NAV_MAX_FRAMES):
            obs, *_ = env.step(nav.step(read_snapshot(env.get_ram())).action)
            if nav.success or nav.phase.name == "FAILED":
                break
    return obs, boot_frames, sword, nav


@dataclass
class ControllerStageResult:
    """One reusable controller stage in a live natural-entry chain."""

    name: str
    controller: Any
    max_frames: int
    frames: int = 0
    success: bool = False

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_frames": self.max_frames,
            "frames": self.frames,
            "success": self.success,
            "controller": self.controller.report(),
        }


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

    def report(self) -> dict[str, Any]:
        return {
            "milestone": self.milestone,
            "success": self.success,
            "boot_frames": self.boot_frames,
            "sword": self.sword.report(),
            "nav": self.nav.report(),
            "stages": [stage.report() for stage in self.stages],
        }


def run_controller_stage(
    env,
    obs: Any,
    *,
    name: str,
    controller: Any,
    max_frames: int,
) -> tuple[Any, ControllerStageResult]:
    """Run one controller without duplicating the standard emulator loop."""
    result = ControllerStageResult(
        name=name,
        controller=controller,
        max_frames=max_frames,
    )
    for frame in range(1, max_frames + 1):
        obs, *_ = env.step(controller.step(read_snapshot(env.get_ram())).action)
        result.frames = frame
        if controller.success or controller.phase.name == "FAILED":
            break
    result.success = bool(controller.success)
    return obs, result


_MILESTONE_ORDER = ("level1", "first_key", "north", "clear63", "clear53")


def run_natural_to_milestone(
    env,
    *,
    milestone: str = "clear53",
) -> NaturalMilestoneRun:
    """Compose the natural power-on Level 1 prefix through one milestone."""
    if milestone not in _MILESTONE_ORDER:
        raise ValueError(
            f"unknown milestone {milestone!r}; expected one of {_MILESTONE_ORDER}"
        )

    obs, boot_frames, sword, nav = run_natural_to_level1(env)
    run = NaturalMilestoneRun(
        milestone=milestone,
        obs=obs,
        boot_frames=boot_frames,
        sword=sword,
        nav=nav,
        success=sword.success and nav.success,
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
        )
        run.obs = obs
        run.stages.append(result)
        run.success = run.success and result.success
        if name == milestone or not result.success:
            break
    return run
