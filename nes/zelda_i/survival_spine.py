"""Continuous Survival spine: one emulator session from power-on.

No mid-run state loads, no seam cards, no clip concat. The tape is whatever
this session actually walked. Stop at the first failed stage.

Clean M5 stays on ``run_level1_complete`` without ``--infinite-life``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from zelda_i.chain import (
    ControllerStageResult,
    run_controller_stage,
    run_natural_to_milestone,
)
from zelda_i.level1_finish import LEVEL1_TRIFORCE_BIT, level1_triforce_stages
from zelda_i.level2_overworld import (
    SEGMENT_MAX_FRAMES as L2_NAV_MAX_FRAMES,
    SETTLE_MAX_FRAMES,
    OverworldToLevel2Controller,
    PostTriforceSettleController,
)
from zelda_i.menus import BOOT_FILE_SLOT, BOOT_QUEST
from zelda_i.ram import PLAY_MODE, read_snapshot

BOOT_POLICY = {
    "file_slot": BOOT_FILE_SLOT,
    "quest": BOOT_QUEST,
    "playthrough": "first",
    "file_menu_select": False,
}

Through = Literal["level1", "level2"]

SPINE_THROUGH: tuple[Through, ...] = ("level1", "level2")


def level2_entry_stages():
    """After L1 TF: idle the fanfare, then walk the Moon door and enter L2."""
    return (
        ("settle_l1_tf", PostTriforceSettleController(), SETTLE_MAX_FRAMES),
        (
            "enter_level2",
            OverworldToLevel2Controller(door_path=True, require_dungeon=True),
            L2_NAV_MAX_FRAMES,
        ),
    )


@dataclass
class SpineRun:
    """One continuous power-on session."""

    through: str
    success: bool
    boot_frames: int
    stages: list[ControllerStageResult] = field(default_factory=list)
    prefix: Any = None
    end_frame: int = 0
    failed_stage: str | None = None
    obs: Any = None

    def report(self) -> dict[str, Any]:
        return {
            "ok": self.success,
            "through": self.through,
            "continuous_emulator_session": True,
            "tape_kind": "continuous_survival_spine",
            "mid_run_state_load": False,
            "seamed": False,
            "status_claim": False,
            "boot_policy": dict(BOOT_POLICY),
            "boot_frames": self.boot_frames,
            "end_frame": self.end_frame,
            "failed_stage": self.failed_stage,
            "prefix": self.prefix.report() if self.prefix is not None else None,
            "stages": [stage.report() for stage in self.stages],
        }


def validate_l5_endpoint(report: dict[str, object]) -> None:
    """Accept only a continuous L5 TF stop (no stitch manifest)."""
    if not report.get("continuous_emulator_session"):
        raise ValueError("L5 endpoint must be a continuous emulator session")
    if report.get("seamed") or report.get("tape_kind") == "state_seamed_viewing_compose":
        raise ValueError("seamed L5 tapes are not a spine endpoint")
    final = report.get("final")
    if not isinstance(final, dict):
        raise ValueError("Level 5 report has no final snapshot")
    if not report.get("ok"):
        raise ValueError("Level 5 report is not successful")
    if int(final.get("level", -1)) != 5 or int(final.get("screen", -1)) != 0x14:
        raise ValueError("Level 5 report does not end in the Triforce room (0x14)")
    if int(final.get("triforce", 0)) & 0x10 == 0:
        raise ValueError("Level 5 report does not have Triforce bit 0x10")
    assist = report.get("assist")
    if not isinstance(assist, dict):
        raise ValueError("Level 5 report is missing Survival telemetry")
    if int(assist.get("progression_writes", -1)) != 0:
        raise ValueError("Level 5 report has progression writes")
    if int(assist.get("capacity_writes", -1)) != 0:
        raise ValueError("Level 5 report has capacity writes")


def run_survival_spine(
    env,
    obs: Any,
    *,
    assist: Any,
    on_frame=None,
    room_timer=None,
    through: Through = "level1",
) -> SpineRun:
    """Power-on → requested dungeon stop. One env. No state reload."""
    if through not in SPINE_THROUGH:
        raise ValueError(f"unknown spine stop {through!r}; wired: {SPINE_THROUGH}")
    if assist is None:
        raise ValueError("Survival spine requires UnlimitedHealthAssist")

    prefix = run_natural_to_milestone(
        env,
        milestone="clear53",
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
        first_playthrough=True,
    )
    run = SpineRun(
        through=through,
        success=bool(prefix.success),
        boot_frames=prefix.boot_frames,
        prefix=prefix,
        end_frame=prefix.end_frame,
        obs=prefix.obs,
        failed_stage=None if prefix.success else "prefix_clear53",
    )
    if not run.success:
        return run

    for name, controller, max_frames in level1_triforce_stages(
        natural_entry=True,
        survival=True,
    ):
        obs, stage = run_controller_stage(
            env,
            run.obs,
            name=name,
            controller=controller,
            max_frames=max_frames,
            room_timer=room_timer,
            assist=assist,
            on_frame=on_frame,
            frame_base=run.end_frame,
        )
        run.obs = obs
        run.stages.append(stage)
        run.end_frame = stage.end_frame
        if not stage.success:
            run.success = False
            run.failed_stage = name
            return run

    snap = read_snapshot(env.get_ram())
    run.success = bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    if not run.success:
        run.failed_stage = "triforce_bit"
        return run
    if through == "level1":
        return run

    for name, controller, max_frames in level2_entry_stages():
        obs, stage = run_controller_stage(
            env,
            run.obs,
            name=name,
            controller=controller,
            max_frames=max_frames,
            room_timer=room_timer,
            assist=assist,
            on_frame=on_frame,
            frame_base=run.end_frame,
        )
        run.obs = obs
        run.stages.append(stage)
        run.end_frame = stage.end_frame
        if not stage.success:
            run.success = False
            run.failed_stage = name
            return run

    snap = read_snapshot(env.get_ram())
    run.success = (
        snap.level == 2
        and snap.mode == PLAY_MODE
        and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    )
    if not run.success:
        run.failed_stage = "level2_entry"
    return run
