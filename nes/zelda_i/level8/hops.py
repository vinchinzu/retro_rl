"""Named Level 8 chapter factories and Survival ``SpineHop`` rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.level8.dungeon import (
    ENTRY_TO_MAGIC_KEY_SPEC,
    MAGIC_KEY_TO_SHARD_SPEC,
    UNOBSERVED_LEVEL8_CLEAR,
    UNOBSERVED_LEVEL8_TOPOLOGY,
    Level8ChapterSpec,
    Level8ClearEndpoint,
    Level8Topology,
    level8_clear_stop,
    level8_entry_stop,
    level8_magic_key_stop,
)
from zelda_i.level8.entry import (
    BURN_MAX_FRAMES,
    SELECT_MAX_FRAMES,
    UNMEASURED_POST_L7_HANDOFF,
    UNVERIFIED_BUSH_BURN_TARGET,
    BushBurnTarget,
    PostLevel7Handoff,
    make_burn_level8_bush_controller,
    make_post_l7_to_bush_controller,
    make_select_red_candle_controller,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.ram import ADDR_CANDLE, ADDR_MAGIC_KEY, ZeldaSnapshot, read_u8
from zelda_i.spine.hops import SpineHop


@dataclass
class UnobservedLevel8ChapterController:
    """Explicit placeholder: hypotheses cannot execute as route chapters."""

    spec: Level8ChapterSpec
    max_frames: int | None = None
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_frames is None:
            self.max_frames = self.spec.max_frames

    def step(self, _snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.failed = True
        reason = f"{self.spec.chapter_id}_not_live_observed"
        if not self.notes:
            self.notes.append(reason)
        return FrameAction(nes_idle_action(), reason)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "chapter_id": self.spec.chapter_id,
            "objective": self.spec.objective,
            "evidence": self.spec.evidence,
            "route_eligible": self.spec.route_eligible,
            "notes": list(self.notes),
        }


def make_entry_to_magic_key_controller() -> UnobservedLevel8ChapterController:
    return UnobservedLevel8ChapterController(ENTRY_TO_MAGIC_KEY_SPEC)


def make_magic_key_to_shard_controller() -> UnobservedLevel8ChapterController:
    return UnobservedLevel8ChapterController(MAGIC_KEY_TO_SHARD_SPEC)


def _entry_stages(
    *,
    handoff: PostLevel7Handoff,
    post_l7_hops: tuple[ScreenHop, ...],
    burn_target: BushBurnTarget,
):
    approach = make_post_l7_to_bush_controller(
        handoff=handoff,
        hops=post_l7_hops,
    )
    select = make_select_red_candle_controller()
    burn = make_burn_level8_bush_controller(target=burn_target)
    return (
        ("level8_post_l7_to_bush", approach, approach.max_frames),
        ("level8_select_red_candle", select, SELECT_MAX_FRAMES),
        ("level8_burn_bush_enter", burn, BURN_MAX_FRAMES),
    )


def _magic_key_stages():
    controller = make_entry_to_magic_key_controller()
    return (
        (ENTRY_TO_MAGIC_KEY_SPEC.chapter_id, controller, int(controller.max_frames)),
    )


def _clear_stages():
    controller = make_magic_key_to_shard_controller()
    return (
        (MAGIC_KEY_TO_SHARD_SPEC.chapter_id, controller, int(controller.max_frames)),
    )


def l8_hops(
    env,
    *,
    handoff: PostLevel7Handoff = UNMEASURED_POST_L7_HANDOFF,
    post_l7_hops: tuple[ScreenHop, ...] = (),
    burn_target: BushBurnTarget = UNVERIFIED_BUSH_BURN_TARGET,
    topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY,
    clear_endpoint: Level8ClearEndpoint = UNOBSERVED_LEVEL8_CLEAR,
) -> tuple[SpineHop, ...]:
    """Build fresh L8 rows. Defaults are intentionally non-executable."""

    def entry_ok(snap: ZeldaSnapshot, **_) -> bool:
        return level8_entry_stop(
            snap,
            candle=read_u8(env.get_ram(), ADDR_CANDLE),
            topology=topology,
        )

    def magic_key_ok(snap: ZeldaSnapshot, **_) -> bool:
        return level8_magic_key_stop(
            snap,
            magic_key=read_u8(env.get_ram(), ADDR_MAGIC_KEY),
            topology=topology,
        )

    def clear_ok(snap: ZeldaSnapshot, **_) -> bool:
        return level8_clear_stop(
            snap,
            magic_key=read_u8(env.get_ram(), ADDR_MAGIC_KEY),
            endpoint=clear_endpoint,
        )

    return (
        SpineHop(
            "level8-entry",
            "level8_entry_live",
            lambda: _entry_stages(
                handoff=handoff,
                post_l7_hops=post_l7_hops,
                burn_target=burn_target,
            ),
            entry_ok,
        ),
        SpineHop(
            "level8-magic-key",
            "level8_magic_key_natural",
            _magic_key_stages,
            magic_key_ok,
        ),
        SpineHop(
            "level8",
            "level8_triforce_0x80",
            _clear_stages,
            clear_ok,
        ),
    )


__all__ = [
    "UnobservedLevel8ChapterController",
    "l8_hops",
    "make_entry_to_magic_key_controller",
    "make_magic_key_to_shard_controller",
]
