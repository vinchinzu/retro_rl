"""Level 7 chapter factories and Survival ``SpineHop`` rows.

The public surface has three chapters.  Internal stage names provide precise
handoffs without exposing room-level ``--through`` targets.  The post-L6
handoff stays ``verified=False`` and shop geometry is unmeasured, so every
factory here fails closed.
"""

from __future__ import annotations

from typing import Callable

from zelda_i.level7.dungeon import (
    level7_complete_stop,
    level7_entry_stop,
    level7_red_candle_stop,
)
from zelda_i.level7.entry import (
    UNVERIFIED_BAIT_PLAN,
    BaitPurchasePlan,
    make_bait_purchase_controller,
    make_post_l6_overworld_controller,
)
from zelda_i.level7.graph import ledger_notes
from zelda_i.level7.overworld import POST_L6_TO_BAIT_HOPS
from zelda_i.level7.path import (
    HungryGoriyaGateController,
    Level7PathController,
    RedCandlePickupController,
    unverified_path_controller,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.stitch import UNMEASURED_HANDOFF, OverworldHandoff
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_WHISTLE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.spine.hops import SpineHop

Stage = tuple[str, Level7PathController, int]
ControllerFactory = Callable[[], Level7PathController]


def make_pond_entry_controller() -> Level7PathController:
    return unverified_path_controller(
        "level7_pond_drain_entry",
        "live pond screen, normal Whistle selection/drain, and observed entry room",
    )


def make_entry_to_goriya_controller() -> Level7PathController:
    return HungryGoriyaGateController()


def make_tip_stairs_controller() -> Level7PathController:
    return unverified_path_controller(
        "level7_tip_of_nose_stairs",
        "live tip-of-nose room, push tile, and stairs endpoint",
        notes=ledger_notes(),
    )


def make_red_candle_controller() -> Level7PathController:
    return RedCandlePickupController()


def make_forced_digdogger_controller() -> Level7PathController:
    return unverified_path_controller(
        "level7_forced_digdogger",
        "live post-Candle route and forced Digdogger room census",
    )


def make_aquamentus_heart_controller() -> Level7PathController:
    return unverified_path_controller(
        "level7_aquamentus_heart",
        "live boss room, natural defeat, and one heart-container pickup",
    )


def make_level7_shard_leave_controller() -> Level7PathController:
    return unverified_path_controller(
        "level7_shard_and_settled_leave",
        "live shard room and exact settled post-fanfare overworld handoff",
    )


def _stage(name: str, factory: ControllerFactory) -> Stage:
    controller = factory()
    return (name, controller, controller.max_frames)


def level7_entry_chapter_stages(
    *,
    handoff: OverworldHandoff = UNMEASURED_HANDOFF,
    post_l6_hops: tuple[ScreenHop, ...] = POST_L6_TO_BAIT_HOPS,
    bait_plan: BaitPurchasePlan = UNVERIFIED_BAIT_PLAN,
) -> tuple[Stage, ...]:
    """Fresh post-L6 OW -> Bait -> Whistle pond -> observed L7 entry."""
    post = make_post_l6_overworld_controller(handoff=handoff, hops=post_l6_hops)
    bait = make_bait_purchase_controller(plan=bait_plan)
    pond = make_pond_entry_controller()
    return (
        ("level7_post_l6_overworld", post, post.max_frames),
        ("level7_bait_purchase", bait, bait.max_frames),
        ("level7_pond_drain_entry", pond, pond.max_frames),
    )


def level7_red_candle_chapter_stages() -> tuple[Stage, ...]:
    """Fresh entry -> Hungry Goriya -> tip stairs -> natural Red Candle."""
    return (
        _stage("level7_entry_to_hungry_goriya", make_entry_to_goriya_controller),
        _stage("level7_tip_of_nose_stairs", make_tip_stairs_controller),
        _stage("level7_red_candle_pickup", make_red_candle_controller),
    )


def level7_complete_chapter_stages() -> tuple[Stage, ...]:
    """Fresh Red Candle boundary -> bosses -> heart -> shard -> settled leave."""
    return (
        _stage("level7_forced_digdogger", make_forced_digdogger_controller),
        _stage("level7_aquamentus_heart", make_aquamentus_heart_controller),
        _stage("level7_shard_and_settled_leave", make_level7_shard_leave_controller),
    )


def _entry_success(env):
    def success(snap: ZeldaSnapshot, **_) -> bool:
        ram = env.get_ram()
        return level7_entry_stop(
            snap,
            whistle=read_u8(ram, ADDR_WHISTLE),
            food=read_u8(ram, ADDR_FOOD),
        )

    return success


def _red_candle_success(env):
    def success(snap: ZeldaSnapshot, **_) -> bool:
        ram = env.get_ram()
        return level7_red_candle_stop(
            snap,
            candle=read_u8(ram, ADDR_CANDLE),
            whistle=read_u8(ram, ADDR_WHISTLE),
            food=read_u8(ram, ADDR_FOOD),
        )

    return success


def _complete_success(env, incoming_heart_containers: int):
    def success(snap: ZeldaSnapshot, **_) -> bool:
        ram = env.get_ram()
        return level7_complete_stop(
            snap,
            candle=read_u8(ram, ADDR_CANDLE),
            whistle=read_u8(ram, ADDR_WHISTLE),
            incoming_heart_containers=incoming_heart_containers,
        )

    return success


def l7_hops(
    env,
    *,
    handoff: OverworldHandoff = UNMEASURED_HANDOFF,
    post_l6_hops: tuple[ScreenHop, ...] = POST_L6_TO_BAIT_HOPS,
    bait_plan: BaitPurchasePlan = UNVERIFIED_BAIT_PLAN,
) -> tuple[SpineHop, ...]:
    """Build fresh L7 chapter rows.  Defaults stay non-executable."""

    def _entry_stages() -> tuple[Stage, ...]:
        return level7_entry_chapter_stages(
            handoff=handoff, post_l6_hops=post_l6_hops, bait_plan=bait_plan
        )

    incoming = read_snapshot(env.get_ram())
    return (
        SpineHop(
            "level7-entry",
            "level7_entry",
            _entry_stages,
            _entry_success(env),
        ),
        SpineHop(
            "level7-red-candle",
            "level7_red_candle",
            level7_red_candle_chapter_stages,
            _red_candle_success(env),
        ),
        SpineHop(
            "level7",
            "level7_complete",
            level7_complete_chapter_stages,
            _complete_success(env, incoming.heart_containers),
        ),
    )


__all__ = [
    "l7_hops",
    "level7_complete_chapter_stages",
    "level7_entry_chapter_stages",
    "level7_red_candle_chapter_stages",
    "make_aquamentus_heart_controller",
    "make_bait_purchase_controller",
    "make_entry_to_goriya_controller",
    "make_forced_digdogger_controller",
    "make_level7_shard_leave_controller",
    "make_pond_entry_controller",
    "make_post_l6_overworld_controller",
    "make_red_candle_controller",
    "make_tip_stairs_controller",
    "UNMEASURED_HANDOFF",
]
