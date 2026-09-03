"""Level 7 chapter factories and Survival ``SpineHop`` rows.

The public surface has three chapters.  Internal stage names provide precise
handoffs without exposing room-level ``--through`` targets.

``MEASURED_POST_L6_EXIT.verified`` is True and the Survival bait stage is
green.  Pond drain on the spine stays fail-closed: missing a natural-whistle
drain from the L6 leave (bead ``rr-8t4.4``).  Do not wire the recon
``ADDR_WHISTLE`` poke onto the spine.  Entry room ``0x79`` is observed.
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
    make_survival_bait_purchase_controller,
)
from zelda_i.level7.graph import ledger_notes
from zelda_i.level7.overworld import POST_L6_TO_BAIT_HOPS
from zelda_i.level7.path import (
    EntryNorthDoorController,
    HungryGoriyaGateController,
    Level7PathController,
    RedCandlePickupController,
    Room6AEastController,
    Room6BEastController,
    Room69EastController,
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
        "natural-whistle drain from the L6 leave (rr-8t4.4)",
    )


def make_entry_first_door_controller() -> Level7PathController:
    return EntryNorthDoorController()


def make_room69_east_controller() -> Level7PathController:
    """0x69 goriya kill-clear → OPEN east doorway to live $EB=0x6A (2/2)."""
    return Room69EastController()


def make_room6a_east_controller() -> Level7PathController:
    """0x6A dark KEESE room west mouth → OPEN east doorway to live 0x6B (2/2)."""
    return Room6AEastController()


def make_room6b_east_controller() -> Level7PathController:
    """0x6B GORIYA_HINT west mouth → OPEN east doorway to live 0x6C (2/2).

    Assumes the six 0x6B goriya 0x05 are cleared upstream (recon fixture or a
    prior kill-clear stage).  Not yet on the executable chapter chain — the
    0x6B goriya-clear and Hungry Goriya rooms past it are still fail-closed.
    """
    return Room6BEastController()


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
    survival: bool = False,
) -> tuple[Stage, ...]:
    """Fresh post-L6 OW -> Bait -> Whistle pond -> observed L7 entry.

    ``survival=True`` swaps the fail-closed natural Bait buy for the disclosed
    ``SurvivalBaitPurchaseController`` (one ``ADDR_FOOD`` write); Clean keeps the
    natural buy.
    """
    post = make_post_l6_overworld_controller(handoff=handoff, hops=post_l6_hops)
    bait = (
        make_survival_bait_purchase_controller(plan=bait_plan)
        if survival
        else make_bait_purchase_controller(plan=bait_plan)
    )
    pond = make_pond_entry_controller()
    return (
        ("level7_post_l6_overworld", post, post.max_frames),
        ("level7_bait_purchase", bait, bait.max_frames),
        ("level7_pond_drain_entry", pond, pond.max_frames),
    )


def level7_red_candle_chapter_stages() -> tuple[Stage, ...]:
    """Fresh entry first-door -> Hungry Goriya -> tip stairs -> natural Red Candle."""
    return (
        _stage("level7_entry_first_door", make_entry_first_door_controller),
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
    survival: bool = False,
) -> tuple[SpineHop, ...]:
    """Build fresh L7 chapter rows.  Defaults stay non-executable.

    ``survival=True`` (the ``continue_level7_spine`` seam) swaps the Bait stage
    for the disclosed ``ADDR_FOOD`` fixture.  ``level7_entry_first_door`` is
    a live ``0x79`` north walker; pond drain, Hungry Goriya, tip stairs,
    candle, and bosses stay fail-closed.
    """

    def _entry_stages() -> tuple[Stage, ...]:
        return level7_entry_chapter_stages(
            handoff=handoff,
            post_l6_hops=post_l6_hops,
            bait_plan=bait_plan,
            survival=survival,
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
    "make_entry_first_door_controller",
    "make_entry_to_goriya_controller",
    "make_forced_digdogger_controller",
    "make_level7_shard_leave_controller",
    "make_pond_entry_controller",
    "make_post_l6_overworld_controller",
    "make_red_candle_controller",
    "make_room69_east_controller",
    "make_room6a_east_controller",
    "make_room6b_east_controller",
    "make_tip_stairs_controller",
    "UNMEASURED_HANDOFF",
]
