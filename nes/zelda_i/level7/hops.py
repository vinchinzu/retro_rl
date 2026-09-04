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

from zelda_i.level7.cellar import make_nose_cellar_cross_controller
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
from zelda_i.dungeon.bomb_wall import BombWallController
from zelda_i.level7.path import (
    L7_ROOM08_EAST_APPROACH,
    L7_ROOM08_EAST_BOMB,
    L7_ROOM18_NORTH_BOMB,
    L7_ROOM19_EAST_APPROACH,
    L7_ROOM19_EAST_BOMB,
    L7_ROOM1A_EAST_APPROACH,
    L7_ROOM1A_EAST_BOMB,
    L7_ROOM0C_EAST_APPROACH,
    L7_ROOM0C_EAST_BOMB,
    L7_ROOM69_WEST_BOMB,
    EntryNorthDoorController,
    HungryGoriyaGateController,
    Level7PathController,
    RedCandlePickupController,
    Room6AEastController,
    Room6BEastController,
    Room6BNorthController,
    Room6CEastController,
    Room09DownController,
    Room1ACandleController,
    Room4AReturnController,
    Room1BKeyEastController,
    Room0DClearController,
    Room58EastController,
    Room58NorthController,
    Room38UpController,
    Room39LeftController,
    Room49UpController,
    Room59UpController,
    Room68DownController,
    Room68NorthController,
    Room69EastController,
    unverified_path_controller,
)
from zelda_i.level7.stairs0d import make_stairs0d_controller
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


def make_room6b_north_controller() -> Level7PathController:
    """0x6B west mouth → OPEN north notch (x~118) to live 0x5B (2/2).

    Dead-end spur (OLD_MAN_NOSE, bubble 0x40 + 0x50).  Goriyas assumed
    cleared upstream.  Recon-wired only.
    """
    return Room6BNorthController()


def make_room6c_east_controller() -> Level7PathController:
    """0x6C (DIGDOGGER_1) west mouth → east door to live 0x6D STALFOS_KEY (2/2).

    0x6D is a dead-end (stalfos 0x2a + small_key 0x19).  Recon-wired only.
    """
    return Room6CEastController()


def make_room68_north_controller() -> Level7PathController:
    """0x68 (KEESE_TRAPS) → OPEN north door to live 0x58 DODONGOS_UPGRADE (2/2).

    Reached via make_room69_west_bomb_controller.  Recon-wired only.
    """
    return Room68NorthController()


def make_room58_east_controller() -> Level7PathController:
    """0x58 (DODONGOS_UPGRADE) → OPEN east door to live 0x59 GORIYA_COMPASS (2/2).

    3x invulnerable 0x31 roamers are dodged.  Recon-wired only.
    """
    return Room58EastController()


def make_room68_down_controller() -> Level7PathController:
    """0x68 (KEESE_TRAPS) → OPEN south door to live 0x78 ROPES_KEY (2/2).

    Dead-end (ropes 0x28 + floor key 0x19).  Recon-wired only.
    """
    return Room68DownController()


def make_room58_north_controller() -> Level7PathController:
    """0x58 (DODONGOS_UPGRADE) → KEY north door to live 0x48 BOMB_UPGRADE (2/2).

    Dead-end old-man 100-rupee bomb-capacity room.  Do not write max_bombs.
    Recon-wired only.
    """
    return Room58NorthController()


def make_room59_up_controller() -> Level7PathController:
    """0x59 (GORIYA_COMPASS) west mouth: kill-clear goriya 0x05/0x06 -> perimeter
    waypoint micro around the central mass -> UP door to live 0x49 GORIYA_BUBBLE
    (2/2).  Recon-wired only.
    """
    return Room59UpController()


def make_room49_up_controller() -> Level7PathController:
    """0x49 (GORIYA_BUBBLE) south mouth: kill-clear goriya 0x05 -> UP across
    the water moat at x=120 (Stepladder) to live 0x39 DIGDOGGER_2 (2/2).
    Requires ADDR_LADDER=1 on the recon fixture.  Recon-wired only.
    """
    return Room49UpController()


def make_room39_left_controller() -> Level7PathController:
    """0x39 (DIGDOGGER_2) south mouth → OPEN west door to live 0x38
    GORIYA_PRE_HUNGRY (2/2).  Skips the Digdogger fight.  Recon-wired only.
    """
    return Room39LeftController()


def make_room38_up_controller() -> Level7PathController:
    """0x38 (GORIYA_PRE_HUNGRY) east mouth: kill-clear, rise east pocket
    x=208, KEY-UP to live 0x28 HUNGRY_GORIYA (2/2).  Recon-wired only.
    """
    return Room38UpController()


def make_room69_west_bomb_controller() -> BombWallController:
    """0x69 west BOMB wall → live 0x68 (source GORIYA_BOMB_HUB → KEESE_TRAPS).

    The candle-path branch after the Stalfos-key dead-end.  Needs bombs +
    bomb selected on B.  Recon-wired only (interior of 0x68 unobserved).
    """
    return BombWallController(wall=L7_ROOM69_WEST_BOMB, level=7)


def make_room18_north_bomb_controller() -> BombWallController:
    """0x18 MAP north BOMB wall → live 0x08 HIDDEN_RUPEES (2/2).

    Stand (120,93) face UP.  Needs bombs + bomb on B.  Recon-wired only.
    """
    return BombWallController(wall=L7_ROOM18_NORTH_BOMB, level=7)


def make_room08_east_bomb_controller() -> BombWallController:
    """0x08 HIDDEN_RUPEES east BOMB wall → live 0x09 GORIYA_POST_RUPEE (2/2).

    South-band around the diamond cross, stand (208,141) face RIGHT.
    Recon-wired only.
    """
    return BombWallController(
        wall=L7_ROOM08_EAST_BOMB,
        level=7,
        approach_waypoints=L7_ROOM08_EAST_APPROACH,
        approach_tol=4,
    )


def make_room09_down_controller() -> Level7PathController:
    """0x09 GORIYA_POST_RUPEE kill-clear → south shutter to live 0x19 (2/2).

    Dead: south is OPEN on spawn.  Recon-wired only.
    """
    return Room09DownController()


def make_room1a_candle_controller() -> Level7PathController:
    """0x1A kill-clear, 0x68 UP, stairs to cellar 0x4A, natural Red Candle (2/2).

    ADDR_CANDLE 0→2 by walking onto the pad.  Recon-wired only.  Chapter
    ``RedCandlePickupController`` stays fail-closed.
    """
    return Room1ACandleController()


def make_room4a_return_controller() -> Level7PathController:
    """0x4A west-ladder stairs return → live play 0x1A CANDLE_PUSH (2/2).

    Dead: walk off the candle pad at y=141 (tile 243).  Recon-wired only.
    """
    return Room4AReturnController()


def make_room1b_key_east_controller() -> Level7PathController:
    """0x1B GORIYA_PRE_DIG KEY-east → live 0x1C FORCED_DIGDOGGER (2/2).

    y=141 RIGHT, natural key 3→2.  Recon-wired only.
    """
    return Room1BKeyEastController()


def make_room19_east_bomb_controller() -> BombWallController:
    """0x19 WEST_LOCK_SKIP east BOMB wall → live 0x1A CANDLE_PUSH (2/2).

    South-around the diamond floor to stand (208,141) face RIGHT.
    Recon-wired only.
    """
    return BombWallController(
        wall=L7_ROOM19_EAST_BOMB,
        level=7,
        approach_waypoints=L7_ROOM19_EAST_APPROACH,
        approach_tol=4,
    )


def make_room1a_east_bomb_controller() -> BombWallController:
    """0x1A CANDLE_PUSH east BOMB wall → live 0x1B GORIYA_PRE_DIG (2/2).

    South-around from cellar-return leftover (96,157) to (208,141) face RIGHT.
    Recon-wired only.
    """
    return BombWallController(
        wall=L7_ROOM1A_EAST_BOMB,
        level=7,
        approach_waypoints=L7_ROOM1A_EAST_APPROACH,
        approach_tol=4,
    )


def make_room0c_east_bomb_controller() -> BombWallController:
    """0x0C DODONGOS_BOSS_PATH east BOMB wall → live 0x0D TIP_OF_NOSE (2/2).

    East-around the y=141 tile-181 mass.  Recon-wired only.
    """
    return BombWallController(
        wall=L7_ROOM0C_EAST_BOMB,
        level=7,
        approach_waypoints=L7_ROOM0C_EAST_APPROACH,
        approach_tol=4,
    )


def make_room0d_clear_controller() -> Room0DClearController:
    """0x0D TIP_OF_NOSE kill 5 wallmasters.  2/2 room_all_dead.  Recon-wired only."""
    return Room0DClearController()


def make_entry_to_goriya_controller() -> Level7PathController:
    return HungryGoriyaGateController()


def make_tip_stairs_controller() -> Level7PathController:
    """Live 0x0D walk-on of cellar 0x7B (rr-8t4.3, fixture-live 2/2).

    RIGHT push of the 0x68 at (192,144) reveals the staircase at the
    (208,96) cell, then the ring walk (west off the door row, UP the
    x=32 column, east along the y=96 row) steps onto it. Dest is RAM:
    mode 9, screen 0x7B. ``position_writes`` stays 0.
    """
    return make_stairs0d_controller()


def make_red_candle_controller() -> Level7PathController:
    return RedCandlePickupController()


def make_forced_digdogger_controller() -> Level7PathController:
    """Fail-closed until the live post-Candle walk and forced Digdogger census.

    Graph hyp (source ids, not RAM): return CANDLE_PUSH, bomb-east
    GORIYA_PRE_DIG, key-east FORCED_DIGDOGGER (must kill). Whistle shrinks
    type ``0x38`` → ``0x18``. Recipe: ``level5.whistle_path.select_b_item_menu``
    want=5 (``level5.boss_path.WHISTLE_B_SLOT``, recorder), then 12×B as in
    ``level5.boss_path.fight_digdogger``. Do not poke ``ADDR_SELECTED_ITEM``.
    Do not invent a live ``$EB`` for this room.
    """
    return unverified_path_controller(
        "level7_forced_digdogger",
        "live post-Candle route and forced Digdogger room census "
        "(Whistle B-slot 5 shrink; no invented $EB)",
    )


def make_aquamentus_heart_controller() -> Level7PathController:
    """Fail-closed until live boss room, natural defeat, and one HC pickup.

    Reuse ``zelda_i.level1.finish.Level1AquamentusController`` combat
    (ALIGN/FACE/ATTACK/DODGE/COLLECT_HEART). Do not copy a second engine.
    Skip L1 ROUTE_ENTRY (0x45 waypoints / enter UP): L7 graph enters from
    the west (PRE_BOSS bomb-east). Parameterize room_id once live ``$EB``
    is observed — do not assume L1 ``0x35``. Remeasure stance and heart
    tile; verify type ``0x3D`` + fireballs ``0x55``. Sword-only
    (Survival ``tank_hits=True``).
    """
    return unverified_path_controller(
        "level7_aquamentus_heart",
        "live boss room, natural defeat, and one heart-container pickup "
        "(reuse Level1AquamentusController; no invented $EB)",
    )


def make_level7_shard_leave_controller() -> Level7PathController:
    """Fail-closed until live shard room and settled post-fanfare OW leftover.

    Graph hyp: heart then east TRIFORCE, then idle through fanfare (same
    shape as ``Level6ExitController`` — do not walk the warp). Fill
    MEASURED_POST_L7_EXIT from that leftover via
    ``overworld.stitch.handoff_from_ram`` using the MEASURED_POST_L6_EXIT
    field list. Screen/x/y stay None until measured. ``verified`` stays
    False. Do not invent the leave screen. L8 keeps
    ``PostLevel7Handoff.verified=False`` until this packet is real.
    """
    return unverified_path_controller(
        "level7_shard_and_settled_leave",
        "live shard room and exact settled post-fanfare overworld handoff "
        "(MEASURED_POST_L7_EXIT screen still None; verified stays False)",
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
    """Fresh Red Candle boundary -> bosses -> heart -> shard -> settled leave.

    All three factories are fail-closed blockers. Do not mark route_eligible.
    Public leftover contract is in ``level7.dungeon.LEVEL7_COMPLETE_STOP``
    and ``docs/tasks/l7c-prep-2026-09-03.md``.
    """
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
    "make_nose_cellar_cross_controller",
    "make_pond_entry_controller",
    "make_post_l6_overworld_controller",
    "make_red_candle_controller",
    "make_room69_east_controller",
    "make_room69_west_bomb_controller",
    "make_room18_north_bomb_controller",
    "make_room08_east_bomb_controller",
    "make_room09_down_controller",
    "make_room19_east_bomb_controller",
    "make_room1a_east_bomb_controller",
    "make_room0c_east_bomb_controller",
    "make_room0d_clear_controller",
    "make_room1a_candle_controller",
    "make_room4a_return_controller",
    "make_room1b_key_east_controller",
    "make_room6a_east_controller",
    "make_room6b_east_controller",
    "make_room6b_north_controller",
    "make_room6c_east_controller",
    "make_room58_east_controller",
    "make_room58_north_controller",
    "make_room38_up_controller",
    "make_room39_left_controller",
    "make_room49_up_controller",
    "make_room59_up_controller",
    "make_room68_down_controller",
    "make_room68_north_controller",
    "make_tip_stairs_controller",
    "UNMEASURED_HANDOFF",
]
