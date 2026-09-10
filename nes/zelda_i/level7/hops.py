"""Level 7 chapter factories and Survival ``SpineHop`` rows.

The public surface has three chapters plus dedicated ``level7-bait-shop``.
Internal stage names provide precise handoffs without exposing room-level
``--through`` targets.

``MEASURED_POST_L6_EXIT.verified`` is True. Survival ``--through level7``
is spine-green from power-on (Recorder warp to pond ``0x42``, drain into
entry ``0x79``, disclosed Food poke). Clean ``survival=False`` /
``allow_pokes=False`` never writes ``ADDR_FOOD``. Natural 60R bait shop is
``rr-8t4.4`` (``--through level7-bait-shop``). Do not wire the recon
``ADDR_WHISTLE`` poke.
"""

from __future__ import annotations

from typing import Callable

from zelda_i.level7.aquamentus import make_level7_aquamentus_heart_controller
from zelda_i.level7.cellar import make_nose_cellar_cross_controller
from zelda_i.level7.digdogger import make_level7_forced_digdogger_controller
from zelda_i.level7.dungeon import (
    MEASURED_POST_L7_EXIT,
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
from zelda_i.level7.hungry import make_level7_hungry_goriya_controller
from zelda_i.level7.overworld import (
    POST_L6_TO_WARP_HOPS,
    WARP_ISLAND_SCREEN,
    WARP_JOIN_TO_POND_HOPS,
    WARP_JOIN_TO_SHOP_HOPS,
    WARP_LAUNCH_SCREEN,
    OverworldToLevel7PondController,
    on_level7_bait_shop_hyp,
)
from zelda_i.level7.pond import make_pond_drain_controller
from zelda_i.level7.warp import make_recorder_warp_controller
from zelda_i.dungeon.bomb_wall import BombWallController
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS
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
    EntryNorthDoorController,
    Level7PathController,
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
    Room69WestBombController,
)
from zelda_i.level7.pre_boss import make_room29_east_bomb_controller as make_room29_east_bomb_controller_impl
from zelda_i.level7.shard import (
    make_level7_shard_leave_controller as make_level7_shard_leave_controller_impl,
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
# rw_full_route_t2/t3: post-L6 → warp → pond 0x42 is 4913f end to end;
# the join walk itself is well under 5k.  Budget matches the probe's cap.
POND_APPROACH_MAX_FRAMES = 30_000
ControllerFactory = Callable[[], Level7PathController]


def make_pond_entry_controller() -> Level7PathController:
    """Pause-select Whistle on OW 0x42, drain, enter play 0x79."""
    return make_pond_drain_controller()


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


def make_room69_west_bomb_controller() -> Level7PathController:
    """0x69 kill-clear goriyas -> west BOMB wall -> live 0x68 (KEESE_TRAPS).

    Live census: the five 0x69 goriyas read HP=0 for a few frames while they
    materialize, then go live at HP=80 and throw boomerangs.  A naive
    approach-and-place (no clear) reaches the stand fine but the goriyas
    interrupt the bomb placement (``bomb_not_consumed``).  Clear them first,
    then bomb via ``L7_ROOM69_WEST_APPROACH`` (mirrors ``east_route_step``'s
    detour around the centre-row diamond blocks).
    """
    return Room69WestBombController()


def make_room18_north_bomb_controller() -> BombWallController:
    """0x18 MAP north BOMB wall → live 0x08 HIDDEN_RUPEES (2/2).

    Stand (120,93) face UP.  Needs bombs + bomb on B.  Recon-wired only.
    """
    return BombWallController(
        wall=L7_ROOM18_NORTH_BOMB, level=7, select_item=B_SLOT_BOMBS
    )


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
        select_item=B_SLOT_BOMBS,
    )


def make_room09_down_controller() -> Level7PathController:
    """0x09 GORIYA_POST_RUPEE kill-clear → south shutter to live 0x19 (2/2).

    Dead: south is OPEN on spawn.  Recon-wired only.
    """
    return Room09DownController()


def make_room1a_candle_controller() -> Level7PathController:
    """0x1A kill-clear, 0x68 UP, stairs to cellar 0x4A, natural Red Candle (2/2).

    ADDR_CANDLE 0→2 by walking onto the pad.  Wired as
    ``level7_red_candle_pickup``.
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
        select_item=B_SLOT_BOMBS,
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
        select_item=B_SLOT_BOMBS,
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
        select_item=B_SLOT_BOMBS,
    )


def make_room0d_clear_controller() -> Room0DClearController:
    """0x0D TIP_OF_NOSE kill 5 wallmasters.  2/2 room_all_dead.  Recon-wired only."""
    return Room0DClearController()


def make_entry_to_goriya_controller() -> Level7PathController:
    """Live 0x28 feed (pause-select Bait slot 6) then UP to MAP 0x18 (2/2)."""
    return make_level7_hungry_goriya_controller()


def make_tip_stairs_controller() -> Level7PathController:
    """Live 0x0D walk-on of cellar 0x7B (rr-8t4.3, fixture-live 2/2).

    RIGHT push of the 0x68 at (192,144) reveals the staircase at the
    (208,96) cell, then the ring walk (west off the door row, UP the
    x=32 column, east along the y=96 row) steps onto it. Dest is RAM:
    mode 9, screen 0x7B. ``position_writes`` stays 0.
    """
    return make_stairs0d_controller()


def make_red_candle_controller() -> Level7PathController:
    """Live 0x1A push + cellar 0x4A natural Red Candle (ADDR_CANDLE 0→2)."""
    return Room1ACandleController()


def make_forced_digdogger_controller() -> Level7PathController:
    """Live 0x1C leftover-relative whistle-shrink; dest is RAM 0x0C (2/2).

    Pause-select recorder B-slot 5 (cycle past Red Candle=4). No
    ``ADDR_SELECTED_ITEM`` poke. Blow until type ``0x38``→``0x18``.
    """
    return make_level7_forced_digdogger_controller()


def make_room29_east_bomb_controller() -> BombWallController:
    """0x29 PRE_BOSS east BOMB wall → live 0x2A (probe geometry)."""
    return make_room29_east_bomb_controller_impl()


def make_aquamentus_heart_controller() -> Level7PathController:
    """Live 0x2A kill plus the natural heart container (rr-8t4.3, 2/2).

    ``20260904_W3``/``W4`` on the walk-on lineage (cleared 0x0D pin ->
    stairs -> cellar 0x7B -> 0x29 -> bomb-E), HC 3 -> 4 both runs, 398
    controller frames each, ``position_writes=0``. Combat is the shared
    ``level1.finish.Level1AquamentusController`` (ALIGN/FACE/ATTACK/DODGE,
    ``tank_hits=True``) aliased onto live ``$EB=0x2A``, type ``0x3D``; only
    the pickup is L7 code, because the container is not at L1's fixed
    ``(192,141)`` cell — it was collected at ``(136,141)``
    (``scratch/probe_l7_2a_heart.py``, ``20260904_H1``).
    ``route_eligible`` stays false: the lineage is a recon pin, not Survival.
    """
    return make_level7_aquamentus_heart_controller()


def make_level7_shard_leave_controller() -> Level7PathController:
    """Live 0x2A east shutter -> 0x2B shard -> idled fanfare -> OW (2/2).

    ``20260904_W3``/``W4``: shard taken south-around the diamond floor, the
    fanfare idled (never walked), OW leftover ``0x42`` ``(96,93)`` mode 5,
    TF ``0x40``, HC 4, hearts full, ``position_writes=0``.

    The fixture-lineage leftover is TF ``0x40`` (pin starts at TF 0) and is
    **not** the Survival packet. ``MEASURED_POST_L7_EXIT`` is filled from
    power-on ``--through level7`` 2/2, not from this factory.
    """
    return make_level7_shard_leave_controller_impl()


def _stage(name: str, factory: ControllerFactory) -> Stage:
    controller = factory()
    return (name, controller, controller.max_frames)


def level7_entry_chapter_stages(
    *,
    handoff: OverworldHandoff = UNMEASURED_HANDOFF,
    post_l6_hops: tuple[ScreenHop, ...] = POST_L6_TO_WARP_HOPS,
    warp_launch: int = WARP_LAUNCH_SCREEN,
    warp_target: int = WARP_ISLAND_SCREEN,
    join_hops: tuple[ScreenHop, ...] = WARP_JOIN_TO_POND_HOPS,
    bait_plan: BaitPurchasePlan = UNVERIFIED_BAIT_PLAN,
    survival: bool = False,
) -> tuple[Stage, ...]:
    """Post-L6 OW -> Recorder warp -> pond approach -> Bait -> L7 entry.

    The post-L6 pocket is mountain-locked, so the walk stops on the warp
    launch screen ``0x24`` and ``level7.warp.RecorderWarpController`` blows the
    owned Recorder until the cycle lands on the L4 island door ``0x45``; the
    join hops rejoin the already-green pond chain at ``0x55``
    (``LEVEL7_ROUTE.md`` H1).  ``survival=True`` swaps the fail-closed natural
    Bait buy for the disclosed ``SurvivalBaitPurchaseController`` (one
    ``ADDR_FOOD`` write); Clean keeps the natural buy.
    """
    post = make_post_l6_overworld_controller(
        handoff=handoff, hops=post_l6_hops, dest_screen=warp_launch
    )
    warp = make_recorder_warp_controller(
        target_screen=warp_target, launch_screen=warp_launch
    )
    approach = OverworldToLevel7PondController(
        hops=join_hops, max_frames=POND_APPROACH_MAX_FRAMES
    )
    bait = (
        make_survival_bait_purchase_controller(plan=bait_plan)
        if survival
        else make_bait_purchase_controller(plan=bait_plan)
    )
    pond = make_pond_entry_controller()
    return (
        ("level7_post_l6_overworld", post, post.max_frames),
        ("level7_recorder_warp", warp, warp.max_frames),
        ("level7_pond_approach", approach, approach.max_frames),
        ("level7_bait_purchase", bait, bait.max_frames),
        ("level7_pond_drain_entry", pond, pond.max_frames),
    )


def level7_bait_shop_chapter_stages(
    *,
    handoff: OverworldHandoff = UNMEASURED_HANDOFF,
    post_l6_hops: tuple[ScreenHop, ...] = POST_L6_TO_WARP_HOPS,
    warp_launch: int = WARP_LAUNCH_SCREEN,
    warp_target: int = WARP_ISLAND_SCREEN,
    join_hops: tuple[ScreenHop, ...] = WARP_JOIN_TO_SHOP_HOPS,
) -> tuple[Stage, ...]:
    """Post-L6 OW -> Recorder warp -> peel at 0x54 north to shop 0x34.

    Dedicated ``--through level7-bait-shop`` (rr-8t4.4). No Food write.
    Clean ``survival=False`` keeps ``NaturalBaitPurchaseController``. Survival
    ``level7-entry`` still uses the disclosed Food fixture until ``rr-8t4.4``.
    """
    post = make_post_l6_overworld_controller(
        handoff=handoff, hops=post_l6_hops, dest_screen=warp_launch
    )
    warp = make_recorder_warp_controller(
        target_screen=warp_target, launch_screen=warp_launch
    )
    shop = OverworldToLevel7PondController(
        hops=join_hops, max_frames=POND_APPROACH_MAX_FRAMES
    )
    return (
        ("level7_post_l6_overworld", post, post.max_frames),
        ("level7_recorder_warp", warp, warp.max_frames),
        ("level7_shop_approach", shop, shop.max_frames),
    )


def level7_red_candle_chapter_stages() -> tuple[Stage, ...]:
    """0x79 first door → west candle mainline → Hungry feed → MAP bombs → 0x4A.

    Tip-of-nose stairs is near-boss (complete chapter), not between Hungry
    and Candle.
    """
    return (
        _stage("level7_entry_first_door", make_entry_first_door_controller),
        _stage("level7_room69_west_bomb", make_room69_west_bomb_controller),
        _stage("level7_room68_north", make_room68_north_controller),
        _stage("level7_room58_east", make_room58_east_controller),
        _stage("level7_room59_up", make_room59_up_controller),
        _stage("level7_room49_up", make_room49_up_controller),
        _stage("level7_room39_left", make_room39_left_controller),
        _stage("level7_room38_up", make_room38_up_controller),
        _stage("level7_entry_to_hungry_goriya", make_entry_to_goriya_controller),
        _stage("level7_room18_north_bomb", make_room18_north_bomb_controller),
        _stage("level7_room08_east_bomb", make_room08_east_bomb_controller),
        _stage("level7_room09_down", make_room09_down_controller),
        _stage("level7_room19_east_bomb", make_room19_east_bomb_controller),
        _stage("level7_red_candle_pickup", make_red_candle_controller),
    )


def level7_complete_chapter_stages() -> tuple[Stage, ...]:
    """Candle return → Digdogger → 0x0D stairs → cellar → 0x2A heart → shard.

    Interior stage factories stay ``route_eligible=false``; the Survival
    leave packet is filled separately from power-on.
    """
    return (
        _stage("level7_room4a_return", make_room4a_return_controller),
        _stage("level7_room1a_east_bomb", make_room1a_east_bomb_controller),
        _stage("level7_room1b_key_east", make_room1b_key_east_controller),
        _stage("level7_forced_digdogger", make_forced_digdogger_controller),
        _stage("level7_room0c_east_bomb", make_room0c_east_bomb_controller),
        _stage("level7_room0d_clear", make_room0d_clear_controller),
        _stage("level7_tip_of_nose_stairs", make_tip_stairs_controller),
        _stage("level7_nose_cellar_cross", make_nose_cellar_cross_controller),
        _stage("level7_room29_east_bomb", make_room29_east_bomb_controller),
        _stage("level7_aquamentus_heart", make_aquamentus_heart_controller),
        _stage("level7_shard_and_settled_leave", make_level7_shard_leave_controller),
    )


def _bait_shop_success(env):
    def success(snap: ZeldaSnapshot, **_) -> bool:
        ram = env.get_ram()
        return on_level7_bait_shop_hyp(snap) and int(read_u8(ram, ADDR_FOOD)) == 0

    return success


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
    post_l6_hops: tuple[ScreenHop, ...] = POST_L6_TO_WARP_HOPS,
    bait_plan: BaitPurchasePlan = UNVERIFIED_BAIT_PLAN,
    survival: bool = False,
) -> tuple[SpineHop, ...]:
    """Build fresh L7 chapter rows.  Defaults stay ``route_eligible=false``.

    ``survival=True`` (the ``continue_level7_spine`` seam) swaps the Bait stage
    for the disclosed ``ADDR_FOOD`` fixture. ``survival=False`` keeps the
    natural buy and writes no Food. Interior chapter factories stay
    ``route_eligible=false``.
    """

    def _shop_stages() -> tuple[Stage, ...]:
        return level7_bait_shop_chapter_stages(
            handoff=handoff, post_l6_hops=post_l6_hops
        )

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
            "level7-bait-shop",
            "level7_bait_shop",
            _shop_stages,
            _bait_shop_success(env),
            dedicated=True,
        ),
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
    "level7_bait_shop_chapter_stages",
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
    "make_room29_east_bomb_controller",
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
    "MEASURED_POST_L7_EXIT",
    "UNMEASURED_HANDOFF",
]
