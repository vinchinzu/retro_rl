"""ZD The Gathering grid arithmetic vs the Q1 catalog and known OW traps."""

from __future__ import annotations

from zelda_i.overworld.gathering import (
    SWORD_MAX,
    pre_l1_bomb_shop_success,
    pre_l1_stages,
)
from zelda_i.overworld.shop_p7 import (
    COAST_TEKTITE_SCREEN,
    PRE_L1_BOMB_HOPS,
    SCREEN_79_BEACH_Y,
    SCREEN_7A_EAST_BAND,
    SCREEN_7E_EAST_BAND,
    SHOP_P7_HOPS,
    SHOP_P7_NOT_ON_WALK,
    SHOP_P7_PRICE,
    SHOP_P7_SCREEN,
    SHOP_P7_TRANSIT_SCREENS,
    make_shop_p7_walk_controller,
    pre_l1_walk_hops,
    shop_p7_arrived,
    shop_p7_screens,
)
from zelda_i.overworld.zd_map import map1_route
from zelda_i.overworld.graph import (
    SCREEN_LABELS,
    SCREEN_START,
    ScreenHop,
    screen_id,
    screen_to_grid,
)
from zelda_i.overworld.locations import (
    CAVE_SHOP_ARROWS,
    OPEN_OPEN,
    location,
)
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.sword_cave import SEGMENT_MAX_FRAMES, SwordCaveController
from zelda_i.ram import CAVE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram


def _shift(screen: int, *, east: int = 0, north: int = 0) -> int:
    col, row = screen_to_grid(screen)
    return screen_id(col + east, row - north)


def test_zd_bomb_shop_grid_is_0x6f_and_map1_paints_row7() -> None:
    """ZD 1.1: right 8, up 1 from start. Map-1.png is that walk, dest shop_p7."""
    assert _shift(SCREEN_START, east=8) == 0x7F
    assert _shift(SCREEN_START, east=8, north=1) == 0x6F
    shop = location("shop_p7")
    assert shop is not None and shop.screen == 0x6F
    assert shop.kind == "shop"
    assert shop.open == "open"
    assert SCREEN_LABELS[0x79] == "rocky_deadend_east_of_78"
    east_of_start = _shift(SCREEN_START, east=2)
    assert east_of_start == 0x79
    assert 0x79 in map1_route().screens


def test_zd_heart_l8_is_down1_left4_from_bomb_shop() -> None:
    """ZD 1.2: from 0x6F, down 1 left 4, bomb the north wall."""
    assert _shift(0x6F, north=-1, east=-4) == 0x7B
    heart = location("heart_l8")
    assert heart is not None
    assert heart.screen == 0x7B
    assert heart.open == "bomb"
    assert heart.vanilla == "heart_container"


def test_zd_heart_m3_is_the_center_rock() -> None:
    heart = location("heart_m3")
    assert heart is not None
    assert heart.screen == 0x2C
    assert heart.open == "bomb"


def test_zd_letter_candle_white_sword_cluster() -> None:
    """NE coast: 100R 0x0F, letter 0x0E, candle shop 0x0C, White Sword 0x0A."""
    rupees = location("rupees_100_p1")
    letter = location("letter")
    candle = location("shop_m1")
    sword = location("white_sword")
    assert rupees is not None and rupees.screen == 0x0F
    assert letter is not None and letter.screen == 0x0E and letter.open == "open"
    assert candle is not None
    assert candle.screen == 0x0C
    assert candle.kind == "shop"
    assert candle.open == "open"
    assert sword is not None and sword.screen == 0x0A and sword.open == "open"


def test_zd_left_two_from_candle_shop_is_lost_hills() -> None:
    """ZD 1.3 'left two then climb' from 0x0C walks into 0x1B. Do not."""
    down = _shift(0x0C, north=-1)
    assert down == 0x1C
    lost_hills = _shift(down, east=-1)
    assert lost_hills == 0x1B
    west_of_hills = _shift(lost_hills, east=-1)
    assert west_of_hills == 0x1A
    assert _shift(west_of_hills, north=1) == 0x0A


def test_zd_burn_heart_and_90r_shield_are_west_of_0x48() -> None:
    heart = location("heart_h5")
    shield = location("shop_g5")
    assert heart is not None and heart.screen == 0x47 and heart.open == "burn"
    assert shield is not None and shield.screen == 0x46 and shield.open == "burn"
    assert _shift(0x48, east=-1) == 0x47
    assert _shift(0x47, east=-1) == 0x46


def test_zd_arrows_and_blue_ring_match_catalog() -> None:
    arrows = location("arrow_shop")
    ring = location("special_shop_e4")
    assert arrows is not None and arrows.screen == 0x4A
    assert ring is not None and ring.screen == 0x34 and ring.open == "armos"
    assert _shift(0x51, east=3, north=2) == 0x34


_SHOP_P7_RAM = {
    "mode": PLAY_MODE,
    "level": 0,
    "screen": SHOP_P7_SCREEN,
    "x": 128,
    "y": 141,
    "sword": 1,
}


def _shop_p7_ram(**fields: int):
    return make_ram(_SHOP_P7_RAM, **fields)


def test_shop_bomb_path_follows_map1() -> None:
    screens = shop_p7_screens()
    painted = map1_route()
    assert screens[0] == SCREEN_START == 0x77
    assert screens[-1] == SHOP_P7_SCREEN == painted.dest == 0x6F
    assert COAST_TEKTITE_SCREEN == 0x7A
    assert screens == painted.screens
    assert 0x7A in screens
    assert 0x7B in screens
    assert 0x6B not in screens
    assert 0x48 not in screens
    assert 0x4A not in screens
    assert 0x79 in screens
    assert screens[:4] == (0x77, 0x78, 0x79, 0x7A)


def test_shop_bomb_path_skips_inland_and_row6_pocket() -> None:
    screens = shop_p7_screens()
    assert 0x79 in screens
    assert SCREEN_LABELS[0x79] == "rocky_deadend_east_of_78"
    assert SHOP_P7_NOT_ON_WALK.isdisjoint(screens)
    assert 0x4A not in screens
    assert 0x5C not in screens
    assert 0x5E not in screens
    assert 0x68 not in screens


def test_shop_bomb_hops_follow_map1_coast_with_79_beach() -> None:
    painted = map1_route()
    assert SHOP_P7_HOPS == PRE_L1_BOMB_HOPS
    assert shop_p7_screens() == painted.screens
    assert tuple(h.target for h in SHOP_P7_HOPS) == tuple(
        h.target for h in painted.hops
    )
    assert tuple(h.direction for h in SHOP_P7_HOPS) == tuple(
        h.direction for h in painted.hops
    )
    assert tuple(h.target for h in SHOP_P7_HOPS) == (
        0x78,
        0x79,
        0x7A,
        0x7B,
        0x7C,
        0x7D,
        0x7E,
        0x7F,
        0x6F,
    )
    # Overlay names the screens. It is not the hop generator: three live
    # lanes replace the painted centre row.
    by_target = {h.target: h for h in SHOP_P7_HOPS}
    assert painted.hops[2].align_y != SCREEN_79_BEACH_Y
    assert by_target[0x7A] == ScreenHop(0x7A, "RIGHT", align_y=SCREEN_79_BEACH_Y)
    assert by_target[0x7B].y_band == SCREEN_7A_EAST_BAND == (133, 141)
    # 0x7D scrolls from every row; 0x7E does not. The dead 133 row is how
    # ``pre_l1_topup_live`` entered 0x7E and died at y=131. The 0x7E band
    # therefore sits on the hop that leaves 0x7D, not only on 0x7E itself.
    assert by_target[0x7E].y_band == SCREEN_7E_EAST_BAND == (137, 145)
    assert by_target[0x7F].y_band == SCREEN_7E_EAST_BAND == (137, 145)
    assert 120 <= int(painted.hops[2].align_y or 0) <= 145


def test_a_spit_south_of_the_7e_band_does_not_walk_north() -> None:
    """The same 0x7D leave as the hunt duck, on the walk's own rung.

    ``pre_l1_shortfall1`` died at (44, 109) after ``spit_duck`` held UP off
    the 137–145 band. The shot was in the water. A spit on the band still
    leaves the row; this one is not on it.
    """
    from retro_harness.nes import nes_action
    from zelda_i.dungeon.behaviors import FIREBALL_TYPE
    from zelda_i.ram import ZeldaObject, ZeldaSnapshot

    hop = next(h for h in SHOP_P7_HOPS if h.target == 0x7E)
    ctl = make_shop_p7_walk_controller()
    act = None
    for i in range(4):
        shot_x = 180 - 8 * i
        shot_y = 200 - 3 * i
        snap = ZeldaSnapshot(
            mode=PLAY_MODE, level=0, screen=0x7D, next_screen=0x7D,
            link_x=72, link_y=138, facing=0x01, sword=1, bombs=0, rupees=9,
            keys=0, health=0x21, heart_partial=0x7F, triforce=0, compass=0,
            dialog_timer=0, colliding_tile=0, room_item_id=0, room_all_dead=0,
            room_obj_count=0, cur_opened_doors=0, open_doorway_mask=0,
            objects=(
                ZeldaObject(
                    slot=10, type_id=FIREBALL_TYPE, x=shot_x, y=shot_y,
                    facing=0x0A, hp=0, state=0x10,
                ),
            ),
        )
        ctl._observe_threats(snap)
        act = ctl._threat_action(snap, hop)
    assert act is not None and act.reason == "spit_duck"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))


def test_7d_exit_band_walks_down_off_the_dead_133_row() -> None:
    """y=131 is outside SCREEN_7E_EAST_BAND; the 0x7D→0x7E hop must DOWN."""
    from zelda_i.overworld.common import align_and_push

    hop = next(h for h in SHOP_P7_HOPS if h.target == 0x7E)
    snap = read_snapshot(make_ram(
        {"mode": PLAY_MODE, "level": 0, "screen": 0x7D, "x": 200, "y": 131, "sword": 1}
    ))
    act = align_and_push(
        snap, direction=hop.direction, reason="hop6", y_band=hop.y_band
    )
    assert hop.y_band == SCREEN_7E_EAST_BAND
    assert act.reason == "band_down"


def test_shop_p7_arrived_only_on_play_6f_with_sword() -> None:
    play = read_snapshot(_shop_p7_ram())
    assert shop_p7_arrived(play)
    # Arrival is not the errand: the spine stop is the 4-pack.
    assert not pre_l1_bomb_shop_success(play)
    assert pre_l1_bomb_shop_success(read_snapshot(_shop_p7_ram(bombs=4)))
    assert not shop_p7_arrived(read_snapshot(_shop_p7_ram(mode=CAVE_MODE)))
    assert not shop_p7_arrived(read_snapshot(_shop_p7_ram(screen=0x7A)))
    assert not shop_p7_arrived(read_snapshot(_shop_p7_ram(sword=0)))
    assert not shop_p7_arrived(read_snapshot(_shop_p7_ram(level=1)))


def test_shop_p7_walk_controller_is_walk_only_with_scoop_knobs() -> None:
    ctl = make_shop_p7_walk_controller()
    assert isinstance(ctl, OverworldPathController)
    assert ctl.hops == SHOP_P7_HOPS
    assert ctl.farm_below_hearts == 0
    assert ctl.need_rupees == 0  # t1 restock-farmed 0x78; price stays 20 for the buy
    assert ctl.scoop_rupees is True
    assert ctl.scoop_bombs is True
    assert SHOP_P7_PRICE == 20
    assert ctl.evade is True
    assert ctl.occupied_lane is True
    assert ctl.require_sword is True
    assert ctl.door_x is None
    assert ctl.door_screen is None
    assert not ctl.require_dungeon
    assert ctl.hunter is not None  # the walk is the rupee farm
    assert ctl.hunter.transit_screens == SHOP_P7_TRANSIT_SCREENS == {0x7B, 0x7D}
    assert ctl.hunter.reopen_on_enter is False  # one pass; see the lap test
    play = read_snapshot(_shop_p7_ram())
    ctl.hunter.done.add(SHOP_P7_SCREEN)  # the wave itself is its own test
    assert ctl._at_stop(play)
    cave = read_snapshot(_shop_p7_ram(mode=CAVE_MODE))
    assert not ctl._at_stop(cave)
    hop1 = SHOP_P7_HOPS[1]
    assert hop1.target == 0x79
    assert hop1.direction == "RIGHT"
    hop2 = SHOP_P7_HOPS[2]
    assert hop2.target == 0x7A
    assert hop2.direction == "RIGHT"
    hop8 = SHOP_P7_HOPS[8]
    assert hop8.target == 0x6F
    assert hop8.direction == "UP"


def test_pre_l1_stages_are_sword_then_walk_then_topup_then_buy() -> None:
    stages = pre_l1_stages()
    assert len(stages) == 4
    sword_name, sword_ctl, sword_max = stages[0]
    walk_name, walk_guard, walk_max = stages[1]
    topup_name, topup_guard, topup_max = stages[2]
    buy_name, buy_ctl, _ = stages[3]
    # The coast hunt and its top-up play on the ROM's next frames near a body.
    from zelda_i.rollout import PolicyGuard

    assert isinstance(walk_guard, PolicyGuard) and isinstance(topup_guard, PolicyGuard)
    assert walk_guard.trigger_radius and topup_guard.trigger_radius
    walk_ctl, topup_ctl = walk_guard.inner, topup_guard.inner
    assert topup_name == "bomb_topup"
    assert topup_ctl.shop_screen == 0x6F and topup_ctl.price == 20
    assert topup_ctl.hunter is not None
    # A lap re-enters a screen it has already fought; a walk must not.
    assert topup_ctl.hunter.reopen_on_enter is True
    assert topup_max >= 8000
    assert "sword" in sword_name
    assert "walk" in walk_name
    assert isinstance(sword_ctl, SwordCaveController)
    assert isinstance(walk_ctl, OverworldPathController)
    assert sword_max == SWORD_MAX == SEGMENT_MAX_FRAMES
    assert walk_ctl.hops == SHOP_P7_HOPS == PRE_L1_BOMB_HOPS
    assert getattr(walk_ctl, "laps", 0) == 0
    assert walk_ctl.hunter.reopen_on_enter is False
    assert walk_max >= 30000
    assert buy_name == "bomb_buy"
    assert buy_ctl.hops == ()
    assert buy_ctl.shop_screen == 0x6F
    assert buy_ctl.price == 20
    assert buy_ctl.farm is None
    assert buy_ctl.cave_x == 48
    assert buy_ctl.cave_y == 77
    from zelda_i.overworld import bomb_shop as inland_bomb_shop

    assert inland_bomb_shop.BOMB_SHOP_SCREEN == 0x4A
    assert inland_bomb_shop.BOMB_SHOP_HOPS != walk_ctl.hops


def test_under_the_price_the_coast_hunt_stays_open() -> None:
    """600f retires a wave with the drop still down. Under 20 the cap is
    the destination budget; at the price it drops back to travel."""
    from zelda_i.overworld.hunt import HUNT_DESTINATION_FRAMES, HUNT_SCREEN_MAX_FRAMES

    ctl = make_shop_p7_walk_controller()
    assert ctl.hunter is not None
    assert ctl._before_play(read_snapshot(_shop_p7_ram(rupees=0))) is None
    assert ctl.hunter.screen_max_frames == HUNT_DESTINATION_FRAMES
    assert ctl._before_play(read_snapshot(_shop_p7_ram(rupees=SHOP_P7_PRICE))) is None
    assert ctl.hunter.screen_max_frames == HUNT_SCREEN_MAX_FRAMES
    assert ctl._before_play(
        read_snapshot(_shop_p7_ram(rupees=SHOP_P7_PRICE + 8))
    ) is None
    assert ctl.hunter.screen_max_frames == HUNT_SCREEN_MAX_FRAMES


def test_arrival_short_stops_so_topup_can_leave() -> None:
    """18R at 0x6F is ``bomb_topup``'s job. Fighting the shop wave until
    ``destination_hunted`` deadlocked 22403f in cave mode (``pre_l1_c3_melee1``)."""
    ctl = make_shop_p7_walk_controller()
    arrived = read_snapshot(_shop_p7_ram(health=0x22, rupees=18))
    assert shop_p7_arrived(arrived)
    assert ctl._at_stop(arrived)
    funded = read_snapshot(_shop_p7_ram(rupees=SHOP_P7_PRICE))
    assert ctl._at_stop(funded)


def test_the_destination_wave_is_not_worth_a_2400_frame_stand_at_one_heart() -> None:
    """``_after_hops`` answers a declining hunt with an idle, and the hunt
    declines for the whole guard branch — so on the destination screen the
    two meet as a stand for ``HUNT_DESTINATION_FRAMES``. ``pre_l1_anyrow1``
    reached 0x6F for the first time and died on frame 593 of that stand, to
    the shop screen's own Zora, with the cave mouth two tiles away."""
    ctl = make_shop_p7_walk_controller()
    guarding = read_snapshot(_shop_p7_ram(health=0x20))  # 1 of 3
    assert guarding.whole_hearts <= ctl.hunter.min_hearts
    assert 0x6F not in ctl.hunter.done
    assert ctl.destination_hunted(guarding) is True
    assert ctl._at_stop(guarding) is True


def test_lapped_walk_is_wired_but_composer_stays_one_pass() -> None:
    ctl = make_shop_p7_walk_controller(laps=1)
    assert ctl.hops == pre_l1_walk_hops(1)
    assert ctl.laps == 1
    assert ctl.hunter.reopen_on_enter is True
    _, walk, _ = pre_l1_stages()[1]
    assert walk.laps == 0
    assert walk.hunter.reopen_on_enter is False
    assert walk.hops == pre_l1_walk_hops(0)


def test_funded_arrival_stops_even_mid_lap() -> None:
    ctl = make_shop_p7_walk_controller(laps=1)
    ctl.hop_index = len(PRE_L1_BOMB_HOPS)
    assert ctl.hop_index < len(ctl.hops)
    snap = read_snapshot(_shop_p7_ram(rupees=SHOP_P7_PRICE))
    assert ctl._at_stop(snap)


def test_unfunded_mid_lap_does_not_stop_on_6f() -> None:
    ctl = make_shop_p7_walk_controller(laps=1)
    ctl.hop_index = len(PRE_L1_BOMB_HOPS)
    ctl.hunter.done.add(SHOP_P7_SCREEN)
    snap = read_snapshot(_shop_p7_ram(rupees=8))
    assert not ctl._at_stop(snap)


def test_shop_p7_catalog_row_is_open_arrows_family() -> None:
    shop = location("shop_p7")
    assert shop is not None
    assert shop.screen == 0x6F
    assert shop.cave_id == CAVE_SHOP_ARROWS
    assert shop.open == OPEN_OPEN
    assert shop.kind == "shop"


def test_the_travelling_walk_spends_the_edge_on_a_lined_up_body() -> None:
    """The walk, not only the hunt, must be able to fire the full-health shot.

    Measured (``scratch/probe_beam.py --phase lane``, tag ``b3``): on the live
    coast walk the shot was up for 807 of 6218 frames and a body stood in a
    9 px lane on 361 of them, but the hunt's own branch aimed on **0** —
    while Link travels, the hop table owns the frame and ``walk_or_swing``
    only presses A at contact range.
    """
    from retro_harness.controls import pressed_nes_buttons
    from zelda_i.dungeon.ids import TEKTITE_BLUE_OBJECT_TYPE
    from zelda_i.ram import (
        ADDR_HEART_PARTIAL,
        ADDR_LINK_X,
        ADDR_LINK_Y,
        ADDR_OBJ_HP,
        ADDR_OBJ_TYPE,
    )

    ram = _shop_p7_ram(screen=0x78, x=60, y=133, health=0x22, bombs=0, rupees=0)
    ram[ADDR_HEART_PARTIAL] = 0xFF
    ram[ADDR_OBJ_TYPE + 1] = TEKTITE_BLUE_OBJECT_TYPE
    ram[ADDR_LINK_X + 1] = 210
    ram[ADDR_LINK_Y + 1] = 133
    ram[ADDR_OBJ_HP + 1] = 1
    snap = read_snapshot(ram)

    walk = make_shop_p7_walk_controller()
    walk.hunter.observe(snap)
    act = walk.hunter.take_beam(snap)
    assert act is not None and act.reason == "beam_78"
    assert "A" in pressed_nes_buttons(list(act.action))
    assert walk.hunter.beam.pressed == 1


def test_the_travelling_shot_is_off_when_the_walk_does_not_hunt() -> None:
    from zelda_i.dungeon.ids import TEKTITE_BLUE_OBJECT_TYPE
    from zelda_i.ram import (
        ADDR_HEART_PARTIAL,
        ADDR_LINK_X,
        ADDR_LINK_Y,
        ADDR_OBJ_HP,
        ADDR_OBJ_TYPE,
    )

    ram = _shop_p7_ram(screen=0x77, x=60, y=133, health=0x22, bombs=0, rupees=0)
    ram[ADDR_HEART_PARTIAL] = 0xFF
    ram[ADDR_OBJ_TYPE + 1] = TEKTITE_BLUE_OBJECT_TYPE
    ram[ADDR_LINK_X + 1] = 210
    ram[ADDR_LINK_Y + 1] = 133
    ram[ADDR_OBJ_HP + 1] = 1
    snap = read_snapshot(ram)

    walk = make_shop_p7_walk_controller()
    walk.hunter = None
    act = walk._do_hop(snap)
    assert not act.reason.startswith("beam_")
