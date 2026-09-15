"""ZD The Gathering grid arithmetic vs the Q1 catalog and known OW traps."""

from __future__ import annotations

from zelda_i.level8.overworld import LEVEL8_BUSH_HOPS
from zelda_i.overworld.gathering import (
    DEAD_68_EAST_Y141,
    DEAD_6C_EAST_BUSH,
    SHOP_P7_HOPS,
    SHOP_P7_HOPS_LIVE_PREFIX,
    SHOP_P7_PRICE,
    SHOP_P7_SCREEN,
    SOURCE_HYPOTHESIS,
    SWORD_MAX,
    make_shop_p7_walk_controller,
    pre_l1_bomb_shop_success,
    pre_l1_stages,
    shop_p7_arrived,
    shop_p7_screens,
)
from zelda_i.overworld.graph import (
    SCREEN_LABELS,
    SCREEN_START,
    ScreenHop,
    screen_id,
    screen_to_grid,
)
from zelda_i.overworld.locations import (
    CAVE_SHOP_ARROWS,
    HEART_H5_NEAREST_SPINE_HOP_DIR,
    HEART_H5_NEAREST_SPINE_SCREEN,
    HEART_H5_SCREEN,
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


def test_zd_bomb_shop_grid_is_0x6f_but_row7_hits_0x79() -> None:
    """ZD 1.1: right 8, up 1 from start. Dest is shop_p7; the walk is not."""
    assert _shift(SCREEN_START, east=8) == 0x7F
    assert _shift(SCREEN_START, east=8, north=1) == 0x6F
    shop = location("shop_p7")
    assert shop is not None and shop.screen == 0x6F
    assert shop.kind == "shop"
    assert shop.open == "open"
    assert SCREEN_LABELS[0x79] == "rocky_deadend_east_of_78"
    east_of_start = _shift(SCREEN_START, east=2)
    assert east_of_start == 0x79


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
    assert HEART_H5_SCREEN == 0x47
    assert HEART_H5_NEAREST_SPINE_SCREEN == 0x48
    assert HEART_H5_NEAREST_SPINE_HOP_DIR == "LEFT"
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


def test_shop_bomb_path_screens_start_77_end_4a() -> None:
    screens = shop_p7_screens()
    assert screens[0] == SCREEN_START == 0x77
    assert screens[-1] == SHOP_P7_SCREEN == 0x4A
    assert screens == (0x77, 0x78, 0x68, 0x58, 0x59, 0x49, 0x4A)


def test_shop_bomb_path_bypasses_traps_and_dead_corridors() -> None:
    screens = shop_p7_screens()
    assert 0x79 not in screens
    assert SCREEN_LABELS[0x79] == "rocky_deadend_east_of_78"
    assert 0x67 not in screens
    assert 0x1B not in screens  # Lost Hills is later; this hop does not go there
    assert 0x6C not in screens  # row 6 west pocket is bypassed via 0x49 -> 0x4A


def test_shop_bomb_hops_match_level2_prefix() -> None:
    assert SHOP_P7_HOPS[:4] == LEVEL8_BUSH_HOPS[:4]
    assert SHOP_P7_HOPS == (
        ScreenHop(0x78, "RIGHT", align_y=140),
        ScreenHop(0x68, "UP", align_x=48),
        ScreenHop(0x58, "UP", align_x=48),
        ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
        ScreenHop(0x49, "UP", align_x=112),
        ScreenHop(0x4A, "RIGHT", align_y=141),
    )
    assert SOURCE_HYPOTHESIS is True
    assert DEAD_68_EAST_Y141 is True  # 0x68 east is dead
    assert DEAD_6C_EAST_BUSH is True  # 0x6C east is dead


def test_shop_p7_arrived_only_on_play_4a_with_sword() -> None:
    play = read_snapshot(_shop_p7_ram())
    assert shop_p7_arrived(play)
    assert pre_l1_bomb_shop_success(play)
    assert not shop_p7_arrived(read_snapshot(_shop_p7_ram(mode=CAVE_MODE)))
    assert not shop_p7_arrived(read_snapshot(_shop_p7_ram(screen=0x68)))
    assert not shop_p7_arrived(read_snapshot(_shop_p7_ram(sword=0)))
    assert not shop_p7_arrived(read_snapshot(_shop_p7_ram(level=1)))


def test_shop_p7_walk_controller_is_walk_only_with_scoop_knobs() -> None:
    ctl = make_shop_p7_walk_controller()
    assert isinstance(ctl, OverworldPathController)
    assert ctl.hops == SHOP_P7_HOPS
    assert ctl.farm_below_hearts == 0
    assert ctl.need_rupees == 0  # t1 restock-farmed 0x78; price stays 20 for the buy
    assert SHOP_P7_PRICE == 20
    assert ctl.evade is True
    assert ctl.occupied_lane is True
    assert ctl.require_sword is True
    assert ctl.door_x is None
    assert ctl.door_screen is None
    assert not ctl.require_dungeon
    play = read_snapshot(_shop_p7_ram())
    assert ctl._at_stop(play)
    cave = read_snapshot(_shop_p7_ram(mode=CAVE_MODE))
    assert not ctl._at_stop(cave)
    hop4 = SHOP_P7_HOPS[4]
    assert hop4.target == 0x49
    assert hop4.direction == "UP"
    assert hop4.align_x == 112
    hop5 = SHOP_P7_HOPS[5]
    assert hop5.target == 0x4A
    assert hop5.direction == "RIGHT"
    assert hop5.align_y == 141


def test_pre_l1_stages_are_sword_then_walk() -> None:
    stages = pre_l1_stages()
    assert len(stages) == 2
    sword_name, sword_ctl, sword_max = stages[0]
    walk_name, walk_ctl, walk_max = stages[1]
    assert "sword" in sword_name
    assert "shop" in walk_name or "walk" in walk_name
    assert isinstance(sword_ctl, SwordCaveController)
    assert isinstance(walk_ctl, OverworldPathController)
    assert sword_max == SWORD_MAX == SEGMENT_MAX_FRAMES
    assert walk_ctl.hops == SHOP_P7_HOPS
    assert walk_max >= 30000


def test_shop_p7_catalog_row_is_open_arrows_family() -> None:
    shop = location("shop_p7")
    assert shop is not None
    assert shop.screen == 0x6F
    assert shop.cave_id == CAVE_SHOP_ARROWS
    assert shop.open == OPEN_OPEN
    assert shop.kind == "shop"
