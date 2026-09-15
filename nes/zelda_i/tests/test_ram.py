from __future__ import annotations

import numpy as np

from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOOMERANG,
    ADDR_BOW,
    ADDR_HEALTH,
    ADDR_HELP_DROP_COUNT,
    ADDR_HELP_DROP_VALUE,
    ADDR_LINK_IFRAMES,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    ADDR_WORLD_KILL_COUNT,
    CAVE_MODE,
    PASSAGE_MODE,
    PLAY_MODE,
    SCREEN_START,
    ZeldaSnapshot,
    capabilities_from_ram,
    full_health_byte,
    is_level1_ready,
    parse_game_state,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram


def test_mode_constants() -> None:
    assert PLAY_MODE == 5
    assert PASSAGE_MODE == 9
    assert CAVE_MODE == 11


def test_parse_game_state_menu_by_default() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    state = parse_game_state(ram, frame=0)
    assert state.extras["ram_map_partial"] is False
    assert state.extras["sword"] == 0


def test_hearts_full_is_lo_eq_hi_not_nibble_f() -> None:
    """$066F low nibble is whole hearts. Full is lo==hi (0x22=3/3), never 0xF."""
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_HEALTH] = 0x22
    snap = read_snapshot(ram)
    assert snap.heart_containers == 3
    assert snap.filled_hearts == 2
    assert snap.health_is_full is True
    assert full_health_byte(0x20) == 0x22

    ram[ADDR_HEALTH] = 0x21
    snap = read_snapshot(ram)
    assert snap.filled_hearts == 1
    assert snap.health_is_full is False

    ram[ADDR_HEALTH] = 0x2F
    snap = read_snapshot(ram)
    assert snap.filled_hearts == 0xF
    assert snap.health_is_full is False
    ram[ADDR_HEALTH] = 0x0F
    assert read_snapshot(ram).health_is_full is False


def test_is_level1_ready_requires_play_mode_and_health() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    assert is_level1_ready(ram) is False
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_HEALTH] = 0x22
    assert is_level1_ready(ram) is True
    assert is_level1_ready(ram, obs_mean=10.0) is False


def test_snapshot_and_capabilities() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_SCREEN] = SCREEN_START
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 141
    ram[ADDR_HEALTH] = 0x22
    ram[ADDR_SWORD] = 1
    snap = read_snapshot(ram)
    assert snap.overworld is True
    assert snap.has_sword is True
    assert snap.screen_col == 7
    assert snap.screen_row == 7
    assert snap.boomerang == 0
    assert snap.magical_boomerang == 0
    assert snap.bow == 0
    assert snap.arrows == 0
    ram[ADDR_BOW] = 1
    ram[ADDR_ARROWS] = 1
    snap_bow = read_snapshot(ram)
    assert snap_bow.bow == 1
    assert snap_bow.arrows == 1
    ram[ADDR_BOW] = 0
    ram[ADDR_ARROWS] = 0
    caps = capabilities_from_ram(ram)
    assert "wooden_sword" in caps
    assert "boomerang" not in caps
    assert "magical_boomerang" not in caps
    ram[ADDR_BOOMERANG] = 1
    assert "boomerang" in capabilities_from_ram(ram)
    assert read_snapshot(ram).boomerang == 1
    ram[ADDR_MAGIC_BOOMERANG] = 1
    caps_magic = capabilities_from_ram(ram)
    assert "magical_boomerang" in caps_magic
    assert read_snapshot(ram).magical_boomerang == 1
    # Magical supersedes wooden in the capability set.
    assert "boomerang" not in caps_magic
    assert snap.world_kill_count == 0
    assert snap.help_drop_count == 0
    assert snap.help_drop_value == 0


def test_snapshot_kill_counters_default_and_from_ram() -> None:
    """New fields default to 0; read_snapshot pulls $0627 / $50 / $51 / $04F0."""
    snap = ZeldaSnapshot(
        mode=PLAY_MODE,
        level=0,
        screen=SCREEN_START,
        next_screen=SCREEN_START,
        link_x=120,
        link_y=141,
        facing=0,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=0x22,
        triforce=0,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
    )
    assert snap.world_kill_count == 0
    assert snap.help_drop_count == 0
    assert snap.help_drop_value == 0
    assert snap.link_iframes == 0

    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_WORLD_KILL_COUNT] = 16
    ram[ADDR_HELP_DROP_COUNT] = 10
    ram[ADDR_HELP_DROP_VALUE] = 1
    ram[ADDR_LINK_IFRAMES] = 24
    snap = read_snapshot(ram)
    assert snap.world_kill_count == 16
    assert snap.help_drop_count == 10
    assert snap.help_drop_value == 1
    assert snap.link_iframes == 24

    ram_h = make_ram(
        {"mode": PLAY_MODE, "health": 0x22},
        world_kill=14,
        help_count=9,
        help_value=0,
    )
    snap_h = read_snapshot(ram_h)
    assert snap_h.world_kill_count == 14
    assert snap_h.help_drop_count == 9
    assert snap_h.help_drop_value == 0
