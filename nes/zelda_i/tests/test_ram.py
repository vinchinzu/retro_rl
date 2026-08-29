from __future__ import annotations

import numpy as np

from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOOMERANG,
    ADDR_BOW,
    ADDR_HEALTH,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    CAVE_MODE,
    PASSAGE_MODE,
    PLAY_MODE,
    SCREEN_START,
    capabilities_from_ram,
    full_health_byte,
    is_level1_ready,
    parse_game_state,
    read_snapshot,
)


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
