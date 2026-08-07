from __future__ import annotations

import numpy as np

from zelda_i.ram import (
    ADDR_BOOMERANG,
    ADDR_HEALTH,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    PLAY_MODE,
    SCREEN_START,
    capabilities_from_ram,
    is_level1_ready,
    parse_game_state,
    read_snapshot,
)


def test_parse_game_state_menu_by_default() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    state = parse_game_state(ram, frame=0)
    assert state.extras["ram_map_partial"] is False
    assert state.extras["sword"] == 0


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
    caps = capabilities_from_ram(ram)
    assert "wooden_sword" in caps
    assert "boomerang" not in caps
    assert "magical_boomerang" not in caps
    ram[ADDR_BOOMERANG] = 1
    assert "boomerang" in capabilities_from_ram(ram)
    ram[ADDR_MAGIC_BOOMERANG] = 1
    caps_magic = capabilities_from_ram(ram)
    assert "magical_boomerang" in caps_magic
    # Magical supersedes wooden in the capability set.
    assert "boomerang" not in caps_magic
