"""Unit tests for Level 5 leftover walks that would burn again."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import GenericDungeonRoomController
from zelda_i.level5.dungeon import (
    GIBDO_OBJECT_TYPE,
    LEVEL_5,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_66_SPEC,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_POLS_77,
    level5_room_66_cleared,
    level5_room_77_key_success,
)
from zelda_i.level5.path import make_room66_controller
from zelda_i.level5.spine import ROOM_66_SPINE_SPEC
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROOM_ALL_DEAD,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    level: int = LEVEL_5,
    room: int = ROOM_L5_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
    doors: int = 0,
    all_dead: int = 0,
    enemy_type: int = GIBDO_OBJECT_TYPE,
    enemies: int = 0,
    hp: int = 112,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    ram[ADDR_CUR_OPENED_DOORS] = doors
    ram[ADDR_ROOM_ALL_DEAD] = all_dead
    for slot in range(1, enemies + 1):
        ram[ADDR_OBJ_TYPE + slot] = enemy_type
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_LINK_X + slot] = 64 + slot * 16
        ram[ADDR_LINK_Y + slot] = 141
    return ram


def test_room_66_combat_uses_occupancy_across_river() -> None:
    """TF suffix leftover (79,165): cardinal patrol never crossed the river."""
    assert ROOM_66_SPEC.combat.occupancy_patrol is True
    assert ROOM_66_SPEC.combat.occupancy_bounds == (16, 216, 77, 205)
    assert ROOM_66_SPEC.combat.contact_backstep == 16
    assert ROOM_66_SPINE_SPEC.combat.contact_backstep == 16


def test_room_66_peels_contact_gibdo_at_leftover() -> None:
    """Death (128,133): peel a north Gibdo; do not greedy-close UP into the body."""
    ram = _ram(room=ROOM_L5_GIBDO_66, x=128, y=133, enemies=1, hp=112)
    ram[ADDR_LINK_X + 1] = 128
    ram[ADDR_LINK_Y + 1] = 125
    act = GenericDungeonRoomController(spec=ROOM_66_SPINE_SPEC).step(
        read_snapshot(ram)
    )
    assert act.reason == "combat_backstep"
    assert act.action == nes_action("DOWN")
    assert act.action != nes_action("UP")
    assert act.action != nes_action("UP", "A")


def test_room_66_leaves_nw_pocket_south() -> None:
    """Timeout (64,120): 1 east Gibdo — DOWN off the pocket, not idle wait."""
    ram = _ram(room=ROOM_L5_GIBDO_66, x=64, y=120, enemies=1, hp=112)
    ram[ADDR_LINK_X + 1] = 192
    ram[ADDR_LINK_Y + 1] = 149
    act = make_room66_controller(spec=ROOM_66_SPINE_SPEC).step(read_snapshot(ram))
    assert act.reason == "66_pocket_south"
    assert act.action == nes_action("DOWN")
    assert act.action != nes_idle_action()


def test_room_66_cleared_predicate() -> None:
    assert level5_room_66_cleared(
        _ram(room=ROOM_L5_GIBDO_66, enemies=0, doors=0x08, all_dead=20)
    )
    assert not level5_room_66_cleared(
        _ram(room=ROOM_L5_GIBDO_66, enemies=3, doors=0x08, all_dead=20, hp=112)
    )
    assert not level5_room_66_cleared(
        _ram(room=ROOM_L5_GIBDO_66, enemies=0, doors=0x00, all_dead=20)
    )


def test_room_77_key_success() -> None:
    assert level5_room_77_key_success(_ram(room=ROOM_L5_POLS_77, keys=1, enemies=0))
    assert not level5_room_77_key_success(_ram(room=ROOM_L5_POLS_77, keys=0, enemies=0))
    assert not level5_room_77_key_success(
        _ram(
            room=ROOM_L5_POLS_77,
            keys=1,
            enemies=2,
            enemy_type=POLS_VOICE_OBJECT_TYPE,
            hp=160,
        )
    )
    assert not level5_room_77_key_success(_ram(room=ROOM_L5_ENTRY, keys=1, enemies=0))
