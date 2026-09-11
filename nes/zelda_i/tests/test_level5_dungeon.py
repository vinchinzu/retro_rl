"""Unit tests for Level 5 leftover walks that would burn again."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level5.dungeon import (
    GIBDO_OBJECT_TYPE,
    LEVEL_5,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_66_SPEC,
    ROOM_77_SPEC,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_POLS_77,
    Level5PolsVoiceController,
    level5_room_66_cleared,
    level5_room_77_key_success,
)
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
    ZeldaObject,
    ZeldaSnapshot,
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


def test_pols_voice_controller_is_solid() -> None:
    ctrl = Level5PolsVoiceController(spec=ROOM_77_SPEC)
    # (153, 189) is under the right 2x3 cluster on the south wall - MUST be solid
    assert ctrl._is_solid(153, 189) is True
    # (200, 189) is in the east column on the south wall - open
    assert ctrl._is_solid(200, 189) is False
    # x in 152..184 at y >= 109 is strictly blocked
    for x in (152, 160, 176, 184):
        for y in (109, 141, 165, 189):
            assert ctrl._is_solid(x, y) is True


def test_pols_voice_controller_occupancy_miss_and_stand() -> None:
    ctrl = Level5PolsVoiceController(spec=ROOM_77_SPEC)
    # Pre-block DOWN, LEFT, RIGHT so only UP is considered from (120, 141)
    ctrl.blocked_cells.add((120, 145))  # DOWN
    ctrl.blocked_cells.add((116, 141))  # LEFT
    ctrl.blocked_cells.add((124, 141))  # RIGHT

    ram = _ram(
        room=ROOM_L5_POLS_77,
        x=120,
        y=141,
        enemies=1,
        enemy_type=POLS_VOICE_OBJECT_TYPE,
        hp=160,
    )
    ram[ADDR_LINK_X + 1] = 200
    ram[ADDR_LINK_Y + 1] = 93
    snap = read_snapshot(ram)

    # Frame 1: Link is at (120, 141), moves UP
    act1 = ctrl.step(snap)
    assert act1.action == nes_action("UP")
    assert ctrl.last_dir == "UP"

    # Frames 2-4: Link remains at (120, 141)
    ctrl.step(snap)
    ctrl.step(snap)
    ctrl.step(snap)

    # Frame 5: 4th stuck frame -> occupancy miss records cell ahead (120, 137)
    act5 = ctrl.step(snap)
    assert (120, 137) in ctrl.blocked_cells
    assert ctrl.misses == 1
    # Now all directions are blocked -> controller stands on no path
    assert act5.reason == "stand_no_path"
    assert act5.action == nes_idle_action()

