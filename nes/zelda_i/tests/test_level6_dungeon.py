"""Unit tests for Level 6 leftover walks that would burn again."""

from __future__ import annotations

import numpy as np
import pytest

from retro_harness.nes import nes_action
from zelda_i.dungeon.engine import DungeonPhase
from zelda_i.dungeon.ids import (
    KEESE_OBJECT_TYPE,
    LIKE_LIKE_OBJECT_TYPE,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level6.clear29 import (
    CLEAR29_COMBAT_Y,
    CLEAR29_HANDOFF_Y,
    Level6Clear29Controller,
    clear29_handoff_ok,
)
from zelda_i.level6.dungeon import (
    CLEAR29_WEST_X,
    LEVEL6_COMPASS_BIT,
    ROOM_29_SPEC,
    ROOM_78_SPEC,
    ROOM_7A_SPEC,
    ROOM_L6_COMPASS,
    ROOM_L6_DARK_29,
    ROOM_L6_EAST_KEY,
    ROOM_L6_ENTRY,
    ROOM_L6_HARD_38,
    ROOM_L6_KEESE,
    ROOM_L6_MAP,
    ROOM_L6_ROD_WIZZ,
    ROOM_L6_WEST_WIZZROBE,
    ROOM_L6_WIZZROBE_28,
    level6_room_09_clear_success,
    level6_room_19_clear_success,
    level6_room_28_clear_success,
    level6_room_38_clear_success,
    level6_room_58_clear_success,
    level6_room_68_compass_success,
    level6_room_78_clear_success,
    level6_room_7a_key_success,
)
from zelda_i.level6.overworld import (
    Level6WestKeyDoorController,
    WIZZROBE_ORANGE_TYPE,
)
from zelda_i.ram import (
    ADDR_COMPASS,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    level: int = 6,
    room: int = ROOM_L6_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
    wizzrobes: int = 0,
    hp: int = 64,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    for slot in range(1, wizzrobes + 1):
        ram[ADDR_OBJ_TYPE + slot] = WIZZROBE_ORANGE_TYPE
        ram[ADDR_OBJ_HP + slot] = hp
    return ram


def test_live_wizzrobes_type_and_hp() -> None:
    snap = read_snapshot(_ram(room=ROOM_L6_EAST_KEY, wizzrobes=5, hp=64))
    assert len(ROOM_7A_SPEC.live_enemies(snap)) == 5
    snap_dead = read_snapshot(_ram(room=ROOM_L6_EAST_KEY, wizzrobes=5, hp=0))
    assert len(ROOM_7A_SPEC.live_enemies(snap_dead)) == 0
    snap78 = read_snapshot(_ram(room=ROOM_L6_WEST_WIZZROBE, wizzrobes=5, hp=64))
    assert len(ROOM_78_SPEC.live_enemies(snap78)) == 5


@pytest.mark.parametrize(
    "success_fn, room, extra, live_type, live_hp",
    [
        (level6_room_7a_key_success, ROOM_L6_EAST_KEY, {"keys": 1}, WIZZROBE_ORANGE_TYPE, 64),
        (level6_room_78_clear_success, ROOM_L6_WEST_WIZZROBE, {}, WIZZROBE_ORANGE_TYPE, 64),
        (level6_room_68_compass_success, ROOM_L6_COMPASS, {"compass": LEVEL6_COMPASS_BIT}, ZOL_OBJECT_TYPE, 64),
        (level6_room_19_clear_success, ROOM_L6_MAP, {}, ZOL_OBJECT_TYPE, 32),
        (level6_room_09_clear_success, ROOM_L6_ROD_WIZZ, {}, WIZZROBE_ORANGE_TYPE, 64),
        (level6_room_58_clear_success, ROOM_L6_KEESE, {}, KEESE_OBJECT_TYPE, 0),
        (level6_room_38_clear_success, ROOM_L6_HARD_38, {}, LIKE_LIKE_OBJECT_TYPE, 64),
        (level6_room_28_clear_success, ROOM_L6_WIZZROBE_28, {}, WIZZROBE_ORANGE_TYPE, 64),
    ],
    ids=["7a", "78", "68", "19", "09", "58", "38", "28"],
)
def test_clear_success(success_fn, room, extra, live_type, live_hp) -> None:
    kwargs = {k: v for k, v in extra.items() if k != "compass"}
    ram = _ram(room=room, **kwargs)
    if "compass" in extra:
        ram[ADDR_COMPASS] = extra["compass"]
    assert success_fn(ram)
    ram[ADDR_OBJ_TYPE + 1] = live_type
    ram[ADDR_OBJ_HP + 1] = live_hp
    assert not success_fn(ram)


def test_west_key_door_controller_from_east_edge() -> None:
    ctl = Level6WestKeyDoorController()
    # East door channel after free return from 0x7a — must LEFT first
    # (vertical blocked at x≈224).
    snap = read_snapshot(
        _ram(room=ROOM_L6_ENTRY, x=224, y=141, keys=1)
    )
    act = ctl.step(snap)
    assert act.reason == "leave_east_door_channel"

    # At fire-wall column, adjust y before crossing.
    ctl2 = Level6WestKeyDoorController()
    snap2 = read_snapshot(
        _ram(room=ROOM_L6_ENTRY, x=208, y=141, keys=1)
    )
    act2 = ctl2.step(snap2)
    assert act2.reason == "east_to_wall_y"

    arrived = Level6WestKeyDoorController()
    arrived.step(read_snapshot(
        _ram(room=ROOM_L6_WEST_WIZZROBE, x=224, y=141, keys=0)
    ))
    assert arrived.success


def _29_ram(*, x: int, y: int, enemy_x: int, enemy_y: int) -> np.ndarray:
    ram = _ram(room=ROOM_L6_DARK_29, x=x, y=y, wizzrobes=1, hp=64)
    ram[ADDR_LINK_X + 1] = enemy_x
    ram[ADDR_LINK_Y + 1] = enemy_y
    return ram


def _clear29_fight(*, x: int, y: int, enemy_x: int, enemy_y: int):
    ctl = Level6Clear29Controller()
    ctl.phase = DungeonPhase.FIGHT
    ctl.combat_frames = 24
    snap = read_snapshot(_29_ram(x=x, y=y, enemy_x=enemy_x, enemy_y=enemy_y))
    return ctl.step(snap)


def test_clear29_patrol_omits_sw_trap() -> None:
    assert CLEAR29_WEST_X == 64
    assert CLEAR29_HANDOFF_Y == 133
    assert CLEAR29_COMBAT_Y == 141
    assert (48, 157) not in ROOM_29_SPEC.combat.patrol
    assert (56, 133) in ROOM_29_SPEC.combat.patrol


def test_clear29_downs_inland_from_north_mouth() -> None:
    """Red 3: LEFT at (120,77) is the door channel. DOWN to y=109 first."""
    act = _clear29_fight(x=120, y=77, enemy_x=184, enemy_y=144)
    assert act.reason == "north_inland"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))


def test_clear29_peels_left_from_north_band_not_east() -> None:
    """(120,109) LEFT is the north band; RIGHT chases into leftover (184,144)."""
    act = _clear29_fight(x=120, y=109, enemy_x=184, enemy_y=144)
    assert act.reason == "west_peel"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("DOWN"))


def test_clear29_may_chase_east_from_west_aisle() -> None:
    """West-only chase left two wizzrobes live for 15000f (reds 1–2)."""
    act = _clear29_fight(x=48, y=133, enemy_x=184, enemy_y=144)
    assert act.reason != "west_peel"
    assert list(act.action) != list(nes_action("LEFT"))


def test_clear29_does_not_walk_south_of_combat_band() -> None:
    """At y=141 hold/slash south; y=157 still peels UP. Do not chase the trap."""
    act = _clear29_fight(x=56, y=141, enemy_x=48, enemy_y=157)
    assert act.reason in ("south_hold", "south_hold_slash")
    assert list(act.action) != list(nes_action("UP"))


def test_clear29_peels_north_from_sw_trap() -> None:
    act = _clear29_fight(x=48, y=157, enemy_x=48, enemy_y=173)
    assert act.reason == "north_handoff"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("DOWN"))


def _cleared_29(*, x: int, y: int):
    ctl = Level6Clear29Controller()
    ctl.phase = DungeonPhase.FIGHT
    ctl.max_live_enemies = 5
    snap = read_snapshot(_ram(room=ROOM_L6_DARK_29, x=x, y=y, wizzrobes=0))
    return ctl, ctl.step(snap)


def test_clear29_rejects_sw_trap_as_success() -> None:
    """Minimized repro: (56,157) with five dead must not be reason=done."""
    ctl, act = _cleared_29(x=56, y=157)
    assert not ctl.success
    assert act.reason != "done"
    assert act.reason == "north_handoff"
    assert list(act.action) == list(nes_action("UP"))


def test_clear29_accepts_historical_handoff() -> None:
    ctl, act = _cleared_29(x=55, y=133)
    assert ctl.success
    assert act.reason == "done"


def test_clear29_accepts_live_north_inland_handoff() -> None:
    """l6_clear29_north_inland leftover (63,133): x<64 y<=133."""
    ctl, act = _cleared_29(x=63, y=133)
    assert ctl.success
    assert act.reason == "done"


def test_clear29_spine_success_requires_handoff_pose() -> None:
    ram = _ram(room=ROOM_L6_DARK_29, x=55, y=133)
    ram[ADDR_ROD] = 1
    ram[ADDR_TRIFORCE] = 0x1F
    assert clear29_handoff_ok(read_snapshot(ram))
    ram[ADDR_LINK_X] = 56
    ram[ADDR_LINK_Y] = 157
    assert not clear29_handoff_ok(read_snapshot(ram))
