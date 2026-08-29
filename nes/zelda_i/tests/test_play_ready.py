"""Composer stop predicate: play_ready / ready. No emulator."""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon.engine import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
)
from zelda_i.ram import (
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
)
from zelda_i.spine.hops import play_ready, ready

STALFOS = 0x2A


def _ram(
    *,
    level: int = 1,
    screen: int = 0x53,
    mode: int = PLAY_MODE,
    x: int = 120,
    y: int = 141,
    keys: int = 1,
    triforce: int = 0x01,
    enemies: int = 0,
    hp: int = 0x20,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    ram[ADDR_TRIFORCE] = triforce
    for slot in range(1, enemies + 1):
        ram[ADDR_OBJ_TYPE + slot] = STALFOS
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_LINK_X + slot] = 80 + slot * 8
        ram[ADDR_LINK_Y + slot] = 141
    return ram


def _snap(**fields: int) -> ZeldaSnapshot:
    return read_snapshot(_ram(**fields))


def _spec() -> DungeonRoomSpec:
    return DungeonRoomSpec(
        spec_id="l1_53",
        source_room=0x53,
        room_id=0x53,
        entry=DoorRoute("UP", ((120, 205),)),
        enemy_types=(STALFOS,),
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE_AND_HP,
        combat=CombatTuning(patrol=((120, 141),)),
        level=1,
    )


def test_play_ready_matching_play_room() -> None:
    assert play_ready(_snap(), level=1, screen=0x53) is True


def test_play_ready_rejects_wrong_level_mode_or_screen() -> None:
    assert play_ready(_snap(level=2), level=1, screen=0x53) is False
    assert play_ready(_snap(mode=11), level=1, screen=0x53) is False
    assert play_ready(_snap(screen=0x54), level=1, screen=0x53) is False


def test_play_ready_rejects_scroll_unless_allowed() -> None:
    scroll = _snap(mode=6)
    assert play_ready(scroll, level=1, screen=0x53) is False
    assert play_ready(scroll, level=1, screen=0x53, allow_transition=True) is False
    assert play_ready(scroll, level=1, screen=0x53, mode=6, allow_transition=True) is True


def test_play_ready_rejects_live_enemies_on_spec() -> None:
    live = _snap(enemies=3)
    clear = _snap(enemies=3, hp=0)
    spec = _spec()
    assert play_ready(live, level=1, screen=0x53, spec=spec) is False
    assert play_ready(clear, level=1, screen=0x53, spec=spec) is True


def test_play_ready_triforce_bit_and_key_gain() -> None:
    assert play_ready(_snap(triforce=0x01), level=1, screen=0x53, tf_bit=0x02) is False
    assert play_ready(_snap(triforce=0x03), level=1, screen=0x53, tf_bit=0x02) is True
    assert play_ready(_snap(keys=1), level=1, screen=0x53, keys_before=1, keys_cmp="gt") is False
    assert play_ready(_snap(keys=2), level=1, screen=0x53, keys_before=1, keys_cmp="gt") is True


def test_ready_binds_as_hop_success() -> None:
    ok = ready(level=1, screen=0x53, tf_bit=0x01)
    assert ok(_snap(triforce=0x01)) is True
    assert ok(_snap(triforce=0x00)) is False
