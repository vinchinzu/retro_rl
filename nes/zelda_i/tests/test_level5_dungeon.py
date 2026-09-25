"""Unit tests for Level 5 leftover walks that would burn again."""

from __future__ import annotations

import numpy as np
import pytest

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import GenericDungeonRoomController, KEESE_OBJECT_TYPE
from zelda_i.level5 import boss_path
from zelda_i.level5.dungeon import (
    GIBDO_OBJECT_TYPE,
    LEVEL_5,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_POLS_77,
    ROOM_25_SPEC,
    ROOM_26_SPEC,
    ROOM_27_SPEC,
    ROOM_66_SPEC,
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
    ADDR_ROOM_ITEM_X,
    ADDR_ROOM_ITEM_Y,
    ADDR_ROOM_ALL_DEAD,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
    room_flags_addr,
    room_item_taken,
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


def test_room_66_peels_contact_gibdo_at_leftover() -> None:
    """Death (128,133): peel a north Gibdo; do not greedy-close UP into the body."""
    ram = _ram(room=ROOM_L5_GIBDO_66, x=128, y=133, enemies=1, hp=112)
    ram[ADDR_LINK_X + 1] = 128
    ram[ADDR_LINK_Y + 1] = 125
    act = GenericDungeonRoomController(spec=ROOM_66_SPINE_SPEC).step(
        read_snapshot(ram)
    )
    assert act.reason == "combat_contact_peel"
    assert act.action == nes_action("DOWN")
    assert act.action != nes_action("UP")
    assert act.action != nes_action("UP", "A")


def test_room_66_chase_stays_inside_west_door(monkeypatch) -> None:
    """C8a replay: a lattice detour LEFT at (32,141) scrolled into 0x65."""
    ram = _ram(room=0x66, x=32, y=141, enemies=1, hp=80)
    ram[ADDR_LINK_X + 1] = 55
    ram[ADDR_LINK_Y + 1] = 141
    snap = read_snapshot(ram)
    ctl = GenericDungeonRoomController(spec=ROOM_66_SPEC)
    monkeypatch.setattr(ctl, "_contact", lambda *args, **kwargs: None)
    action = ctl._engage(snap, snap.objects[1], direction="LEFT")
    assert action.reason == "combat_engage"
    assert action.action == nes_action("RIGHT")


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


@pytest.mark.parametrize(
    ("room", "expect", "spec"),
    ((0x26, 0x25, ROOM_26_SPEC), (0x25, 0x24, ROOM_25_SPEC)),
)
def test_west_walk_fights_its_room_once_before_door(
    monkeypatch, room: int, expect: int, spec
) -> None:
    class Env:
        ram = _ram(room=room)

        def get_ram(self):
            return self.ram

    env = Env()
    calls: list[object] = []

    def fight(_env, _assist, _total, _hops, selected, types, _name):
        calls.append((selected, types))
        return True

    def walk(_env, _assist, _total):
        calls.append("door")
        env.ram[ADDR_SCREEN] = expect
        return {}

    monkeypatch.setattr(boss_path, "_fight_if_live", fight)
    monkeypatch.setattr(boss_path, "wait_play", lambda *args, **kwargs: None)
    hops: list[dict] = []
    assert boss_path._walk_west(env, None, [], hops, walk, expect, "west")
    expected_calls = [(spec, spec.enemy_types)]
    expected_calls.append("door")
    assert calls == expected_calls
    assert hops[-1]["success"]


def test_west_27_probes_optional_key_without_full_clear(monkeypatch) -> None:
    class Env:
        ram = _ram(room=0x27, enemies=1)

        def get_ram(self):
            return self.ram

    env = Env()
    monkeypatch.setattr(
        boss_path, "_fight_if_live",
        lambda *args, **kwargs: pytest.fail("0x27 full clear is optional"),
    )
    monkeypatch.setattr(boss_path, "wait_play", lambda *args, **kwargs: None)

    def walk(_env, _assist, _total):
        env.ram[ADDR_SCREEN] = 0x26
        return {}

    hops: list[dict] = []
    assert boss_path._walk_west(env, None, [], hops, walk, 0x26, "27_west")
    assert hops[0] == {"hop": "key_27", "ok": False, "reason": "not_visible"}
    assert hops[-1]["success"]


def test_west_room_27_counts_hp_zero_keese_as_live(monkeypatch) -> None:
    class Env:
        ram = _ram(room=0x27, enemy_type=KEESE_OBJECT_TYPE, enemies=1, hp=0)

        def get_ram(self):
            return self.ram

    fought = []
    monkeypatch.setattr(boss_path, "wait_ram", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        boss_path, "fight_ctl",
        lambda _env, _assist, _total, spec: fought.append(spec) or {"ok": True},
    )
    assert boss_path._fight_if_live(
        Env(), None, [], [], ROOM_27_SPEC, ROOM_27_SPEC.enemy_types, "fight_27"
    )
    assert len(fought) == 1
    assert fought[0].expected_enemy_count == 1


def test_west_key_pickup_checks_room_item_flag_before_door(monkeypatch) -> None:
    class Env:
        ram = _ram(room=0x26, keys=3)

        def get_ram(self):
            return self.ram

    env = Env()
    env.ram[ADDR_ROOM_ITEM_X] = 120
    env.ram[ADDR_ROOM_ITEM_Y] = 141
    monkeypatch.setattr(boss_path, "room_step", lambda *args, **kwargs: "LEFT")

    def step(_env, _assist, _total, _action):
        env.ram[room_flags_addr(LEVEL_5) + 0x26] |= 0x10

    monkeypatch.setattr(boss_path, "_step", step)
    hops: list[dict] = []
    assert boss_path._collect_west_key(env, None, [], hops, 0x26, 3)
    assert room_item_taken(env.ram, LEVEL_5, 0x26)
    assert hops[-1]["item_taken"]
    assert hops[-1]["ok"]
