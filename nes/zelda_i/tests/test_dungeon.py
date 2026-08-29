from __future__ import annotations

from dataclasses import replace

import numpy as np

from retro_harness.nes import nes_idle_action
from zelda_i.dungeon.engine import (
    DungeonPhase,
    GenericDungeonRoomController,
)
from zelda_i.level1.dungeon import (
    ROOM_23_SPEC,
    ROOM_53_SPEC,
    ROOM_54_SPEC,
)
from zelda_i.ram import (
    ADDR_HEALTH,
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


def _room_ram(
    *,
    room: int,
    enemy_type: int = 0,
    enemies: int = 0,
    hp: int = 0,
    x: int = 120,
    y: int = 141,
    keys: int = 0,
    enemy_x: int | None = None,
    enemy_y: int | None = None,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 1
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_HEALTH] = 0x20
    ram[ADDR_KEYS] = keys
    if enemies <= 0:
        return ram
    for slot in range(1, enemies + 1):
        ram[ADDR_OBJ_TYPE + slot] = enemy_type
        ram[ADDR_OBJ_HP + slot] = hp
        if enemy_x is None:
            ram[ADDR_LINK_X + slot] = 80 + slot * 8
        else:
            ram[ADDR_LINK_X + slot] = enemy_x + (slot - 1) * 4
        ram[ADDR_LINK_Y + slot] = (
            93 + slot * 8 if enemy_y is None else enemy_y
        )
    return ram


def test_room_specs_support_hp_and_type_only_liveness() -> None:
    stalfos = read_snapshot(
        _room_ram(room=0x53, enemy_type=0x2A, enemies=5, hp=0x20)
    )
    dead_stalfos = read_snapshot(
        _room_ram(room=0x53, enemy_type=0x2A, enemies=5, hp=0)
    )
    keese = read_snapshot(
        _room_ram(room=0x54, enemy_type=0x1B, enemies=8, hp=0)
    )
    assert len(ROOM_53_SPEC.live_enemies(stalfos)) == 5
    assert len(ROOM_53_SPEC.live_enemies(dead_stalfos)) == 0
    assert len(ROOM_54_SPEC.live_enemies(keese)) == 8


def test_generic_controller_routes_and_clears_type_only_room() -> None:
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    source = read_snapshot(_room_ram(room=0x53, x=120, y=109))
    action = controller.step(source)
    assert action.reason == "entry_route"

    live_ram = _room_ram(room=0x54, enemy_type=0x1B, enemies=8, hp=0)
    action = controller.step(read_snapshot(live_ram))
    assert controller.phase is DungeonPhase.FIGHT
    assert action.reason.startswith("combat_")
    assert controller.max_live_enemies == 8

    clear_ram = _room_ram(room=0x54, enemies=0)
    clear_ram[ADDR_ROOM_ALL_DEAD] = 20
    action = controller.step(read_snapshot(clear_ram))
    assert controller.success is True
    assert controller.phase is DungeonPhase.DONE
    assert action.reason == "done"


def test_controller_fails_fast_after_leaving_target_room() -> None:
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    controller.clear_signal_seen = True
    ram = _room_ram(room=0x73, x=160, y=173, keys=0)
    action = controller.step(read_snapshot(ram))
    assert controller.success is False
    assert controller.phase is DungeonPhase.FAILED
    assert action.reason == "left_target_room"
    assert "left_target_room" in controller.notes


def test_collect_reward_stands_after_one_waypoint_lap() -> None:
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    ram = _room_ram(room=0x23, x=112, y=93, keys=0)
    ram[ADDR_ROOM_ALL_DEAD] = 24
    n = len(ROOM_23_SPEC.reward.waypoints)
    action = None
    for _ in range(n * 30):
        action = controller.step(read_snapshot(ram))
        if action.reason == "collect_wait":
            break
    assert action is not None
    assert action.reason == "collect_wait"
    assert np.array_equal(action.action, nes_idle_action())
    assert controller._collect_skips >= n


def test_combat_stands_while_waiting_for_clear() -> None:
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.max_live_enemies = 3
    ram = _room_ram(room=0x54, x=120, y=141)
    ram[ADDR_ROOM_ALL_DEAD] = 0
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_wait"
    assert np.array_equal(action.action, nes_idle_action())


def test_engage_far_enemy_walks_without_slash() -> None:
    """Chase within engage distance but outside sword reach: no A pulse."""
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x54,
        x=120,
        y=141,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
        enemy_x=120 + 40,
        enemy_y=141,
    )
    snap = read_snapshot(ram)
    action = controller.step(snap)
    assert action.reason == "combat_engage"
    assert not action.reason.endswith("_slash")
    for _ in range(6):
        action = controller.step(snap)
        assert action.reason == "combat_engage"
        assert "_slash" not in action.reason


def test_engage_enemy_in_sword_hitbox_slashes() -> None:
    """Enemy in blade rectangle → combat_engage_slash on attack hold frames."""
    tuning = replace(
        ROOM_54_SPEC.combat,
        engage_distance=64,
        engage_attack_period=8,
        engage_attack_hold=4,
        attack_phase=0,
    )
    spec = replace(ROOM_54_SPEC, combat=tuning)
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x54,
        x=120,
        y=141,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
        enemy_x=120 + 12,
        enemy_y=141,
    )
    snap = read_snapshot(ram)
    action = controller.step(snap)
    assert action.reason == "combat_engage_slash"


def test_patrol_does_not_slash() -> None:
    """Patrol (enemies beyond engage distance, or none) walks without A."""
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x54,
        x=120,
        y=141,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
        enemy_x=120 + 80,
        enemy_y=141,
    )
    snap = read_snapshot(ram)
    for _ in range(16):
        action = controller.step(snap)
        assert action.reason == "combat_patrol"
        assert "_slash" not in action.reason
