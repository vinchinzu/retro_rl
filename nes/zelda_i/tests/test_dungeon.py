from __future__ import annotations

from dataclasses import replace

import numpy as np

from retro_harness.controls import pressed_nes_buttons
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
    GORIYA_OBJECT_TYPE,
)
from zelda_i.level1_dungeon import (
    ROOM_23_SPEC,
    ROOM_44_SURVIVAL_SPEC,
    ROOM_53_SPEC,
    ROOM_54_SPEC,
)
from zelda_i.level1_east_dungeon import Room44SurvivalController
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


def test_room23_center_occupancy_sidesteps_mid_water() -> None:
    """From (128,149) a north Goriya is UP through water; miss then sidesteps."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(room=0x23, x=128, y=149, enemy_type=0x06, enemies=1, hp=0x20)
    ram[ADDR_LINK_X + 1] = 128
    ram[ADDR_LINK_Y + 1] = 80
    snap = read_snapshot(ram)
    first = controller.step(snap)
    assert first.reason == "combat_patrol"
    assert np.array_equal(first.action, nes_action("UP"))
    second = controller.step(snap)
    assert controller.walker.misses == 1
    assert (128, 148) in controller.walker.grid.blocked
    assert second.reason == "combat_patrol"
    assert not np.array_equal(second.action, nes_action("UP"))


def test_room44_survival_cannot_stand_north_door() -> None:
    """v3 leftover (40,93): Survival chase must step DOWN off the north band."""
    controller = Room44SurvivalController(ROOM_44_SURVIVAL_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x44, x=40, y=93, enemy_type=GORIYA_OBJECT_TYPE, enemies=1, hp=0x30
    )
    ram[ADDR_LINK_X + 1] = 80
    ram[ADDR_LINK_Y + 1] = 93
    action = controller.step(read_snapshot(ram))
    pressed = pressed_nes_buttons(list(action.action))
    assert "DOWN" in pressed
    assert "UP" not in pressed

    boxed = Room44SurvivalController(ROOM_44_SURVIVAL_SPEC)
    boxed.phase = DungeonPhase.FIGHT
    boxed.walker.grid.blocked.update((x, 109) for x in range(16, 80))
    ram2 = _room_ram(
        room=0x44, x=66, y=109, enemy_type=GORIYA_OBJECT_TYPE, enemies=1, hp=0x30
    )
    ram2[ADDR_LINK_X + 1] = 48
    ram2[ADDR_LINK_Y + 1] = 93
    action2 = boxed.step(read_snapshot(ram2))
    pressed2 = pressed_nes_buttons(list(action2.action))
    assert action2.reason.startswith("combat_patrol") or action2.reason.startswith(
        "combat_wait"
    )
    assert not (pressed2 == {"LEFT"} or pressed2 == {"LEFT", "A"})

    split = Room44SurvivalController(ROOM_44_SURVIVAL_SPEC)
    split.phase = DungeonPhase.FIGHT
    ram3 = _room_ram(
        room=0x44, x=66, y=109, enemy_type=GORIYA_OBJECT_TYPE, enemies=2, hp=0x30
    )
    ram3[ADDR_LINK_X + 1] = 48
    ram3[ADDR_LINK_Y + 1] = 93
    ram3[ADDR_LINK_X + 2] = 192
    ram3[ADDR_LINK_Y + 2] = 141
    action3 = split.step(read_snapshot(ram3))
    pressed3 = pressed_nes_buttons(list(action3.action))
    assert action3.reason.startswith("west_inland")
    assert "UP" not in pressed3
    assert "LEFT" in pressed3


def test_room44_survival_east_first_from_west_mouth() -> None:
    """v8 leftover (32,167): peel east on y=165; do not occupancy-chase."""

    def _fight(x: int, y: int, gx: int, gy: int, *, blocked: int = 0):
        ctl = Room44SurvivalController(ROOM_44_SURVIVAL_SPEC)
        ctl.phase = DungeonPhase.FIGHT
        if blocked:
            ctl.walker.grid.blocked.update((16 + i, 167) for i in range(blocked))
        ram = _room_ram(
            room=0x44,
            x=x,
            y=y,
            enemy_type=GORIYA_OBJECT_TYPE,
            enemies=1,
            hp=0x30,
        )
        ram[ADDR_LINK_X + 1] = gx
        ram[ADDR_LINK_Y + 1] = gy
        action = ctl.step(read_snapshot(ram))
        return ctl, action, pressed_nes_buttons(list(action.action))

    leftover, act, pressed = _fight(32, 167, 192, 117, blocked=76)
    assert act.reason.startswith("west_inland")
    assert "RIGHT" in pressed
    assert "LEFT" not in pressed
    assert leftover.walker.misses == 0
    assert not leftover.walker.grid.blocked

    _, tunnel_act, tunnel_pressed = _fight(16, 141, 192, 141)
    assert tunnel_act.reason.startswith("west_inland")
    assert "RIGHT" in tunnel_pressed
    assert "DOWN" not in tunnel_pressed

    _, door_act, door_pressed = _fight(32, 141, 192, 141)
    assert door_act.reason.startswith("west_inland")
    assert "DOWN" in door_pressed
    assert "RIGHT" not in door_pressed

    _, inland_act, inland_pressed = _fight(80, 141, 192, 141)
    assert inland_act.reason.startswith("east_column")
    assert "DOWN" in inland_pressed

    _, north_act, north_pressed = _fight(88, 141, 128, 109)
    assert north_act.reason.startswith("east_column")
    assert "DOWN" in north_pressed
    assert "RIGHT" not in north_pressed

    _, aisle_act, aisle_pressed = _fight(88, 165, 128, 109)
    assert aisle_act.reason.startswith("east_column")
    assert "RIGHT" in aisle_pressed

    _, col_act, col_pressed = _fight(192, 165, 128, 109)
    assert col_act.reason.startswith("east_column")
    assert "UP" in col_pressed

    _, band_act, band_pressed = _fight(56, 109, 176, 133)
    assert band_act.reason.startswith("west_inland")
    assert "LEFT" in band_pressed
    assert "DOWN" not in band_pressed

    _, mid_act, mid_pressed = _fight(80, 157, 176, 141)
    assert mid_act.reason.startswith("east_column")
    assert "DOWN" in mid_pressed


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
