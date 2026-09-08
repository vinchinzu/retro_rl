from __future__ import annotations

from dataclasses import replace

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import (
    AliveRule,
    DungeonPhase,
    GenericDungeonRoomController,
    RewardKind,
)
from zelda_i.level1.dungeon import (
    ROOM_23_SPEC,
    ROOM_42_SPEC,
    ROOM_43_SPEC,
    ROOM_45_SPEC,
    ROOM_53_SPEC,
    ROOM_54_SPEC,
    ROOM_72_SPEC,
)
from zelda_i.level1.east_dungeon import ROOM_44_SPEC, ROOM_44_SURVIVAL_SPEC
from zelda_i.level1.path import level1_room_72_key_success
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


def test_room72_spec_is_three_keese_floor_key() -> None:
    spec = ROOM_72_SPEC
    assert spec.room_id == 0x72
    assert spec.source_room == 0x73
    assert spec.expected_enemy_count == 3
    assert spec.alive_rule is AliveRule.TYPE
    assert spec.reward.kind is RewardKind.FIXED_INVENTORY
    assert spec.reward.inventory_field == "keys"
    assert spec.room_item_id == 0x19
    keese = read_snapshot(
        _room_ram(room=0x72, enemy_type=0x1B, enemies=3, hp=0)
    )
    assert len(spec.live_enemies(keese)) == 3
    empty = read_snapshot(_room_ram(room=0x72, enemies=0))
    assert len(spec.live_enemies(empty)) == 0


def test_room72_stop_predicate_is_key_increase() -> None:
    """Contract: keys increased in 0x72 with no live Keese. Not room_item gone."""
    live = _room_ram(room=0x72, enemy_type=0x1B, enemies=3, hp=0, keys=1)
    assert level1_room_72_key_success(live, keys_before=1) is False
    still_one = _room_ram(room=0x72, enemies=0, keys=1)
    assert level1_room_72_key_success(still_one, keys_before=1) is False
    got = _room_ram(room=0x72, enemies=0, keys=2)
    assert level1_room_72_key_success(got, keys_before=1) is True
    assert level1_room_72_key_success(got, keys_before=0) is True

    controller = GenericDungeonRoomController(ROOM_72_SPEC)
    controller.step(read_snapshot(live))
    assert controller.phase is DungeonPhase.FIGHT
    assert controller.initial_inventory == 1
    assert controller.success is False
    cleared = _room_ram(room=0x72, enemies=0, keys=1)
    cleared[ADDR_ROOM_ALL_DEAD] = 20
    controller.step(read_snapshot(cleared))
    assert controller.success is False
    picked = _room_ram(room=0x72, enemies=0, keys=2)
    picked[ADDR_ROOM_ALL_DEAD] = 20
    action = controller.step(read_snapshot(picked))
    assert controller.success is True
    assert action.reason == "done"


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
    """Beyond engage_distance: patrol walk, no A. Maze rooms (0x33) need this."""
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


def test_near_enemy_right_walks_right() -> None:
    """Inside engage_distance to the RIGHT: chase RIGHT, never mash UP."""
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x54,
        x=120,
        y=141,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
        enemy_x=120 + 24,
        enemy_y=141,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_engage"
    assert np.array_equal(action.action, nes_action("RIGHT"))
    assert not np.array_equal(action.action, nes_action("UP"))


def test_single_vertex_patrol_empty_room_waits() -> None:
    """On a single-vertex patrol with no live enemies: stand."""
    tuning = replace(ROOM_54_SPEC.combat, patrol=((120, 141),))
    spec = replace(ROOM_54_SPEC, combat=tuning)
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT
    controller.max_live_enemies = 3
    ram = _room_ram(room=0x54, x=120, y=141)
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_wait"
    assert np.array_equal(action.action, nes_idle_action())


def test_single_vertex_patrol_far_enemy_does_not_mash_up() -> None:
    """Far enemy + already on the only patrol vertex: idle, never mash UP."""
    tuning = replace(ROOM_54_SPEC.combat, patrol=((120, 141),))
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
        enemy_x=120 + 80,
        enemy_y=141,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_wait"
    assert np.array_equal(action.action, nes_idle_action())
    assert not np.array_equal(action.action, nes_action("UP"))


def test_patrol_on_vertex_stands() -> None:
    """_patrol with nowhere to walk returns combat_wait, never mashes UP."""
    tuning = replace(ROOM_54_SPEC.combat, patrol=((120, 141),))
    spec = replace(ROOM_54_SPEC, combat=tuning)
    controller = GenericDungeonRoomController(spec)
    action = controller._patrol(read_snapshot(_room_ram(room=0x54, x=120, y=141)))
    assert action.reason == "combat_wait"
    assert np.array_equal(action.action, nes_idle_action())


def test_occupancy_nopath_greedies_toward_maze_waypoint() -> None:
    """Isolated pocket: greedy toward a maze vertex, do not stand forever."""
    x, y = 120, 141
    tuning = replace(
        ROOM_54_SPEC.combat,
        occupancy_patrol=True,
        occupancy_blocked=((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)),
        patrol=((x, y), (x + 80, y)),
    )
    spec = replace(ROOM_54_SPEC, combat=tuning)
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x54,
        x=x,
        y=y,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
        enemy_x=x + 80,
        enemy_y=y,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_patrol"
    assert np.array_equal(action.action, nes_action("RIGHT"))


def test_occupancy_nopath_far_falls_back_to_patrol_not_up() -> None:
    """Occupancy no-path + far: maze patrol waypoints, never mash UP."""
    x, y = 120, 141
    tuning = replace(
        ROOM_54_SPEC.combat,
        occupancy_patrol=True,
        occupancy_blocked=((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)),
        patrol=((x, y),),
    )
    spec = replace(ROOM_54_SPEC, combat=tuning)
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x54,
        x=x,
        y=y,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
        enemy_x=x + 80,
        enemy_y=y,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_wait"
    assert np.array_equal(action.action, nes_idle_action())
    assert not np.array_equal(action.action, nes_action("UP"))


def test_occupancy_nopath_near_still_closes() -> None:
    """Inside engage_distance, miss-blocked dest still greedy-closes."""
    x, y = 120, 141
    tuning = replace(
        ROOM_54_SPEC.combat,
        occupancy_patrol=True,
        occupancy_blocked=((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)),
        patrol=((x, y),),
        engage_distance=48,
    )
    spec = replace(ROOM_54_SPEC, combat=tuning)
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x54,
        x=x,
        y=y,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
        enemy_x=x + 20,
        enemy_y=y,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason.startswith("combat_engage")
    assert np.array_equal(action.action, nes_action("RIGHT")) or np.array_equal(
        action.action, nes_action("RIGHT", "A")
    )


def test_occupancy_far_chases_along_path() -> None:
    """Path exists: occupancy walks the corridor without avoid_walls freeze."""
    tuning = replace(
        ROOM_54_SPEC.combat,
        occupancy_patrol=True,
        patrol=((120, 141),),
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
        enemy_x=120 + 80,
        enemy_y=141,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_patrol"
    assert np.array_equal(action.action, nes_action("RIGHT"))
    assert "_slash" not in action.reason


def test_parked_wallmaster_is_not_chased() -> None:
    controller = GenericDungeonRoomController(ROOM_45_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x45,
        x=32,
        y=141,
        enemy_type=0x27,
        enemies=1,
        hp=0x20,
        enemy_x=0,
        enemy_y=141,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason in ("combat_patrol", "combat_wait")
    assert not np.array_equal(action.action, nes_action("LEFT"))


def test_occupancy_room_does_not_leave_wall_in_south_pocket() -> None:
    """0x23 south door: occupancy chases, avoid_walls does not mash UP."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x23,
        x=96,
        y=175,
        enemy_type=0x06,
        enemies=1,
        hp=0x20,
        enemy_x=176,
        enemy_y=149,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason != "leave_wall"
    assert action.reason in ("combat_patrol", "combat_engage", "combat_engage_slash")


def test_gel_rooms_chase_across_open_floor() -> None:
    assert ROOM_42_SPEC.combat.engage_distance == 160
    assert ROOM_43_SPEC.combat.engage_distance == 160


def test_room44_spec_uses_three_row_occupancy() -> None:
    assert ROOM_44_SPEC.combat.occupancy_bounds == (16, 216, 109, 189)
    assert ROOM_44_SPEC.combat.engage_distance == 80
    assert (48, 165) in ROOM_44_SPEC.combat.patrol
    assert (
        ROOM_44_SURVIVAL_SPEC.combat.occupancy_bounds
        == ROOM_44_SPEC.combat.occupancy_bounds
    )
