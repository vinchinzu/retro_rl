from __future__ import annotations

from dataclasses import replace

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import (
    AliveRule,
    DungeonPhase,
    GenericDungeonRoomController,
    GORIYA_OBJECT_TYPE,
    RewardKind,
)

from zelda_i.dungeon.ids import HEART_DROP_OBJECT_TYPE, HEART_DROP_STATE
from zelda_i.level1.dungeon import (
    ROOM_23_SPEC,
    ROOM_33_SPEC,
    ROOM_42_SPEC,
    ROOM_43_SPEC,
    ROOM_45_SPEC,
    ROOM_53_SPEC,
    ROOM_54_SPEC,
    ROOM_72_SPEC,
    Room23HeartSafeController,
    Room33ScoopController,
)
from zelda_i.level1.east_dungeon import (
    ROOM_44_SPEC,
    ROOM_44_SURVIVAL_SPEC,
    ROOM_45_SURVIVAL_SPEC,
)
from zelda_i.level1.path import level1_room_72_key_success
from zelda_i.combat import FACING_SOUTH
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_STATE,
    ADDR_OBJ_TYPE,
    ADDR_ROOM_ALL_DEAD,
    ADDR_SCREEN,
    PLAY_MODE,
    ZeldaSnapshot,
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
    assert controller.swings == 1
    assert controller.swings_authorized == 1
    assert controller.engage_frames == 1
    assert controller.report()["damage"]["swings"] == 1


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


def test_room23_water_bar_leaves_west_passage_and_south_floor() -> None:
    """West 16px column and south corridor are floor; $6530 water bar is blocked."""
    blocked = set(ROOM_23_SPEC.combat.occupancy_blocked)
    assert (78, 157) not in blocked
    assert (144, 149) not in blocked
    assert (64, 141) not in blocked
    assert (88, 149) not in blocked
    assert (120, 136) in blocked
    assert (88, 148) in blocked
    walker = GenericDungeonRoomController(ROOM_23_SPEC).walker
    step = walker.next_dir((78, 157), (120, 125))
    assert step == "UP"
    # Live leftover (88,149): around west, never UP into the bar.
    assert walker.next_dir((88, 149), (148, 125)) == "LEFT"


def test_room23_stands_when_boomerang_blocks_the_step() -> None:
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x23,
        x=78,
        y=157,
        enemy_type=0x06,
        enemies=1,
        hp=0x20,
        enemy_x=64,
        enemy_y=125,
    )
    ram[ADDR_OBJ_TYPE + 2] = 0x5C
    ram[ADDR_LINK_X + 2] = 78
    ram[ADDR_LINK_Y + 2] = 149
    action = controller.step(read_snapshot(ram))
    assert action.reason != "leave_wall"
    assert not np.array_equal(action.action, nes_action("UP"))


def test_room23_heart_safe_holds_south_on_one_heart() -> None:
    """Leftover (128,149) health 0x20: peel DOWN, do not chase (128,117)."""
    controller = Room23HeartSafeController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x23,
        x=128,
        y=149,
        enemy_type=0x06,
        enemies=1,
        hp=0x20,
        enemy_x=128,
        enemy_y=117,
    )
    action = controller.step(read_snapshot(ram))
    assert np.array_equal(action.action, nes_action("DOWN"))
    assert action.reason == "heart_safe_peel_south"
    assert not np.array_equal(action.action, nes_action("UP"))
    assert not np.array_equal(action.action, nes_idle_action())


def test_room23_heart_safe_peels_down_off_plus_stem() -> None:
    """Leftover (135,149) health 0x20 + boomerang north: DOWN, never idle/UP."""
    controller = Room23HeartSafeController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x23,
        x=135,
        y=149,
        enemy_type=0x06,
        enemies=1,
        hp=0x20,
        enemy_x=128,
        enemy_y=117,
    )
    ram[ADDR_OBJ_TYPE + 2] = 0x5C
    ram[ADDR_LINK_X + 2] = 128
    ram[ADDR_LINK_Y + 2] = 141
    action = controller.step(read_snapshot(ram))
    assert np.array_equal(action.action, nes_action("DOWN"))
    assert action.reason == "heart_safe_peel_south"
    assert not np.array_equal(action.action, nes_idle_action())
    assert not np.array_equal(action.action, nes_action("UP"))


def test_room23_heart_safe_clamps_corridor_and_returns_from_door() -> None:
    """1-heart: DOWN off plus-stem, UP from south mouth, no DOWN at y=157."""

    def _act(x: int, y: int):
        controller = Room23HeartSafeController(ROOM_23_SPEC)
        controller.phase = DungeonPhase.FIGHT
        ram = _room_ram(
            room=0x23,
            x=x,
            y=y,
            enemy_type=0x06,
            enemies=1,
            hp=0x20,
            enemy_x=128,
            enemy_y=117,
        )
        return controller.step(read_snapshot(ram))

    peel = _act(135, 149)
    assert np.array_equal(peel.action, nes_action("DOWN"))
    door = _act(120, 205)
    assert np.array_equal(door.action, nes_action("UP"))
    mid = _act(120, 189)
    assert np.array_equal(mid.action, nes_action("UP"))
    hold = _act(120, 157)
    assert not np.array_equal(hold.action, nes_action("DOWN"))


def test_room23_occupancy_stands_on_goriya_instead_of_walking() -> None:
    """No occupancy path (already on target): stand/slash, do not chase."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x23,
        x=128,
        y=117,
        enemy_type=0x06,
        enemies=1,
        hp=0x20,
        enemy_x=128,
        enemy_y=117,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason in (
        "combat_wait",
        "combat_engage_slash",
        "combat_backstep",
    )


def test_room23_far_corridor_chase_is_patrol_not_slash() -> None:
    """engage_distance=24: 56px south-corridor Goriya is occupancy walk, no A."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x23,
        x=64,
        y=157,
        enemy_type=0x06,
        enemies=1,
        hp=0x20,
        enemy_x=120,
        enemy_y=157,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_patrol"
    assert "_slash" not in action.reason
    assert controller.patrol_frames >= 1
    assert controller.engage_frames == 0
    assert controller.swings == 0
    assert controller.report()["damage"]["swings"] == 0


def test_room23_hitbox_close_backsteps_before_slash() -> None:
    """contact_backstep=16 covers the sword box: first close frames peel, no A."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x23,
        x=120,
        y=157,
        enemy_type=0x06,
        enemies=1,
        hp=0x20,
        enemy_x=132,
        enemy_y=157,
    )
    snap = read_snapshot(ram)
    action = controller.step(snap)
    assert abs(132 - 120) + abs(157 - 157) < ROOM_23_SPEC.combat.contact_backstep
    # combat_frames % 6 < 2: frame 1 peels; frame 2 (% 6 == 2) already engages.
    assert action.reason == "combat_backstep"
    assert controller.backstep_frames == 1
    assert controller.swings == 0
    assert controller.swings_authorized == 1
    action = controller.step(snap)
    assert action.reason.startswith("combat_engage")
    assert controller.engage_frames == 1
    assert controller.backstep_frames == 1


def test_room23_across_water_is_patrol_until_engage_cap() -> None:
    """32px north across the water bar: occupancy walk, no A (manhattan >= 24)."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x23,
        x=120,
        y=157,
        enemy_type=0x06,
        enemies=1,
        hp=0x20,
        enemy_x=120,
        enemy_y=125,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_patrol"
    assert "_slash" not in action.reason
    assert controller.engage_frames == 0
    assert controller.swings == 0
    assert controller.swings_authorized == 0


def test_room23_leftover_chase_goes_west_not_up_into_water() -> None:
    """(88,149) leftover: occupancy LEFT around the $6530 bar, never UP."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x23,
        x=88,
        y=149,
        enemy_type=0x06,
        enemies=1,
        hp=0x20,
        enemy_x=148,
        enemy_y=125,
    )
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_patrol"
    assert np.array_equal(action.action, nes_action("LEFT"))
    assert not np.array_equal(action.action, nes_action("UP"))
    assert not np.array_equal(action.action, nes_action("UP", "A"))
    assert controller.swings == 0
    assert controller.engage_frames == 0


def test_combat_target_contact_miss_is_not_blocked() -> None:
    """rr-8t4.4 general fix #1: a miss on the fight target's own cell.

    ``_occupancy_bodies`` carves the target's cell out of the BFS planning
    set (it is the goal), which used to mean a physical-collision miss
    there read as ground truth and permanently walled real floor next to
    wherever the live target stood. ``transient_occupants`` (now built from
    every live body, target included) exempts it.
    """
    # A tiny engage_distance keeps this in the occupancy/BFS branch even at
    # short range, isolating the mechanism under test: whether a body within
    # collision range of the *predicted* cell scars the grid on a miss.
    tuning = replace(
        ROOM_54_SPEC.combat,
        occupancy_patrol=True,
        occupancy_blocked=(),
        occupancy_bounds=None,
        patrol=((120, 93),),
        engage_distance=2,
        contact_backstep=0,
    )
    spec = replace(
        ROOM_54_SPEC,
        enemy_types=(GORIYA_OBJECT_TYPE,),
        alive_rule=AliveRule.TYPE,
        combat=tuning,
    )
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT

    def snap_at(link_xy: tuple[int, int], target_xy: tuple[int, int]) -> ZeldaSnapshot:
        ram = _room_ram(
            room=0x54,
            x=link_xy[0],
            y=link_xy[1],
            enemy_type=int(GORIYA_OBJECT_TYPE),
            enemies=1,
            hp=0x40,
            enemy_x=target_xy[0],
            enemy_y=target_xy[1],
        )
        return read_snapshot(ram)

    # Target 9px above Link (still >= engage_distance): the predicted UP
    # step (120,140) sits inside the target's own occupancy halo.
    snap1 = snap_at((120, 141), (120, 132))
    live = spec.live_enemies(snap1)
    controller._combat(snap1, live)
    assert controller.walker.last_dir == "UP"
    # No movement: Link's hitbox blocked the step -- the target is close
    # enough (8px) to physically contest the cell BFS was told is free.
    action2 = controller._combat(snap1, spec.live_enemies(snap1))
    assert controller.walker.misses >= 1
    assert (120, 140) not in controller.walker.grid.blocked
    assert action2.reason in ("combat_patrol", "combat_engage", "combat_engage_slash")


def test_combat_backstep_gap_does_not_stale_grade_the_walker() -> None:
    """rr-8t4.4 general fix #2: grading must not span a multi-frame gap.

    ``contact_backstep`` (and dash / off-wall) used to return early without
    ever calling ``observe``/``next_dir``, so ``last_xy`` went stale for
    however many frames they ran. The next real occupancy-branch call then
    graded a multi-frame real displacement as a single missed 1px step and
    blacklisted a cell with no relation to any wall or body -- this dwarfed
    the target-cell miss count in the ROM (rr-8t4.4 0x23, 4627/6000).
    Grading now happens every combat frame regardless of which branch acts,
    so a real, successful backstep-then-resume sequence never misses.
    """
    tuning = replace(
        ROOM_54_SPEC.combat,
        occupancy_patrol=True,
        occupancy_blocked=(),
        occupancy_bounds=None,
        patrol=((120, 93),),
        engage_distance=4,
        contact_backstep=16,
    )
    spec = replace(
        ROOM_54_SPEC,
        enemy_types=(GORIYA_OBJECT_TYPE,),
        alive_rule=AliveRule.TYPE,
        combat=tuning,
    )
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT
    controller.combat_frames = 4  # next call -> 5 (5 % 6 == 5, not backstep)

    def snap_with(link_xy: tuple[int, int], target_xy: tuple[int, int]) -> ZeldaSnapshot:
        ram = _room_ram(
            room=0x54,
            x=link_xy[0],
            y=link_xy[1],
            enemy_type=int(GORIYA_OBJECT_TYPE),
            enemies=1,
            hp=0x40,
            enemy_x=target_xy[0],
            enemy_y=target_xy[1],
        )
        return read_snapshot(ram)

    # Frame 1 (combat_frames=5): target far, straight open chase UP.
    snap = snap_with((120, 161), (120, 93))
    controller._combat(snap, spec.live_enemies(snap))
    assert controller.walker.last_dir == "UP"

    # Frame 2 (combat_frames=6, backstep-eligible): Link's real UP step
    # landed; the target darts close (distance 10 < contact_backstep 16).
    snap = snap_with((120, 160), (120, 150))
    action = controller._combat(snap, spec.live_enemies(snap))
    assert action.reason == "combat_backstep"
    assert controller.walker.misses == 0

    # Frame 3 (combat_frames=7, still backstep-eligible): Link's real DOWN
    # backstep landed; target holds close.
    snap = snap_with((120, 161), (120, 150))
    action = controller._combat(snap, spec.live_enemies(snap))
    assert action.reason == "combat_backstep"
    assert controller.walker.misses == 0

    # Frame 4 (combat_frames=8, backstep window closed): Link's real DOWN
    # backstep landed again; target retreats, occupancy branch resumes.
    snap = snap_with((120, 162), (120, 93))
    controller._combat(snap, spec.live_enemies(snap))
    assert controller.walker.misses == 0
    assert not controller.walker.grid.blocked


def test_prefix_specs_contact_backstep_keeps_hearts() -> None:
    """0x53 / 0x33 peel contact so 0x23 is entered with lo>=2."""
    assert ROOM_53_SPEC.combat.contact_backstep >= 24
    assert ROOM_33_SPEC.combat.contact_backstep >= 24
    assert ROOM_33_SPEC.combat.evade is True
    assert ROOM_53_SPEC.combat.evade is False
    assert ROOM_42_SPEC.combat.contact_backstep >= 16
    assert ROOM_43_SPEC.combat.contact_backstep >= 16


def test_room33_hold_slashes_in_place_outside_the_pad() -> None:
    """Dump 1962f: Stalfos at cheb 18 was in the UP box while we peeled into pad.

    Hold-north: already facing, A in place, do not walk into MIN_DODGE_BODY.
    """
    controller = Room33ScoopController(ROOM_33_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x33,
        x=120,
        y=117,
        enemy_type=0x2A,
        enemies=1,
        hp=0x20,
        enemy_x=120,
        enemy_y=117 + 18,
    )
    ram[ADDR_HEALTH] = 0x22
    ram[ADDR_LINK_FACING] = FACING_SOUTH
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_hold_slash"
    assert np.array_equal(action.action, nes_action("A"))
    assert not np.array_equal(action.action, nes_action("DOWN"))
    assert not np.array_equal(action.action, nes_action("DOWN", "A"))


def test_room33_hold_steps_away_inside_the_pad() -> None:
    controller = Room33ScoopController(ROOM_33_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x33,
        x=120,
        y=117,
        enemy_type=0x2A,
        enemies=1,
        hp=0x20,
        enemy_x=120,
        enemy_y=117 + 8,
    )
    ram[ADDR_HEALTH] = 0x22
    action = controller.step(read_snapshot(ram))
    assert action.reason == "combat_evade_body"
    assert np.array_equal(action.action, nes_action("UP"))


def test_room33_scoops_heart_when_filled_hearts_one_not_fight() -> None:
    """0x33 leftover (96,173) health 0x21 + heart drop: scoop, not FIGHT."""
    controller = Room33ScoopController(ROOM_33_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x33,
        x=96,
        y=173,
        enemy_type=0x2A,
        enemies=1,
        hp=0x20,
        enemy_x=104,
        enemy_y=173,
    )
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_OBJ_TYPE + 2] = 0x60
    ram[ADDR_OBJ_STATE + 2] = 0x22
    ram[ADDR_LINK_X + 2] = 128
    ram[ADDR_LINK_Y + 2] = 173
    snap = read_snapshot(ram)
    assert snap.filled_hearts == 1
    action = controller.step(snap)
    assert action.reason == "scoop_heart"
    assert not action.reason.startswith("combat_")
    assert not np.array_equal(action.action, nes_idle_action())
    assert np.array_equal(action.action, nes_action("RIGHT"))
    assert controller.report()["last_health"] == 0x21
    assert ROOM_33_SPEC.combat.contact_backstep >= 24


def test_room33_live_stalfos_fights_not_heart_wait() -> None:
    """Live Stalfos + lo=1 + key: fight/backstep, do not start heart-wait."""
    controller = Room33ScoopController(ROOM_33_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.initial_inventory = 0
    controller.max_live_enemies = 3
    ram = _room_ram(
        room=0x33,
        x=96,
        y=173,
        enemy_type=0x2A,
        enemies=1,
        hp=0x20,
        enemy_x=104,
        enemy_y=173,
        keys=1,
    )
    ram[ADDR_HEALTH] = 0x21
    action = controller.step(read_snapshot(ram))
    assert controller.heart_wait == 0
    assert controller.success is False
    assert action.reason != "0x33_needs_heart"
    assert action.reason != "scoop_heart"
    assert action.reason.startswith("combat_")
    assert ROOM_33_SPEC.combat.contact_backstep >= 24


def test_room33_cleared_low_scoops_heart_not_done() -> None:
    """Cleared 0x33, lo=1, key in inventory, nearby heart: scoop, not DONE."""
    controller = Room33ScoopController(ROOM_33_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    controller.max_live_enemies = 3
    ram = _room_ram(room=0x33, x=96, y=173, keys=1)
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_ROOM_ALL_DEAD] = 24
    ram[ADDR_OBJ_TYPE + 1] = 0x60
    ram[ADDR_OBJ_STATE + 1] = 0x22
    ram[ADDR_LINK_X + 1] = 128
    ram[ADDR_LINK_Y + 1] = 173
    action = controller.step(read_snapshot(ram))
    assert action.reason == "scoop_heart"
    assert controller.success is False
    assert controller.phase is not DungeonPhase.DONE
    assert np.array_equal(action.action, nes_action("RIGHT"))


def test_room33_cleared_low_no_drop_holds_then_needs_heart() -> None:
    """Cleared 0x33, lo=1, no drop: not DONE; short wait then 0x33_needs_heart."""
    controller = Room33ScoopController(ROOM_33_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    controller.max_live_enemies = 3
    controller.heart_wait_limit = 2
    ram = _room_ram(room=0x33, x=96, y=173, keys=1)
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_ROOM_ALL_DEAD] = 24
    snap = read_snapshot(ram)
    action = controller.step(snap)
    assert controller.success is False
    assert action.reason != "done"
    assert controller.phase is not DungeonPhase.DONE
    action = controller.step(snap)
    assert action.reason == "0x33_needs_heart"
    assert controller.phase is DungeonPhase.FAILED
    assert controller.success is False
    assert "0x33_needs_heart" in controller.notes


def test_room33_cleared_low_walks_key_tile_from_leftover() -> None:
    """Wave-12 leftover (80,165) live==0 lo=1: walk to (96,173), not idle/DONE.

    Occupancy BFS is south-then-east (DOWN first). Naive 4-way would RIGHT.
    """
    controller = Room33ScoopController(ROOM_33_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    controller.max_live_enemies = 3
    ram = _room_ram(room=0x33, x=80, y=165, keys=1)
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_ROOM_ALL_DEAD] = 24
    ram[ADDR_OBJ_TYPE + 1] = 0x60
    ram[ADDR_OBJ_STATE + 1] = 0x19
    ram[ADDR_LINK_X + 1] = 96
    ram[ADDR_LINK_Y + 1] = 173
    action = controller.step(read_snapshot(ram))
    assert controller.success is False
    assert controller.phase is not DungeonPhase.DONE
    assert action.reason != "done"
    assert action.reason != "0x33_needs_heart"
    assert controller.heart_wait == 0
    assert not np.array_equal(action.action, nes_idle_action())
    assert np.array_equal(action.action, nes_action("DOWN"))
    assert action.reason == "scoop_key_tile"


def test_room33_key_tile_walk_does_not_mash_right_into_east_244() -> None:
    """Live leftover (88,165)→(96,173): 4-way tie is RIGHT into $6530 0xF4.

    Dump of L1 0x33: cell (96,160) is tile 244; y=176 row at x=80..128 is
    floor. Naive abs(dx)>=abs(dy) mashes RIGHT and oscillates 88↔89.
    Occupancy BFS drops south along x=88, then east at y=173.
    """
    dx, dy = 96 - 88, 173 - 165
    assert abs(dx) >= abs(dy) and abs(dx) > 2
    controller = Room33ScoopController(ROOM_33_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    controller.max_live_enemies = 3
    ram = _room_ram(room=0x33, x=88, y=165, keys=1)
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_ROOM_ALL_DEAD] = 24
    action = controller.step(read_snapshot(ram))
    assert action.reason == "scoop_key_tile"
    assert np.array_equal(action.action, nes_action("DOWN"))
    assert not np.array_equal(action.action, nes_action("RIGHT"))


def test_room33_key_tile_walk_replans_around_blocked_cell() -> None:
    """Naive 4-way RIGHT into a blocked cell; occupancy BFS goes around."""
    controller = Room33ScoopController(ROOM_33_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    controller.max_live_enemies = 3
    # Same row as the key tile: naive cardinal is RIGHT onto (89,173).
    controller.walker.grid.blocked.add((89, 173))
    ram = _room_ram(room=0x33, x=88, y=173, keys=1)
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_ROOM_ALL_DEAD] = 24
    action = controller.step(read_snapshot(ram))
    assert action.reason == "scoop_key_tile"
    assert not np.array_equal(action.action, nes_action("RIGHT"))
    assert not np.array_equal(action.action, nes_idle_action())
    assert np.array_equal(action.action, nes_action("DOWN")) or np.array_equal(
        action.action, nes_action("UP")
    )


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


def test_survival_room45_enters_east_door_from_clear44_leftover() -> None:
    """exp5 leftover (192, 149): north-band hop LEFT into the east statues."""
    assert ROOM_45_SURVIVAL_SPEC.entry.waypoints[0] == (192, 165)
    assert ROOM_45_SURVIVAL_SPEC.entry.y_first is True
    controller = GenericDungeonRoomController(ROOM_45_SURVIVAL_SPEC)
    action = controller.step(read_snapshot(_room_ram(room=0x44, x=192, y=149)))
    assert action.reason == "entry_route"
    assert np.array_equal(action.action, nes_action("DOWN"))
    assert not np.array_equal(action.action, nes_action("LEFT"))


def test_scoop_heart_skips_rupee_state_on_shared_type() -> None:
    """Floor drops share ObjType 0x60; rupee ObjState is not a heart."""
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    ram = _room_ram(room=0x54, x=120, y=141)
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_OBJ_TYPE + 1] = 0x60
    ram[ADDR_OBJ_STATE + 1] = 0x18
    ram[ADDR_LINK_X + 1] = 136
    ram[ADDR_LINK_Y + 1] = 141
    assert controller._scoop_heart(read_snapshot(ram)) is None


def _heart_drop_ram(
    *,
    room: int,
    x: int = 120,
    y: int = 141,
    health: int = 0x21,
    drop_x: int = 136,
    drop_y: int = 141,
    all_dead: int = 20,
) -> np.ndarray:
    ram = _room_ram(room=room, x=x, y=y)
    ram[ADDR_HEALTH] = health
    ram[ADDR_ROOM_ALL_DEAD] = all_dead
    ram[ADDR_OBJ_TYPE + 1] = int(HEART_DROP_OBJECT_TYPE)
    ram[ADDR_OBJ_STATE + 1] = int(HEART_DROP_STATE)
    ram[ADDR_LINK_X + 1] = drop_x
    ram[ADDR_LINK_Y + 1] = drop_y
    return ram


def test_cleared_room_scoops_nearby_heart_not_done() -> None:
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.max_live_enemies = ROOM_54_SPEC.expected_enemy_count
    ram = _heart_drop_ram(room=0x54)
    action = controller.step(read_snapshot(ram))
    assert action.reason == "scoop_heart"
    assert np.array_equal(action.action, nes_action("RIGHT"))
    assert controller.success is False
    assert controller.phase is DungeonPhase.FIGHT


def test_full_health_skips_heart_scoop() -> None:
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.max_live_enemies = ROOM_54_SPEC.expected_enemy_count
    ram = _heart_drop_ram(room=0x54, health=0x22)
    snap = read_snapshot(ram)
    assert snap.health_is_full
    assert controller._scoop_heart(snap) is None
    action = controller.step(snap)
    assert action.reason == "done"
    assert controller.success is True


def test_contact_enemy_fights_instead_of_scoop() -> None:
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x54,
        x=120,
        y=141,
        enemy_type=0x1B,
        enemies=1,
        hp=0,
        enemy_x=120 + 8,
        enemy_y=141,
    )
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_OBJ_TYPE + 2] = int(HEART_DROP_OBJECT_TYPE)
    ram[ADDR_OBJ_STATE + 2] = int(HEART_DROP_STATE)
    ram[ADDR_LINK_X + 2] = 136
    ram[ADDR_LINK_Y + 2] = 141
    action = controller.step(read_snapshot(ram))
    assert action.reason.startswith("combat_")
    assert action.reason != "scoop_heart"


def test_collect_reward_scoops_heart_before_key() -> None:
    controller = GenericDungeonRoomController(ROOM_72_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    ram = _heart_drop_ram(room=0x72)
    action = controller.step(read_snapshot(ram))
    assert action.reason == "scoop_heart"
    assert "collect" not in action.reason
    assert controller.success is False
