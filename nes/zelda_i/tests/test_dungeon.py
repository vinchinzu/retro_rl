from __future__ import annotations

from dataclasses import replace

import numpy as np

from retro_harness.controls import pressed_nes_buttons
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import (
    AliveRule,
    DungeonPhase,
    GenericDungeonRoomController,
    GORIYA_OBJECT_TYPE,
    RewardKind,
    ROUTE_STALL_FRAMES,
)

from zelda_i.dungeon.ids import HEART_DROP_OBJECT_TYPE, HEART_DROP_STATE
from zelda_i.level1.dungeon import (
    ROOM_23_SPEC,
    ROOM_33_SPEC,
    ROOM_45_SPEC,
    ROOM_53_SPEC,
    ROOM_54_SPEC,
    ROOM_72_SPEC,
    Room33ScoopController,
)
from zelda_i.level1.east_dungeon import (
    ROOM_45_SURVIVAL_SPEC,
)
from zelda_i.level1.path import level1_room_72_key_success
from zelda_i.tests.ram_helpers import room_tile_env
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
    for _ in range(7):
        assert pressed_nes_buttons(list(controller.step(snap).action)) == ["RIGHT"]


def test_engage_enemy_in_sword_hitbox_slashes() -> None:
    """Enemy in blade rectangle → the shared contact rung owns the swing."""
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
    ram[ADDR_LINK_FACING] = 0x01
    snap = read_snapshot(ram)
    action = controller.step(snap)
    assert action.reason == "combat_contact_strike"
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
        pressed = pressed_nes_buttons(list(controller.step(snap).action))
        assert pressed and "A" not in pressed


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
        enemy_x=x + 32,
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
    """A Wallmaster parked in the west wall is never walked at.

    It sits at x=0, outside the room; stepping LEFT toward it is how Link
    ends up in the door mouth, which is where a grab drags him out of the
    room entirely. Any answer that keeps him off that column is fine —
    the off-wall step included.
    """
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
    # Walk out the bounded entry dash first; it exists to clear the west
    # door mouth, which is the one cell a Wallmaster grab drags Link from.
    snap = read_snapshot(ram)
    for _ in range(ROOM_45_SPEC.combat.inland_dash + 8):
        action = controller.step(snap)
        assert not np.array_equal(action.action, nes_action("LEFT"))
        assert not np.array_equal(action.action, nes_action("LEFT", "A"))
        assert "engage" not in action.reason


def test_occupancy_room_does_not_leave_wall_in_south_pocket() -> None:
    """0x23 south door: occupancy chases, avoid_walls does not mash UP."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.bind_env(room_tile_env("0x23"))
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
    """West 16px column and south corridor are floor; $6530 water bar is blocked.

    Measured from the captured room map, not from a hand-written box list —
    the list this replaces walled 84 cells of real floor.
    """
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.bind_env(room_tile_env("0x23"))
    controller._set_phase(DungeonPhase.FIGHT)
    blocked = controller.walker.grid.blocked
    assert (78, 157) not in blocked
    assert (144, 149) not in blocked
    assert (64, 141) not in blocked
    assert (88, 149) not in blocked
    assert (120, 136) in blocked
    assert (88, 148) in blocked
    walker = controller.walker
    # Live leftover (88,149): around west, never UP into the bar.
    assert walker.next_dir((88, 149), (148, 125)) == "LEFT"


def test_room23_stands_when_boomerang_blocks_the_step() -> None:
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.bind_env(room_tile_env("0x23"))
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


def test_room23_never_answers_a_closing_body_with_an_idle_frame() -> None:
    """The pose that killed the Clean L1 run: idle at (64,157), Goriya at (64,149).

    The old low-health tactic masked UP whenever ``y <= 157`` and substituted
    an idle frame. The planner wants UP essentially always in this room, so
    the mask became a permanent hold: 2278 of 2687 stage frames were idle
    while a Goriya walked down the x=64 column into a stationary Link, who
    took five hits and died.

    The rule this pins is not "prefer UP" — it is that a live body 8px away
    and closing must never be answered by doing nothing. A sword, a step or
    a peel are all acceptable; standing there is not.
    """
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.bind_env(room_tile_env("0x23"))
    controller._set_phase(DungeonPhase.FIGHT)
    for _ in range(8):
        ram = _room_ram(
            room=0x23,
            x=64,
            y=157,
            enemy_type=0x06,
            enemies=1,
            hp=0x30,
            enemy_x=64,
            enemy_y=149,
        )
        action = controller.step(read_snapshot(ram))
        assert not np.array_equal(action.action, nes_idle_action()), (
            f"idled at (64,157) with a Goriya at (64,149): {action.reason}"
        )


def test_room23_occupancy_stands_on_goriya_instead_of_walking() -> None:
    """No occupancy path (already on target): stand/slash, do not chase."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.bind_env(room_tile_env("0x23"))
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
        "combat_evade_peel",
    )


def test_room23_far_corridor_chase_closes_without_slashing() -> None:
    """56px south-corridor Goriya: close on it, but do not swing at thin air.

    The hunt radius is 80 now that the walker knows the real walls, so this
    pose engages rather than patrols — what must stay true is that Link does
    not burn the sword outside its hitbox.
    """
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.bind_env(room_tile_env("0x23"))
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
    assert "_slash" not in action.reason
    assert controller.swings == 0
    assert controller.report()["damage"]["swings"] == 0


def test_room23_hitbox_close_answers_with_the_sword() -> None:
    """Inside the sword box the answer is the blade, not a peel."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.bind_env(room_tile_env("0x23"))
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
    # Blade range: answer with the sword, not the feet. Peeling here concedes
    # the hit and lengthens the fight (0x23: 2005 patrol frames, four swings).
    assert action.reason in ("combat_parry", "combat_parry_face", "combat_backstep")


def test_room23_across_water_does_not_slash_over_the_bar() -> None:
    """32px north across the water bar: walk around it, never swing across."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.bind_env(room_tile_env("0x23"))
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
    controller.bind_env(room_tile_env("0x23"))
    controller._set_phase(DungeonPhase.FIGHT)
    action = controller.step(read_snapshot(ram))
    assert "_slash" not in action.reason
    assert controller.swings == 0
    assert controller.swings_authorized == 0


def test_room23_leftover_chase_goes_west_not_up_into_water() -> None:
    """(88,149) leftover: occupancy LEFT around the $6530 bar, never UP."""
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.bind_env(room_tile_env("0x23"))
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
    controller.bind_env(room_tile_env("0x23"))
    controller._set_phase(DungeonPhase.FIGHT)
    action = controller.step(read_snapshot(ram))
    assert np.array_equal(action.action, nes_action("LEFT"))
    assert not np.array_equal(action.action, nes_action("UP"))
    assert not np.array_equal(action.action, nes_action("UP", "A"))
    assert controller.swings == 0


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
    controller.combat_frames = 4

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

    # Frame 2: Link's real UP step
    # landed; the target darts close (distance 10 < contact_backstep 16).
    snap = snap_with((120, 160), (120, 150))
    action = controller._combat(snap, spec.live_enemies(snap))
    assert action.reason == "combat_contact_peel"
    assert controller.walker.misses == 0

    # Frame 3: Link's real DOWN peel landed; a sword turn now owns the frame.
    snap = snap_with((120, 161), (120, 150))
    action = controller._combat(snap, spec.live_enemies(snap))
    assert action.reason == "combat_contact_turn"
    assert controller.walker.misses == 0

    # Frame 4: Link's real UP turn landed; target retreats, occupancy resumes.
    snap = snap_with((120, 160), (120, 93))
    controller._combat(snap, spec.live_enemies(snap))
    assert controller.walker.misses == 0
    assert not controller.walker.grid.blocked


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


def test_scoop_heart_reaches_drop_pixel_the_grid_calls_solid() -> None:
    """0x23 real geometry: a drop drawn at (57, 104) reads solid at raw (x, y).

    The drop's stored (x, y) is where it is DRAWN; Link collides
    ``LINK_FOOT_OFFSET`` px lower (tilemap.py), so the occupancy grid -- built
    from the same measured ``$6530`` map every other 0x23 test in this file
    trusts -- calls the drop's own pixel solid even though it plainly rests
    on real floor (measured: BFS from (120, 157) to the raw pixel is None;
    to (57, 100), 4px away and inside the reach diamond, it is 155 steps).
    ``_scoop_heart`` must find that nearby standable cell instead of idling
    forever on an exact pixel the grid will never call passable.
    """
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.bind_env(room_tile_env("0x23"))
    ram = _room_ram(room=0x23, x=120, y=157)
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_OBJ_TYPE + 1] = int(HEART_DROP_OBJECT_TYPE)
    ram[ADDR_OBJ_STATE + 1] = int(HEART_DROP_STATE)
    ram[ADDR_LINK_X + 1] = 57
    ram[ADDR_LINK_Y + 1] = 104
    snap = read_snapshot(ram)
    assert not controller.walker.grid.passable(57, 104), (
        "fixture assumption stale: the raw drop pixel is no longer solid"
    )
    action = controller.step(snap)
    assert action.reason == "scoop_heart"
    assert not np.array_equal(action.action, nes_idle_action()), (
        "idled on an exact drop pixel the grid will never call passable"
    )


def test_scoop_heart_unreachable_drop_yields_instead_of_idling() -> None:
    """A sealed-off Link must yield the frame, and not re-flood the BFS.

    Measured cause of the L1 0x45 death (rr coordinator trace): a live body
    sealed the only column to an otherwise-reachable heart, ``next_dir``
    returned None, and the old code answered with an idle frame forever --
    pinning Link while a Wallmaster walked into him, and separately turning
    one 27s ledger run into 2m48s by re-running the reverse-flood BFS every
    frame. ``_scoop_heart`` must return None (so the room policy drives) and
    must not repeat that full BFS on every single frame while stuck.
    """
    controller = GenericDungeonRoomController(ROOM_23_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.bind_env(room_tile_env("0x23"))
    ram = _room_ram(room=0x23, x=120, y=157)
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_OBJ_TYPE + 1] = int(HEART_DROP_OBJECT_TYPE)
    ram[ADDR_OBJ_STATE + 1] = int(HEART_DROP_STATE)
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 181
    snap = read_snapshot(ram)
    # Box Link in on all four sides: nothing is reachable from here, drop
    # included, however generous the standable-cell search is.
    lx, ly = int(snap.link_x), int(snap.link_y)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        controller.walker.grid.blocked.add((lx + dx, ly + dy))

    from zelda_i.walk.physics import OccupancyGrid

    calls = {"n": 0}
    orig_shortest_path = OccupancyGrid.shortest_path

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return orig_shortest_path(self, *args, **kwargs)

    OccupancyGrid.shortest_path = counting
    try:
        for _ in range(40):
            action = controller.step(snap)
            assert not (
                action.reason == "scoop_heart"
                and np.array_equal(action.action, nes_idle_action())
            ), "idled on an unreachable drop instead of yielding the frame"
    finally:
        OccupancyGrid.shortest_path = orig_shortest_path
    # A flood is O(grid cells) per shortest_path call; 40 unthrottled frames
    # measured 2-3 calls each (retry + forget-and-retry). Bounding the count
    # far below that is the cached-unreachable verdict, not incidental luck.
    assert calls["n"] < 15, f"BFS re-ran {calls['n']}x over 40 stuck frames"


def test_fixed_inventory_collect_rebuilds_combat_scars() -> None:
    """Key hunt must not inherit inferred blocks from the fight.

    L1 0x45: after Wallmasters died, collect sat at (144, 141) for 7666
    frames. Combat observe() had boxed the east column; CLEAR_ONLY leftover
    already calls ``_relax_leftover_bounds``, FIXED_INVENTORY did not.
    """
    controller = GenericDungeonRoomController(ROOM_45_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.max_live_enemies = ROOM_45_SPEC.expected_enemy_count
    controller.initial_inventory = 0
    controller.walker.grid.inferred.add((145, 141))
    controller.walker.grid.blocked.add((145, 141))
    ram = _room_ram(room=0x45, x=144, y=141)
    ram[ADDR_ROOM_ALL_DEAD] = 30
    controller.step(read_snapshot(ram))
    assert controller.phase is DungeonPhase.COLLECT_REWARD
    assert (145, 141) not in controller.walker.grid.inferred
    assert (145, 141) not in controller.walker.grid.blocked


def test_collect_skips_waypoint_when_manhattan_stalls() -> None:
    """A 3px y-loop never trips in-place stuck, so collect must skip on stale dist.

    L1 0x45 collect sat at (144, 141) aiming at (160, 141) for 7666 frames.
    """
    from zelda_i.dungeon.engine import _COLLECT_STALE

    controller = GenericDungeonRoomController(ROOM_45_SPEC)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = 0
    ram = _room_ram(room=0x45, x=144, y=141)
    snap = read_snapshot(ram)
    for _ in range(_COLLECT_STALE + 1):
        controller.step(snap)
    assert controller.waypoint_index >= 1


def _stalled_route_controller(spec, *, x: int, y: int, source_room: int):
    """Step ``spec``'s entry route from a pose that never moves."""
    controller = GenericDungeonRoomController(spec)
    ram = _room_ram(room=source_room, x=x, y=y)
    snap = read_snapshot(ram)
    actions = []
    for _ in range(ROUTE_STALL_FRAMES + 4):
        actions.append(controller.step(snap))
    return controller, actions


def test_entry_route_holds_one_button_until_the_stall_threshold() -> None:
    """Below the threshold the route is unchanged — greens must not move."""
    controller = GenericDungeonRoomController(ROOM_45_SURVIVAL_SPEC)
    snap = read_snapshot(_room_ram(room=0x44, x=168, y=141))
    reasons = [controller.step(snap).reason for _ in range(ROUTE_STALL_FRAMES)]
    assert set(reasons) == {"entry_route"}
    assert not controller.notes


def test_entry_route_stall_is_noted_with_the_pose_and_leg() -> None:
    """A walled entry route used to leave only ``timeout`` behind."""
    controller, _ = _stalled_route_controller(
        ROOM_45_SURVIVAL_SPEC, x=168, y=141, source_room=0x44
    )
    stall = [n for n in controller.notes if n.startswith("entry_route_stall_")]
    assert len(stall) == 1
    assert "(168, 141)" in stall[0]
    assert str(ROOM_45_SURVIVAL_SPEC.entry.waypoints[0]) in stall[0]


def test_entry_route_skips_a_walled_leg_without_measured_geometry() -> None:
    """No bound env means no tile map: drop the leg rather than hold a wall."""
    controller, actions = _stalled_route_controller(
        ROOM_45_SURVIVAL_SPEC, x=168, y=141, source_room=0x44
    )
    assert any(a.reason == "entry_route_skip" for a in actions)
    assert controller.waypoint_index == 1


def test_entry_route_replans_around_measured_walls() -> None:
    """With the live map bound, the stall replans instead of skipping."""
    from zelda_i.tests.ram_helpers import tile_map_env

    controller = GenericDungeonRoomController(ROOM_45_SURVIVAL_SPEC)
    # Two solid cells south of Link, as the real 0x44 statues at (176,160)
    # and (176,128) are for the (168,141) leftover: DOWN is walled but the
    # waypoint is still reachable the long way round.
    controller.bind_env(tile_map_env({(160, 160), (176, 160)}))
    ram = _room_ram(room=0x44, x=168, y=141)
    snap = read_snapshot(ram)
    reasons = []
    for _ in range(ROUTE_STALL_FRAMES + 4):
        reasons.append(controller.step(snap).reason)
    assert {"entry_route_replan", "entry_route_lattice"} & set(reasons)
    assert "entry_route_skip" not in reasons
    assert controller.waypoint_index == 0


def test_entry_route_stall_counter_resets_when_link_moves() -> None:
    controller = GenericDungeonRoomController(ROOM_45_SURVIVAL_SPEC)
    for step, y in enumerate(range(133, 133 + ROUTE_STALL_FRAMES + 4)):
        controller.step(read_snapshot(_room_ram(room=0x44, x=168, y=y)))
        assert controller._route_stall_frames == 0, step
    assert not controller.notes


def test_entry_route_replan_latches_until_the_leg_advances() -> None:
    """One replanned step then back to the axis rule re-walls the same wall."""
    from zelda_i.tests.ram_helpers import tile_map_env

    controller = GenericDungeonRoomController(ROOM_45_SURVIVAL_SPEC)
    controller.bind_env(tile_map_env({(160, 160), (176, 160)}))
    snap = read_snapshot(_room_ram(room=0x44, x=168, y=141))
    for _ in range(ROUTE_STALL_FRAMES + 1):
        controller.step(snap)
    assert controller._route_replanning is True
    # Link moves a pixel east; the stall counter resets but the leg stays
    # on the walker, because the axis rule is what walled it.
    moved = read_snapshot(_room_ram(room=0x44, x=169, y=141))
    assert controller.step(moved).reason in ("entry_route_replan", "entry_route_lattice")
    # Reaching the leg hands it back to the cheap axis rule.
    arrived = read_snapshot(_room_ram(room=0x44, x=192, y=165))
    controller.step(arrived)
    assert controller._route_replanning is False
    assert controller.waypoint_index == 1


def _at_reward(spec, *, x: int, y: int, room: int, inventory: int = 3):
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.COLLECT_REWARD
    controller.initial_inventory = inventory
    controller.clear_signal_seen = True
    ram = _room_ram(room=room, x=x, y=y, keys=inventory)
    ram[ADDR_LEVEL] = spec.level
    ram[ADDR_ROOM_ALL_DEAD] = 24
    return controller, read_snapshot(ram)


def test_reward_idle_is_unchanged_below_the_nudge_threshold() -> None:
    from zelda_i.dungeon.engine import REWARD_NUDGE_FRAMES
    from zelda_i.level2.spine import ROOM_7E_SPINE_SPEC

    controller, snap = _at_reward(ROOM_7E_SPINE_SPEC, x=138, y=141, room=0x7E)
    reasons = {
        controller.step(snap).reason for _ in range(REWARD_NUDGE_FRAMES - 1)
    }
    assert reasons <= {"reward_wait", "collect_wait", "collect_reward"}
    assert not any(r.startswith("reward_nudge") for r in reasons)


def test_reward_nudge_closes_the_last_pixels_to_the_pickup() -> None:
    """L2 0x7e idled 6752f at (138,141) with the key 2px west at (136,141)."""
    from zelda_i.dungeon.engine import REWARD_NUDGE_FRAMES
    from zelda_i.level2.spine import ROOM_7E_SPINE_SPEC

    controller, snap = _at_reward(ROOM_7E_SPINE_SPEC, x=138, y=141, room=0x7E)
    action = None
    for _ in range(REWARD_NUDGE_FRAMES + 2):
        action = controller.step(snap)
    assert action is not None
    assert action.reason == "reward_nudge"
    assert np.array_equal(action.action, nes_action("LEFT"))


def test_reward_nudge_wiggles_when_already_on_the_tile() -> None:
    from zelda_i.dungeon.engine import REWARD_NUDGE_FRAMES
    from zelda_i.level2.spine import ROOM_7E_SPINE_SPEC

    controller, snap = _at_reward(ROOM_7E_SPINE_SPEC, x=136, y=141, room=0x7E)
    reasons = [
        controller.step(snap).reason for _ in range(REWARD_NUDGE_FRAMES * 2)
    ]
    assert "reward_nudge_wiggle" in reasons


def test_reward_idle_counter_resets_when_the_collect_walk_moves() -> None:
    from zelda_i.level2.spine import ROOM_7E_SPINE_SPEC

    controller, far = _at_reward(ROOM_7E_SPINE_SPEC, x=60, y=141, room=0x7E)
    for _ in range(40):
        action = controller.step(far)
        assert action.reason != "reward_nudge"
    assert controller._reward_idle_frames == 0


def test_reward_waypoints_only_drive_the_walk_under_occupancy_patrol() -> None:
    """Pins a known gap so it cannot change without someone noticing.

    ``_collect_policy`` walks ``reward.waypoints`` only when
    ``combat.occupancy_patrol`` is on. Without it the waypoint block still
    runs its reached / stuck / stale bookkeeping and then falls through to
    ``reward.target``, so the hunt pattern those specs carry is never walked —
    the specs below each ship a list the engine ignores. That is a real gap
    (L6 0x7a is one; L5 0x77 left the gap when it moved to occupancy),
    but closing it moves rooms that are green today, so it is recorded here
    rather than changed in passing. ``_reward_nudge`` handles the case this
    actually cost a run: idling 2px off the target.
    """
    from zelda_i.dungeon.engine import _ROOM_SPECS_BY_LEVEL, ensure_default_specs

    ensure_default_specs()
    unwalked = {
        spec.spec_id
        for spec in _ROOM_SPECS_BY_LEVEL.values()
        if spec.reward.waypoints and not spec.combat.occupancy_patrol
    }
    # A superset check: the registry is global and other level modules add
    # rooms depending on import order. These are the ones measured today.
    assert {
        "level1_room72",
        "level2_room3e_moldorm_key",
        "level2_room6f_compass",
        "level4_room51_keese_key",
        "level6_room7a_east_key",
    } <= unwalked
    # Rooms that do walk their hunt pattern must stay out of the gap.
    assert "level1_room23" not in unwalked
    assert "level1_room45" not in unwalked
    assert "level4_room40_zols_key" not in unwalked
    assert "level5_room77_pols_voice" not in unwalked


def _recording_evader(controller):
    """Wrap ``controller.evader.decide`` so a test can see what it advised."""
    decisions = []
    inner = controller.evader.decide

    def _decide(*args, **kwargs):
        decision = inner(*args, **kwargs)
        decisions.append(decision)
        return decision

    controller.evader.decide = _decide
    return decisions


def test_inland_dash_survives_a_stand_decision_at_the_0x45_door_mouth() -> None:
    """A reactive STAND must fall through to the positional rules, not idle.

    Entry from 0x44 lands Link at ``x=16``, inside the west door mouth — the
    one cell a Wallmaster grab drags him out of the room, which is the whole
    reason ``ROOM_45_SPEC`` carries ``inland_dash=56``. The evader's bounds
    there are ``avoid_wall_bounds`` ``(56, 200, 109, 173)``, so every step
    from ``x=16`` fails ``_can_move`` and ``decide`` returns ``evade_boxed_in``
    with ``direction=None`` on every frame a body is inside ``trigger_ttc``.

    Consuming that frame (idling on the stand) cancels the dash for exactly
    as long as something is inbound, which is exactly when it is needed. The
    contract is ``overworld.path._threat_action``'s: a step decision and a
    ``shield_hold`` are answers, a bare stand hands the frame back.

    Reactive still runs *first* — this is not the fixed root cause in
    reverse. A pose rule returning before ``threat.decide`` is the dominant
    damage bug in this tree; what changed is only what a no-op advice does.
    """
    controller = GenericDungeonRoomController(ROOM_45_SPEC)
    controller.phase = DungeonPhase.FIGHT
    decisions = _recording_evader(controller)
    reasons = []
    for enemy_x in (72, 64, 56, 48, 40, 32):
        snap = read_snapshot(
            _room_ram(
                room=0x45,
                x=16,
                y=141,
                enemy_type=0x27,
                enemies=1,
                hp=0x20,
                enemy_x=enemy_x,
                enemy_y=141,
            )
        )
        action = controller.step(snap)
        reasons.append(action.reason)
        assert np.array_equal(action.action, nes_action("RIGHT")) or np.array_equal(
            action.action, nes_action("RIGHT", "A")
        ), (action.reason, enemy_x)
        assert not np.array_equal(action.action, nes_idle_action()), action.reason

    # The scenario has to be the one the fix is about: the evader really did
    # advise a bare stand while a body was inside trigger_ttc.
    stands = [d for d in decisions if d is not None and d.direction is None]
    assert stands, decisions
    assert all(not d.shield for d in stands)
    assert {d.reason for d in stands} <= {"evade_boxed_in", "evade_no_gain"}
    # ... and the dash owned every one of those frames.
    assert controller.combat_frames <= ROOM_45_SPEC.combat.inland_dash
    assert all(r.startswith("inland_dash") for r in reasons), reasons


def test_shield_hold_still_consumes_the_frame() -> None:
    """``shield_hold`` is a real answer; only a bare stand falls through.

    Link is already facing a blockable shot, so the idle frame *is* the
    block. Letting the dash steal it would walk him off the shield.
    """
    from zelda_i.dungeon.threat import EvadeDecision

    controller = GenericDungeonRoomController(ROOM_45_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.evader.decide = lambda *a, **k: EvadeDecision(
        direction=None, reason="shield_hold", ttc=4, stand_ttc=4, shield=True
    )
    snap = read_snapshot(
        _room_ram(
            room=0x45,
            x=16,
            y=141,
            enemy_type=0x27,
            enemies=1,
            hp=0x20,
            enemy_x=72,
            enemy_y=141,
        )
    )
    action = controller.step(snap)
    assert action.reason == "combat_shield_hold"
    assert np.array_equal(action.action, nes_idle_action())


def _door_room_ram():
    """A synthetic ``$6530`` map: floor interior, walled ring, bombed holes.

    Hole pairs go at the north (``112, 80``), south (``112, 208``) and west
    (``16, 144``) ring cells with the ids the ROM uses there, measured over
    259 dungeon fixtures: north ``0x8C``/``0x8D``, south ``0x8E``/``0x8F``,
    west ``0x90``/``0x91``. Everything else outside the interior is solid, so
    the only holes in the wall are the three passages.
    """
    from zelda_i.dungeon import tilemap as tm

    ram = np.zeros(tm.WRAM_RAM_OFFSET + 0x2000, dtype=np.uint8)
    tiles = np.zeros((tm.TILE_ROWS, tm.TILE_COLS), dtype=np.uint8)
    floor = np.array([[0x74, 0x76], [0x75, 0x77]], dtype=np.uint8)
    for y in tm.INTERIOR_Y:
        for x in tm.INTERIOR_X:
            col, row = x // tm.TILE_PX, (y - tm.PLAYFIELD_TOP_Y) // tm.TILE_PX
            tiles[row : row + 2, col : col + 2] = floor
    for (x, y), pair in (
        ((112, 80), (0x8C, 0x8D)),
        ((112, 208), (0x8E, 0x8F)),
        ((16, 144), (0x90, 0x91)),
    ):
        col, row = x // tm.TILE_PX, (y - tm.PLAYFIELD_TOP_Y) // tm.TILE_PX
        tiles[row : row + 2, col : col + 2] = np.array(
            [[pair[0], pair[0]], [pair[1], pair[1]]], dtype=np.uint8
        )
    start = tm.WRAM_RAM_OFFSET + tm.ADDR_ROOM_TILE_MAP - tm.WRAM_BASE
    ram[start : start + tm.TILE_COLS * tm.TILE_ROWS] = tiles.T.reshape(-1)

    class _Env:
        def get_ram(self):
            return ram

    return ram, _Env()


def test_fight_grid_never_paths_link_onto_a_door_cell() -> None:
    """A bombed passage is walkable to the entry route and solid to the fight.

    ``tilemap.LINK_WALKABLE_TILES`` grades ``BOMB_HOLE_TILES`` walkable,
    which is right for ``ROUTE_BOUNDS`` — in the rooms that have one, the
    hole is the way out. Inside the room it is not: walking into it leaves
    the room (``tilemap.door_cells``), and the fight box reaches all three.
    ``DEFAULT_BOUNDS`` ``(40, 216, 77, 205)`` covers the north hole row
    (``y=77..84``, feet in tile row 3) and the south hole row
    (``y=197..205``); the L4/L5/L6 ``(16, 216, ...)`` boxes also cover the
    west hole column. Chasing a body parked near one would then scroll Link
    out of the room mid-clear.

    An *open* doorway is not this case and never was: its mouth is plain
    floor, so the fight grid can stand on it too, and what stops the BFS
    walking out is the box bound, not the tile set.
    """
    from zelda_i.dungeon.route_entry import ROUTE_BOUNDS
    from zelda_i.walk.physics import measured_walker

    ram, env = _door_room_ram()
    controller = GenericDungeonRoomController(ROOM_45_SPEC)
    controller.phase = DungeonPhase.FIGHT
    controller.bind_env(env)
    grid = controller.walker.grid

    # Feet collide LINK_FOOT_OFFSET px lower: y=84 -> tile row 3, the lower
    # half of the y=80 north door cell; y=205 -> row 18, the south door.
    north_mouth = (112, 84)
    south_mouth = (112, 205)
    assert north_mouth in grid.blocked
    assert south_mouth in grid.blocked
    assert grid.shortest_path((120, 141), north_mouth) is None
    assert grid.shortest_path((120, 141), south_mouth) is None
    # Real floor is untouched: this is a door rule, not a wall.
    assert (120, 141) not in grid.blocked
    assert grid.shortest_path((120, 141), (176, 109)) is not None

    # The entry route still walks straight in through the same door mouths.
    route = measured_walker(ram, ROUTE_BOUNDS)
    assert route is not None
    assert north_mouth not in route.grid.blocked
    assert (16, 141) not in route.grid.blocked
    assert route.grid.shortest_path((120, 141), north_mouth) is not None


def _walled_enemy_spec(enemy_xy: tuple[int, int]):
    """``ROOM_54_SPEC`` with the enemy's own cell, and its ring, walled off."""
    ex, ey = enemy_xy
    walls = tuple(
        (ex + dx, ey + dy)
        for dx in range(-1, 2)
        for dy in range(-1, 2)
    )
    tuning = replace(
        ROOM_54_SPEC.combat,
        occupancy_patrol=True,
        occupancy_blocked=walls,
        engage_distance=8,
    )
    return replace(ROOM_54_SPEC, combat=tuning)


def test_chase_goal_retargets_an_enemy_standing_on_geometry() -> None:
    """An enemy on a blocked cell must not freeze the chase.

    This is what the ``shortest_path`` goal guard exposed. The guard is
    right -- a goal inside geometry has no path -- but the chase handed the
    resulting ``None`` to ``_engage``, which parried in place. Clean M5 died
    in L1 ``0x23`` on ``f1453 body 0x06 (goriya) from N ... phase=FIGHT``
    with three hits taken standing still.
    """
    from types import SimpleNamespace

    ex, ey = 160, 141
    spec = _walled_enemy_spec((ex, ey))
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT
    ram = _room_ram(
        room=0x54, x=120, y=141, enemy_type=0x1B, enemies=1, hp=0,
        enemy_x=ex, enemy_y=ey,
    )
    controller.step(read_snapshot(ram))

    assert not controller.walker.grid.passable(ex, ey)
    goal = controller._chase_goal(SimpleNamespace(x=ex, y=ey))
    assert goal != (ex, ey)
    assert controller.walker.grid.passable(*goal)
    # And the BFS that stood still now has somewhere to go.
    assert controller.walker.grid.shortest_path((120, 141), (ex, ey)) is None
    assert controller.walker.grid.shortest_path((120, 141), goal) is not None


def test_collect_unreachable_skip_counts_toward_the_lap_guard() -> None:
    """Skipping an unreachable waypoint must end the lap, not restart it.

    ``collect_skip_unreachable`` advanced ``waypoint_index`` without touching
    ``_collect_skips``, so the one-lap guard never tripped: L1 ``0x45`` sat at
    ``(40,157)`` alternating ``collect_skip_2`` and ``collect_skip_3`` for its
    whole 9000-frame budget with the room already cleared and zero hits.
    """
    # Each waypoint sits in a sealed one-cell pocket: the cell itself is
    # passable, so ``nearest_open`` hands it straight back, and there is
    # still no path to it. That is the case the skip exists for.
    waypoints = ((160, 141), (160, 157))
    walls = tuple(
        (wx + dx, wy + dy)
        for wx, wy in waypoints
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    tuning = replace(
        ROOM_54_SPEC.combat, occupancy_patrol=True, occupancy_blocked=walls
    )
    spec = replace(
        ROOM_54_SPEC,
        combat=tuning,
        reward=replace(ROOM_54_SPEC.reward, waypoints=waypoints),
    )
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.COLLECT_REWARD
    ram = _room_ram(room=0x54, x=120, y=141)

    reasons = [controller.step(read_snapshot(ram)).reason for _ in range(400)]
    assert "collect_skip_unreachable" in reasons
    assert "collect_wait" in reasons
    # Once the lap is spent it stays spent. What follows is the ordinary
    # no-waypoint reward policy; what must never follow is a second lap.
    tail = reasons[reasons.index("collect_wait") + 1:]
    assert "collect_skip_unreachable" not in tail
    assert set(tail) <= {"collect_wait", "reward_nudge", "reward_nudge_wiggle"}
    assert controller._collect_skips >= len(waypoints)
