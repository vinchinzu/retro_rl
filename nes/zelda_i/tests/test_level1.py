from __future__ import annotations

import numpy as np

from zelda_i.level1 import (
    FIRST_KEY_ITEM_ID,
    ROOM_ENTRANCE,
    ROOM_FIRST_KEY,
    ROOM_KEY_STALFOS,
    ROOM_NORTH_STALFOS,
    STALFOS_OBJECT_TYPE,
    Level1Clear63Controller,
    Level1Clear63Phase,
    Level1Clear53Controller,
    Level1Clear53Phase,
    Level1FirstKeyController,
    Level1KeyPhase,
    Level1NorthPhase,
    Level1UnlockNorthController,
    level1_first_key_success,
    level1_north_room_success,
    level1_room_63_cleared,
    level1_room_53_cleared,
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
    ADDR_ROOM_ITEM_ID,
    ADDR_ROOM_OBJ_COUNT,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    room: int = ROOM_ENTRANCE,
    keys: int = 0,
    x: int = 120,
    y: int = 205,
    stalfos: int = 0,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 1
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_KEYS] = keys
    ram[ADDR_ROOM_ITEM_ID] = FIRST_KEY_ITEM_ID
    ram[ADDR_ROOM_OBJ_COUNT] = stalfos
    for slot in range(1, stalfos + 1):
        ram[ADDR_OBJ_TYPE + slot] = STALFOS_OBJECT_TYPE
        ram[ADDR_OBJ_HP + slot] = 0x20
        ram[ADDR_LINK_X + slot] = 48 + slot * 16
        ram[ADDR_LINK_Y + slot] = 109 + slot * 8
    return ram


def test_snapshot_reads_dungeon_objects() -> None:
    snap = read_snapshot(_ram(room=ROOM_FIRST_KEY, stalfos=5))
    live = [
        obj
        for obj in snap.objects
        if obj.type_id == STALFOS_OBJECT_TYPE and obj.hp > 0
    ]
    assert snap.room_item_id == FIRST_KEY_ITEM_ID
    assert snap.room_obj_count == 5
    assert len(live) == 5


def test_controller_routes_entrance_to_east_door() -> None:
    controller = Level1FirstKeyController()
    action = controller.step(read_snapshot(_ram()))
    assert controller.phase is Level1KeyPhase.APPROACH_EAST
    assert action.reason.startswith("entry_east")


def test_controller_detects_key_room_combat() -> None:
    controller = Level1FirstKeyController()
    action = controller.step(
        read_snapshot(_ram(room=ROOM_FIRST_KEY, x=16, y=141, stalfos=5))
    )
    assert controller.phase is Level1KeyPhase.FIGHT_KEY_CARRIER
    assert action.reason.startswith("key_room_patrol")


def test_controller_collects_after_room_clear() -> None:
    controller = Level1FirstKeyController(
        phase=Level1KeyPhase.FIGHT_KEY_CARRIER,
        phase_frames=61,
    )
    snap = read_snapshot(_ram(room=ROOM_FIRST_KEY, x=112, y=168, stalfos=5))
    cleared_ram = _ram(room=ROOM_FIRST_KEY, x=112, y=168, stalfos=5)
    cleared_ram[ADDR_OBJ_TYPE + 1] = 0
    cleared_ram[ADDR_OBJ_HP + 1] = 0
    cleared_ram[ADDR_LINK_X + 1] = 107
    cleared_ram[ADDR_LINK_Y + 1] = 189
    controller.step(snap)
    controller.phase_frames = 61
    action = controller.step(read_snapshot(cleared_ram))
    assert controller.phase is Level1KeyPhase.COLLECT_KEY
    assert action.reason == "collect_key"


def test_controller_and_predicate_detect_first_key() -> None:
    controller = Level1FirstKeyController()
    controller.step(read_snapshot(_ram(room=ROOM_FIRST_KEY, keys=0)))
    ram = _ram(room=ROOM_FIRST_KEY, keys=1)
    action = controller.step(read_snapshot(ram))
    assert controller.success is True
    assert controller.phase is Level1KeyPhase.DONE
    assert action.reason == "done"
    assert level1_first_key_success(ram)


def test_unlock_controller_returns_west() -> None:
    controller = Level1UnlockNorthController()
    action = controller.step(
        read_snapshot(_ram(room=ROOM_FIRST_KEY, keys=1, x=184, y=173))
    )
    assert controller.phase is Level1NorthPhase.RETURN_WEST
    assert action.reason.startswith("return_west")


def test_unlock_controller_routes_north_from_entrance() -> None:
    controller = Level1UnlockNorthController()
    controller.step(
        read_snapshot(_ram(room=ROOM_FIRST_KEY, keys=1, x=184, y=173))
    )
    action = controller.step(
        read_snapshot(_ram(room=ROOM_ENTRANCE, keys=1, x=224, y=141))
    )
    assert controller.phase is Level1NorthPhase.ROUTE_NORTH
    assert action.reason.startswith("route_north")


def test_unlock_controller_and_predicate_detect_north_room() -> None:
    controller = Level1UnlockNorthController()
    controller.step(
        read_snapshot(_ram(room=ROOM_FIRST_KEY, keys=1, x=184, y=173))
    )
    ram = _ram(room=ROOM_NORTH_STALFOS, keys=0, x=120, y=165, stalfos=3)
    for _ in range(30):
        action = controller.step(read_snapshot(ram))
    assert controller.success is True
    assert controller.phase is Level1NorthPhase.DONE
    assert action.reason == "done"
    assert level1_north_room_success(ram)


def test_clear63_controller_engages_nearby_stalfos() -> None:
    controller = Level1Clear63Controller()
    ram = _ram(room=ROOM_NORTH_STALFOS, x=120, y=165, stalfos=3)
    # Place one Stalfos adjacent so engage beats patrol.
    ram[ADDR_LINK_X + 1] = 128
    ram[ADDR_LINK_Y + 1] = 165
    action = controller.step(read_snapshot(ram))
    assert controller.phase is Level1Clear63Phase.FIGHT
    assert action.reason.startswith("clear_engage")
    assert controller.last_live_stalfos == 3


def test_clear63_controller_and_predicate_detect_room_clear() -> None:
    controller = Level1Clear63Controller()
    # Seed so the controller has observed enemies before clear settle.
    controller.step(
        read_snapshot(_ram(room=ROOM_NORTH_STALFOS, x=120, y=165, stalfos=3))
    )
    ram = _ram(room=ROOM_NORTH_STALFOS, x=120, y=165, stalfos=0)
    ram[ADDR_ROOM_ALL_DEAD] = 24
    action = None
    for _ in range(8):
        action = controller.step(read_snapshot(ram))
    assert controller.success is True
    assert controller.phase is Level1Clear63Phase.DONE
    assert action is not None and action.reason == "done"
    assert level1_room_63_cleared(ram)


def test_clear63_predicate_requires_settle() -> None:
    ram = _ram(room=ROOM_NORTH_STALFOS, x=120, y=165, stalfos=0)
    ram[ADDR_ROOM_ALL_DEAD] = 5
    assert not level1_room_63_cleared(ram)
    ram[ADDR_ROOM_ALL_DEAD] = 20
    assert level1_room_63_cleared(ram)


def test_clear53_controller_routes_around_room63_blocks() -> None:
    controller = Level1Clear53Controller()
    action = controller.step(
        read_snapshot(_ram(room=ROOM_NORTH_STALFOS, x=72, y=125, stalfos=0))
    )
    assert controller.phase is Level1Clear53Phase.ROUTE_NORTH
    assert action.reason == "route_room53"


def test_clear53_controller_fights_then_targets_fixed_key() -> None:
    controller = Level1Clear53Controller(
        phase=Level1Clear53Phase.FIGHT,
        initial_keys=0,
    )
    live_ram = _ram(room=ROOM_KEY_STALFOS, x=120, y=205, stalfos=5)
    action = controller.step(read_snapshot(live_ram))
    assert action.reason.startswith("room53_clear_")
    assert controller.max_live_stalfos == 5

    cleared_ram = _ram(room=ROOM_KEY_STALFOS, x=88, y=141, stalfos=0)
    cleared_ram[ADDR_ROOM_ALL_DEAD] = 24
    action = controller.step(read_snapshot(cleared_ram))
    assert controller.phase is Level1Clear53Phase.COLLECT_KEY
    assert controller.clear_signal_seen is True
    assert action.reason == "collect_room53_key"


def test_clear53_controller_and_predicate_require_collected_key() -> None:
    controller = Level1Clear53Controller(
        phase=Level1Clear53Phase.COLLECT_KEY,
        initial_keys=0,
        max_live_stalfos=5,
    )
    ram = _ram(room=ROOM_KEY_STALFOS, keys=0, x=120, y=109, stalfos=0)
    ram[ADDR_ROOM_ALL_DEAD] = 83
    assert not level1_room_53_cleared(ram)

    ram[ADDR_KEYS] = 1
    action = controller.step(read_snapshot(ram))
    assert controller.success is True
    assert controller.phase is Level1Clear53Phase.DONE
    assert action.reason == "done"
    assert level1_room_53_cleared(ram)


def test_room45_clean_collect_uses_east_column() -> None:
    from zelda_i.level1_dungeon import ROOM_45_SPEC, ROOM_45_SURVIVAL_SPEC

    waypoints = ROOM_45_SPEC.reward.waypoints
    assert waypoints[0] == (160, 141)
    assert (152, 189) in waypoints
    assert ROOM_45_SPEC.reward.reward_while_live is False
    assert ROOM_45_SURVIVAL_SPEC.combat.avoid_walls is True
    assert ROOM_45_SURVIVAL_SPEC.reward.waypoints == waypoints


def test_l1_complete_assisted_paths_do_not_clobber_clean() -> None:
    from zelda_i.scripts.run_level1_complete import (
        _intro_summary,
        default_report_path,
        default_video_path,
    )

    clean_video = default_video_path(natural_entry=True)
    assisted_video = default_video_path(natural_entry=True, infinite_life=True)
    assert clean_video.name == "level1_complete_natural.mp4"
    assert assisted_video.name == "level1_complete_natural_assisted.mp4"
    assert default_report_path(natural_entry=True).name == (
        "level1_complete_natural.json"
    )
    assert default_report_path(natural_entry=True, infinite_life=True).name == (
        "level1_complete_natural_assisted.json"
    )
    assert "Clean" in _intro_summary(natural_entry=True, infinite_life=False)
    assert "Survival" in _intro_summary(natural_entry=True, infinite_life=True)
