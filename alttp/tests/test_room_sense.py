"""Offline tests for room_sense sprite boxes, maps, and edge detection."""

from __future__ import annotations

import numpy as np

from alttp.primitives import (
    SPRITE_BLUE_SOLDIER,
    SPRITE_HEART,
    SPRITE_STATE,
    SPRITE_TYPE,
    SPRITE_X_HIGH,
    SPRITE_X_LOW,
    SPRITE_Y_HIGH,
    SPRITE_Y_LOW,
    SPRITE_HP,
    active_sprites,
)
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    AlttpSnapshot,
)
from alttp.room_map import KnownDoor, list_room_maps, load_room_map
from alttp.room_sense import (
    SPRITE_KIND_HOSTILE,
    SPRITE_KIND_PICKUP,
    box_for_sprite,
    classify_sprite,
    detect_edge,
    draw_room_overlay,
    path_blocked_by_enemies,
    sprite_boxes,
    world_to_screen,
)
from alttp.opening_route.main_hall_to_zelda import (
    WEST_DOOR_LABEL,
    evaluate_acceptance,
    in_main_hall,
    left_main_hall_west,
    near_west_door,
)


class _FakeEnv:
    def __init__(self, writes: dict[int, int] | None = None) -> None:
        self.writes = dict(writes or {})
        self._ram = np.zeros(0x20000, dtype=np.uint8)
        for addr, value in self.writes.items():
            if 0 <= addr < len(self._ram):
                self._ram[addr] = value & 0xFF

    def get_ram(self) -> np.ndarray:
        ram = self._ram.copy()
        for addr, value in self.writes.items():
            if 0 <= addr < len(ram):
                ram[addr] = value & 0xFF
        return ram


def _set_sprite(
    writes: dict[int, int],
    slot: int,
    *,
    sprite_type: int,
    x: int,
    y: int,
    state: int = 9,
    hp: int = 4,
) -> None:
    writes[SPRITE_TYPE + slot] = sprite_type
    writes[SPRITE_STATE + slot] = state
    writes[SPRITE_HP + slot] = hp
    writes[SPRITE_X_LOW + slot] = x & 0xFF
    writes[SPRITE_X_HIGH + slot] = (x >> 8) & 0xFF
    writes[SPRITE_Y_LOW + slot] = y & 0xFF
    writes[SPRITE_Y_HIGH + slot] = (y >> 8) & 0xFF


def _snap(**kwargs: object) -> AlttpSnapshot:
    base: dict[str, object] = dict(
        game_mode=0x07,
        submodule=0x00,
        room_id=HYRULE_CASTLE_MAIN_HALL_ROOM,
        indoors=True,
        screen_id=0,
        link_x=760,
        link_y=3520,
        link_direction=0,
        link_action=0,
        camera_x=640,
        camera_y=3344,
        dark_world=False,
        sword_level=1,
        lamp_level=0,
        num_keys=0,
        follower=0,
    )
    base.update(kwargs)
    return AlttpSnapshot(**base)  # type: ignore[arg-type]


def test_classify_and_box_for_hostile() -> None:
    assert classify_sprite(SPRITE_BLUE_SOLDIER) == SPRITE_KIND_HOSTILE
    assert classify_sprite(SPRITE_HEART) == SPRITE_KIND_PICKUP
    env = _FakeEnv()
    _set_sprite(env.writes, 0, sprite_type=SPRITE_BLUE_SOLDIER, x=100, y=200)
    sp = active_sprites(env)[0]
    box = box_for_sprite(sp)
    assert box.kind == SPRITE_KIND_HOSTILE
    assert box.x0 < box.x < box.x1
    assert box.y0 < box.y <= box.y1
    assert box.contains(100, 190)


def test_sprite_boxes_filters_kinds() -> None:
    env = _FakeEnv()
    _set_sprite(env.writes, 0, sprite_type=SPRITE_BLUE_SOLDIER, x=110, y=110)
    _set_sprite(env.writes, 1, sprite_type=SPRITE_HEART, x=120, y=120)
    hostiles = sprite_boxes(env, kinds=(SPRITE_KIND_HOSTILE,))
    assert len(hostiles) == 1
    assert hostiles[0].sprite_type == SPRITE_BLUE_SOLDIER


def test_detect_edge_west_and_outdoors() -> None:
    before = _snap(link_x=520, link_y=3320, room_id=HYRULE_CASTLE_MAIN_HALL_ROOM)
    after_west = _snap(link_x=511, link_y=3320, room_id=HYRULE_CASTLE_MAIN_WEST_ROOM)
    edge = detect_edge(before, after_west, expected_room=HYRULE_CASTLE_MAIN_HALL_ROOM)
    assert edge is not None
    assert edge.direction == "LEFT"
    assert edge.to_room == HYRULE_CASTLE_MAIN_WEST_ROOM

    after_out = AlttpSnapshot(
        game_mode=0x07,
        submodule=0,
        room_id=HYRULE_CASTLE_MAIN_HALL_ROOM,
        indoors=False,
        screen_id=0x1B,
        link_x=2040,
        link_y=1740,
        link_direction=2,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
        sword_level=1,
    )
    before_south = _snap(link_x=760, link_y=3496)
    edge_s = detect_edge(
        before_south,
        after_out,
        expected_room=HYRULE_CASTLE_MAIN_HALL_ROOM,
        preferred_direction="DOWN",
    )
    assert edge_s is not None
    assert edge_s.outdoors is True
    assert edge_s.direction == "DOWN"


def test_world_to_screen_and_overlay() -> None:
    snap = _snap(camera_x=640, camera_y=3344, link_x=760, link_y=3520)
    sx, sy = world_to_screen(snap, 760, 3520)
    assert sx == 120
    assert sy == 176
    frame = np.zeros((224, 256, 3), dtype=np.uint8)
    out = draw_room_overlay(frame, snap, title="test")
    assert out.shape == (224, 256, 3)
    assert not np.array_equal(out, frame)  # link box drawn


def test_main_hall_map_json_authority() -> None:
    assert "room_61" in list_room_maps()
    m = load_room_map("room_61")
    assert m.room_base_id == HYRULE_CASTLE_MAIN_HALL_ROOM
    assert m.point("west_door_approach") is not None
    door = m.door(WEST_DOOR_LABEL)
    assert door is not None
    assert isinstance(door, KnownDoor)
    assert door.to_room == HYRULE_CASTLE_MAIN_WEST_ROOM
    assert door.direction == "LEFT"
    assert door.role == "zelda_path"
    wps = m.waypoints_for_door(door)
    assert len(wps) >= 3
    assert wps[-1][0] == door.approach_xy[0]
    assert wps[-1][1] == door.approach_xy[1]
    recovery = m.recovery_waypoints_for_door(door)
    assert [wp[2] for wp in recovery[:-1]] == [
        "natural_clear_left",
        "natural_clear_north",
    ]
    assert recovery[-1][0:2] == door.approach_xy

    summary = m.compact_summary()
    assert summary["roomHex"] == "0x61"
    assert any(d["label"] == WEST_DOOR_LABEL for d in summary["doors"])

    hall = _snap()
    assert in_main_hall(hall)
    assert not left_main_hall_west(hall)
    west = _snap(room_id=HYRULE_CASTLE_MAIN_WEST_ROOM, link_x=511, link_y=3320)
    assert left_main_hall_west(west)
    acc = evaluate_acceptance(west)
    assert acc["left_main_hall_west"] is True
    assert acc["zelda_follower"] is False

    # near_west_door reads approach from map (geometry authority).
    ax, ay = door.approach_xy
    approach = _snap(link_x=ax, link_y=ay)
    assert near_west_door(approach)


def test_all_sanctuary_path_maps_load() -> None:
    """Every measured opening/B1 map loads and has schemaVersion doors."""
    expected = {
        "room_01",
        "room_50",
        "room_51",
        "room_52",
        "room_55",
        "room_60",
        "room_61",
        "room_62",
        "room_71",
        "room_72",
        "room_80",
        "room_81",
        "room_82",
    }
    maps = set(list_room_maps())
    assert expected <= maps, f"missing maps: {expected - maps}"
    for mid in sorted(expected):
        m = load_room_map(mid)
        assert m.room_base_id > 0
        assert m.name
        # Zelda-path rooms should declare at least one door seed.
        assert m.doors or mid in {"room_71"}, mid


def test_room_60_north_door_map() -> None:
    from alttp.ram import HYRULE_CASTLE_NW_ROOM

    m = load_room_map("room_60")
    assert m.room_base_id == HYRULE_CASTLE_MAIN_WEST_ROOM
    door = m.door("north_to_0x50")
    assert door is not None
    assert door.direction == "UP"
    assert door.to_room == HYRULE_CASTLE_NW_ROOM
    assert door.role == "zelda_path"
    wps = m.waypoints_for_door(door)
    assert len(wps) >= 1
    assert wps[-1][0] == door.approach_xy[0]
    assert wps[-1][1] == door.approach_xy[1]


def test_run_room_edge_ok_when_already_at_dest() -> None:
    """Isolated edge: door dest → ok=True (not multi-hop partial failure)."""
    import numpy as np

    from alttp.opening_route.room_engine import run_room_edge
    from alttp.ram import (
        DARK_WORLD_FLAG,
        EQUIP_SWORD,
        INDOORS,
        MODULE,
        ROOM_ID,
        SUBMODULE,
        wram_index,
    )

    class _WestEnv:
        def get_ram(self) -> np.ndarray:
            ram = np.zeros(0x20000, dtype=np.uint8)
            ram[MODULE] = 0x07
            ram[SUBMODULE] = 0x00
            ram[INDOORS] = 1
            ram[DARK_WORLD_FLAG] = 0
            ram[ROOM_ID] = HYRULE_CASTLE_MAIN_WEST_ROOM & 0xFF
            ram[wram_index(EQUIP_SWORD)] = 1
            return ram

    result = run_room_edge(
        _WestEnv(),
        "room_61",
        WEST_DOOR_LABEL,
        clear=False,
        source="test",
    )
    assert result.ok is True
    assert result.phase == f"via_{WEST_DOOR_LABEL}"
    assert result.blocker == ""


def test_path_blocked_uses_enemy_box() -> None:
    env = _FakeEnv()
    _set_sprite(env.writes, 0, sprite_type=SPRITE_BLUE_SOLDIER, x=200, y=200)

    from alttp.ram import MODULE, SUBMODULE, INDOORS, ROOM_ID, LINK_X, LINK_Y

    env.writes[MODULE] = 0x07
    env.writes[SUBMODULE] = 0
    env.writes[INDOORS] = 1
    env.writes[ROOM_ID] = HYRULE_CASTLE_MAIN_HALL_ROOM
    env.writes[LINK_X] = 100 & 0xFF
    env.writes[LINK_X + 1] = (100 >> 8) & 0xFF
    env.writes[LINK_Y] = 200 & 0xFF
    env.writes[LINK_Y + 1] = (200 >> 8) & 0xFF

    hit = path_blocked_by_enemies(env, 300, 200, pad=0, max_distance=400)
    assert hit is not None
    assert hit.sprite_type == SPRITE_BLUE_SOLDIER
