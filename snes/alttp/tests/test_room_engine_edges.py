"""Offline contracts for isolated F1 room_engine door maps.

Live save-state proof is the room_engine CLI (state_load_dev), not these tests.
Graph hop ``room_50_east_to_0x01`` stays natural_entry — do not promote.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from alttp.opening_route.escape_graph import (
    N_ROOM_01,
    N_ROOM_50,
    VERIFICATION_CONTINUOUS,
    VERIFICATION_NATURAL_ENTRY,
    escape_route_graph,
    escape_route_hops,
)
from alttp.opening_route.room_engine import (
    DEST_SETTLE_MAX_FRAMES,
    at_door_destination,
    in_room,
    run_room_edge,
)
from alttp.paths import INTEGRATION_DIR
from alttp.primitives import settle_control
from alttp.ram import (
    DARK_WORLD_FLAG,
    EQUIP_SWORD,
    HYRULE_CASTLE_B1_NORTH_ZELDA_WEST_ROOM,
    HYRULE_CASTLE_B1_PIT_ROOM,
    HYRULE_CASTLE_MAIN_EAST_ROOM,
    HYRULE_CASTLE_NE_ROOM,
    HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
    HYRULE_CASTLE_NW_ROOM,
    INDOORS,
    LINK_X,
    LINK_Y,
    MODULE,
    ROOM_ID,
    SUBMODULE,
    AlttpSnapshot,
    wram_index,
)
from alttp.room_map import load_room_map

# Isolated F1 edges under this bead. Graph is not edited here.
F1_EDGES: tuple[tuple[str, str, str, int, int, str], ...] = (
    (
        "room_50",
        "east_to_0x01",
        "CastleRoom50",
        HYRULE_CASTLE_NW_ROOM,
        HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
        "RIGHT",
    ),
    (
        "room_01",
        "east_to_0x52",
        "CastleRoom01",
        HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
        HYRULE_CASTLE_NE_ROOM,
        "RIGHT",
    ),
    (
        "room_52",
        "south_to_0x62",
        "CastleRoom52",
        HYRULE_CASTLE_NE_ROOM,
        HYRULE_CASTLE_MAIN_EAST_ROOM,
        "DOWN",
    ),
)


def _snap(*, room: int, **kwargs: object) -> AlttpSnapshot:
    base: dict[str, object] = dict(
        game_mode=0x07,
        submodule=0x00,
        room_id=room,
        indoors=True,
        screen_id=0,
        link_x=0,
        link_y=0,
        link_direction=0,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
        sword_level=1,
        lamp_level=0,
        num_keys=0,
        follower=0,
    )
    base.update(kwargs)
    return AlttpSnapshot(**base)  # type: ignore[arg-type]


def _u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


class _DestEnv:
    """Fake env already standing in a door destination room."""

    def __init__(self, room: int) -> None:
        self.room = room

    def get_ram(self) -> np.ndarray:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[MODULE] = 0x07
        ram[SUBMODULE] = 0x00
        ram[INDOORS] = 1
        ram[DARK_WORLD_FLAG] = 0
        ram[ROOM_ID] = self.room & 0xFF
        ram[wram_index(EQUIP_SWORD)] = 1
        return ram


class _StairAnimEnv:
    """Origin with control; dest + submodule 14 on first step; control after anim."""

    def __init__(
        self,
        *,
        origin: int,
        dest: int,
        origin_xy: tuple[int, int],
        dest_xy: tuple[int, int],
        anim_frames: int = 292,
    ) -> None:
        self.origin = origin
        self.dest = dest
        self.origin_xy = origin_xy
        self.dest_xy = dest_xy
        self.anim_frames = anim_frames
        self.room = origin
        self.submodule = 0
        self.xy = origin_xy
        self.steps = 0
        self.dest_steps = 0

    def get_ram(self) -> np.ndarray:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[MODULE] = 0x07
        ram[SUBMODULE] = self.submodule
        ram[INDOORS] = 1
        ram[DARK_WORLD_FLAG] = 0
        _u16(ram, ROOM_ID, self.room)
        _u16(ram, LINK_X, self.xy[0])
        _u16(ram, LINK_Y, self.xy[1])
        ram[wram_index(EQUIP_SWORD)] = 1
        return ram

    def step(self, _action: object) -> None:
        self.steps += 1
        if self.room != self.dest:
            self.room = self.dest
            self.xy = self.dest_xy
            self.submodule = 14
        if self.room == self.dest:
            self.dest_steps += 1
            if self.dest_steps >= self.anim_frames:
                self.submodule = 0


def _pin_ready(state: str) -> bool:
    rom = INTEGRATION_DIR / "rom.sfc"
    pin = INTEGRATION_DIR / f"{state}.state"
    return rom.is_file() and pin.is_file()


@pytest.mark.parametrize(
    "map_id,door_label,source_state,origin,dest,direction",
    F1_EDGES,
)
def test_f1_maps_load_and_door_labels_exist(
    map_id: str,
    door_label: str,
    source_state: str,
    origin: int,
    dest: int,
    direction: str,
) -> None:
    m = load_room_map(map_id)
    assert m.room_base_id == origin
    assert m.source_state == source_state
    door = m.door(door_label)
    assert door is not None, f"{door_label} missing on {map_id}"
    assert door.direction == direction
    assert door.to_room == dest
    assert door.role == "zelda_path"
    assert door.approach_xy != (0, 0)
    assert door.landing_xy is not None
    wps = m.waypoints_for_door(door)
    assert wps
    assert wps[-1][0] == door.approach_xy[0]
    assert wps[-1][1] == door.approach_xy[1]
    labels = [d.label for d in m.doors]
    assert door_label in labels
    summary = m.compact_summary()
    assert summary["roomHex"] == f"0x{origin:02X}"
    assert any(d["label"] == door_label for d in summary["doors"])


@pytest.mark.parametrize(
    "map_id,door_label,origin,dest",
    [(row[0], row[1], row[3], row[4]) for row in F1_EDGES],
)
def test_at_door_destination_matches_to_room(
    map_id: str, door_label: str, origin: int, dest: int
) -> None:
    door = load_room_map(map_id).door(door_label)
    assert door is not None
    origin_snap = _snap(room=origin)
    dest_snap = _snap(room=dest)
    assert in_room(origin_snap, origin)
    assert not at_door_destination(origin_snap, door)
    assert at_door_destination(dest_snap, door)
    assert not in_room(dest_snap, origin)


@pytest.mark.parametrize(
    "map_id,door_label,dest",
    [(row[0], row[1], row[4]) for row in F1_EDGES],
)
def test_run_room_edge_ok_when_already_at_f1_dest(
    map_id: str, door_label: str, dest: int
) -> None:
    result = run_room_edge(
        _DestEnv(dest),
        map_id,
        door_label,
        clear=False,
        source="test",
    )
    assert result.ok is True
    assert result.phase == f"via_{door_label}"
    assert result.blocker == ""
    assert result.acceptance["at_door_dest"] is True
    assert not any(p.phase == "settle_destination" for p in result.phases)
    assert result.frames == 0


def test_room_50_east_graph_stays_natural_entry() -> None:
    """Do not promote 0x50 east → 0x01 to continuous from isolated re-runs."""
    hops = {h.hop_id: h for h in escape_route_hops()}
    hop = hops["room_50_east_to_0x01"]
    assert hop.verification == VERIFICATION_NATURAL_ENTRY
    assert hop.verification != VERIFICATION_CONTINUOUS
    assert hop.meta["map_id"] == "room_50"
    assert hop.meta["door_label"] == "east_to_0x01"
    graph = escape_route_graph()
    edge = next(e for e in graph.edges if e.edge_id == "room_50_east_to_0x01")
    assert edge.source_id == N_ROOM_50
    assert edge.target_id == N_ROOM_01
    assert edge.verification == VERIFICATION_NATURAL_ENTRY


def test_f1_chain_doors_are_not_graph_hops() -> None:
    """0x01 east and 0x52 south stay map-only; graph hop to Zelda remains planned."""
    hops = {h.hop_id: h for h in escape_route_hops()}
    assert "room_01_east_to_0x52" not in hops
    assert "room_52_south_to_0x62" not in hops
    planned = hops["room_01_to_zelda_cell"]
    assert planned.verification == "planned"


def test_settle_control_default_max_frames_stays_240() -> None:
    assert inspect.signature(settle_control).parameters["max_frames"].default == 240
    assert DEST_SETTLE_MAX_FRAMES > 240


def test_dest_settle_can_exceed_default_240() -> None:
    env = _StairAnimEnv(
        origin=HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
        dest=HYRULE_CASTLE_B1_PIT_ROOM,
        origin_xy=(760, 120),
        dest_xy=(1273, 3665),
        anim_frames=292,
    )
    result = run_room_edge(
        env,
        "room_01",
        "down_to_0x72",
        clear=False,
        source="test",
    )
    assert result.ok is True
    dest_phase = next(p for p in result.phases if p.phase == "settle_destination")
    assert dest_phase.ok is True
    assert dest_phase.frames > 240
    assert dest_phase.frames <= DEST_SETTLE_MAX_FRAMES
    assert result.snapshot.has_control is True
    assert result.snapshot.submodule == 0
    assert result.snapshot.room_base_id == HYRULE_CASTLE_B1_PIT_ROOM
    assert result.acceptance["at_door_dest"] is True


STAIR_EDGES: tuple[tuple[str, str, str, int], ...] = (
    (
        "room_01",
        "down_to_0x72",
        "CastleRoom01",
        HYRULE_CASTLE_B1_PIT_ROOM,
    ),
    (
        "room_72",
        "north_to_0x01",
        "CastleB1Guard",
        HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
    ),
    (
        "room_70",
        "north_to_0x71",
        "CastleB2Landing",
        HYRULE_CASTLE_B1_NORTH_ZELDA_WEST_ROOM,
    ),
)


@pytest.mark.rom
@pytest.mark.parametrize(
    "map_id,door_label,source_state,dest",
    [(row[0], row[1], row[2], row[4]) for row in F1_EDGES],
)
def test_isolated_f1_edge_from_save_state(
    map_id: str, door_label: str, source_state: str, dest: int
) -> None:
    if not _pin_ready(source_state):
        pytest.skip(f"ALTTP ROM or {source_state}.state missing")

    from alttp.opening_route.room_engine import run_room_edge as play
    from alttp.startup import build_boot_env

    env = build_boot_env(source_state)
    try:
        env.reset()  # type: ignore[attr-defined]
        settle_control(env)
        result = play(env, map_id, door_label, source="state_load_dev")
    finally:
        env.close()  # type: ignore[attr-defined]

    assert result.ok is True
    assert result.snapshot.room_base_id == dest
    assert result.acceptance["at_door_dest"] is True
    assert Path(INTEGRATION_DIR / f"{source_state}.state").is_file()


@pytest.mark.rom
@pytest.mark.parametrize("map_id,door_label,source_state,dest", STAIR_EDGES)
def test_isolated_stair_edge_dest_settle(
    map_id: str, door_label: str, source_state: str, dest: int
) -> None:
    if not _pin_ready(source_state):
        pytest.skip(f"ALTTP ROM or {source_state}.state missing")

    from alttp.opening_route.room_engine import run_room_edge as play
    from alttp.startup import build_boot_env

    env = build_boot_env(source_state)
    try:
        env.reset()  # type: ignore[attr-defined]
        settle_control(env)
        result = play(
            env, map_id, door_label, clear=False, source="state_load_dev"
        )
    finally:
        env.close()  # type: ignore[attr-defined]

    assert result.ok is True
    assert result.snapshot.room_base_id == dest
    assert result.snapshot.has_control is True
    assert result.acceptance["at_door_dest"] is True
    dest_phase = next(p for p in result.phases if p.phase == "settle_destination")
    assert dest_phase.ok is True
    assert dest_phase.frames <= DEST_SETTLE_MAX_FRAMES
