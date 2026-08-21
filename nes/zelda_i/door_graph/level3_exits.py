"""Level 3 (Manji) door-graph seed edges and room-id constants."""

from __future__ import annotations

from zelda_i.door_graph.core import (
    DoorDir,
    DungeonDoorGraph,
    GateKind,
    RoomExit,
)
from zelda_i.level3_geometry import (
    BOMB_STAND_5B_RIGHT,
    KEY_DOOR_Y,
    STAIRS_69_RIGHT_Y,
    WEST_WALL_5B_X,
)

# Level 3 (Manji) seed — additive LIVE edges (rr-vpl / l3_past_5b 2026-08-07)
# ---------------------------------------------------------------------------

L3_ENTRY = 0x7C
L3_WEST_KEY = 0x7B
L3_NORTH_ZOLS = 0x6B
L3_DARKNUTS = 0x5B
L3_ZOL_KEY_4B = 0x4B
L3_COMPASS = 0x5A
L3_WEST_DARKNUTS = 0x59
L3_SOUTH_DARKNUTS = 0x69
L3_RAFT_PASSAGE = 0x0F
L3_MAP_4C = 0x4C
L3_BOMB_SHORTCUT = 0x5C
L3_KEESE_4A = 0x4A


def _l3_exits() -> dict[int, tuple[RoomExit, ...]]:
    """L3 edges observed assisted LIVE (not all pure-runner encoded)."""
    return {
        L3_ENTRY: (
            RoomExit(
                DoorDir.LEFT,
                L3_WEST_KEY,
                GateKind.OPEN,
                approach_xy=(48, 149),
                notes="LEFT+UP diagonal residual @ y≈149 (pure LEFT sticks x≈32)",
                verification="observed",
            ),
        ),
        L3_WEST_KEY: (
            RoomExit(
                DoorDir.UP,
                L3_NORTH_ZOLS,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes="UP @ x≈120 (|dx|≤4); wider align sticks x≈112",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L3_ENTRY,
                GateKind.OPEN,
                notes="return east to entry",
                verification="observed",
            ),
        ),
        L3_NORTH_ZOLS: (
            RoomExit(
                DoorDir.UP,
                L3_DARKNUTS,
                GateKind.KILL_CLEAR,
                approach_xy=(120, 93),
                notes="after type-0x13 clear; free-explore + UP @ x≈120",
                verification="observed",
            ),
            RoomExit(
                DoorDir.DOWN,
                L3_WEST_KEY,
                GateKind.OPEN,
                notes="return south",
                verification="observed",
            ),
        ),
        L3_DARKNUTS: (
            RoomExit(
                DoorDir.UP,
                L3_ZOL_KEY_4B,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes="open without Darknut clear; 3× Zol + key",
                verification="observed",
            ),
            RoomExit(
                DoorDir.DOWN,
                L3_NORTH_ZOLS,
                GateKind.OPEN,
                notes="return south",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L3_COMPASS,
                GateKind.OPEN,
                approach_xy=(32, KEY_DOOR_Y),
                notes=(
                    "4× Keese + traps + Compass 0x16 — Raft path; "
                    f"west wall x≈{WEST_WALL_5B_X} (not 32); push once x≤48"
                ),
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L3_BOMB_SHORTCUT,
                GateKind.BOMB,
                bomb_stand=BOMB_STAND_5B_RIGHT,
                notes="bomb-RIGHT @(192,141) → 0x5c boss shortcut (recon)",
                verification="observed",
            ),
        ),
        L3_ZOL_KEY_4B: (
            RoomExit(
                DoorDir.DOWN,
                L3_DARKNUTS,
                GateKind.OPEN,
                notes="return south",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L3_KEESE_4A,
                GateKind.KEY,
                approach_xy=(32, 141),
                notes="key door → 5× Keese 0x4a",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L3_MAP_4C,
                GateKind.KEY,
                approach_xy=(208, 141),
                notes="key door → map room item 0x17",
                verification="observed",
            ),
        ),
        L3_COMPASS: (
            RoomExit(
                DoorDir.RIGHT,
                L3_DARKNUTS,
                GateKind.OPEN,
                notes="return east",
                verification="observed",
            ),
            RoomExit(
                DoorDir.LEFT,
                L3_WEST_DARKNUTS,
                GateKind.KEY,
                approach_xy=(32, KEY_DOOR_Y),
                notes=(
                    "key door → 5× Darknut; long y=141 push "
                    "(key-waste trap if y≠141 / short push)"
                ),
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L3_KEESE_4A,
                GateKind.OPEN,
                notes="free north → 5× Keese",
                verification="observed",
            ),
        ),
        L3_WEST_DARKNUTS: (
            RoomExit(
                DoorDir.RIGHT,
                L3_COMPASS,
                GateKind.OPEN,
                notes="return east after key entry",
                verification="observed",
            ),
            RoomExit(
                DoorDir.DOWN,
                L3_SOUTH_DARKNUTS,
                GateKind.KILL_CLEAR,
                approach_xy=(120, 205),
                notes="kill 5 Darknuts opens DOWN; settle spawn ~80–100f",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                0x49,
                GateKind.OPEN,
                notes="north mixed room (probe)",
                verification="observed",
            ),
        ),
        L3_SOUTH_DARKNUTS: (
            RoomExit(
                DoorDir.UP,
                L3_WEST_DARKNUTS,
                GateKind.OPEN,
                notes="return north",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                L3_RAFT_PASSAGE,
                GateKind.KILL_CLEAR,
                approach_xy=(208, STAIRS_69_RIGHT_Y),
                notes=(
                    "stairs RIGHT only @ y≈141 after 8× Darknut clear → "
                    "mode-9 passage 0x0f (Raft)"
                ),
                verification="observed",
            ),
        ),
        L3_RAFT_PASSAGE: (
            RoomExit(
                DoorDir.LEFT,
                None,
                GateKind.OPEN,
                notes=(
                    "underworld mode 9: DOWN y189 → RIGHT x≈176 → UP channel → "
                    "LEFT x≈136 touch Raft (ADDR_RAFT); not a cardinal door"
                ),
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L3_SOUTH_DARKNUTS,
                GateKind.OPEN,
                approach_xy=(48, 77),
                notes=(
                    "EXIT reverse: channel x≈176 south → west → NW stairs hold UP "
                    "→ mode10 → play 0x69 (Level3Raft backtrack)"
                ),
                verification="observed",
            ),
        ),
        L3_BOMB_SHORTCUT: (
            RoomExit(
                DoorDir.LEFT,
                L3_DARKNUTS,
                GateKind.OPEN,
                notes="return west to 0x5b",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                L3_MAP_4C,
                GateKind.OPEN,
                approach_xy=(120, 93),
                notes="north to map room (LIVE post-Raft shortcut)",
                verification="observed",
            ),
            RoomExit(
                DoorDir.RIGHT,
                0x5D,
                GateKind.KILL_CLEAR,
                approach_xy=(208, 141),
                notes="RIGHT only @ y≈141 after 3× Darknut clear → 0x5d boss prep",
                verification="observed",
            ),
        ),
        0x5D: (
            RoomExit(
                DoorDir.LEFT,
                L3_BOMB_SHORTCUT,
                GateKind.OPEN,
                notes="return west to 0x5c",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                0x4D,
                GateKind.KILL_CLEAR,
                approach_xy=(120, 93),
                notes=(
                    "Manhandla 0x4d type 0x3c: clear Zol/Gel/Keese slots 1–12 "
                    "(ignore 0x2b) → doors raw=10 then UP (assisted LIVE 2/2)"
                ),
                verification="observed",
            ),
        ),
        0x4D: (
            RoomExit(
                DoorDir.DOWN,
                0x5D,
                GateKind.OPEN,
                notes="return south to prep",
                verification="observed",
            ),
            RoomExit(
                DoorDir.UP,
                0x3D,
                GateKind.KILL_CLEAR,
                approach_xy=(120, 93),
                notes="after Manhandla kill → TF room 0x3d bit 0x04",
                verification="observed",
            ),
        ),
    }


LEVEL_3_DOOR_GRAPH = DungeonDoorGraph.from_exits(
    _l3_exits(),
    level=3,
    name="level_3_manji",
)


def level_3_door_graph() -> DungeonDoorGraph:
    """Return a fresh copy of the L3 seed graph (safe to mutate rooms)."""
    raw = {
        rid: tuple(
            RoomExit(
                direction=e.direction,
                target_room=e.target_room,
                gate=e.gate,
                bomb_stand=e.bomb_stand,
                approach_xy=e.approach_xy,
                key_cost=e.key_cost,
                notes=e.notes,
                verification=e.verification,
            )
            for e in exits
        )
        for rid, exits in LEVEL_3_DOOR_GRAPH.rooms.items()
    }
    return DungeonDoorGraph.from_exits(raw, level=3, name="level_3_manji")

