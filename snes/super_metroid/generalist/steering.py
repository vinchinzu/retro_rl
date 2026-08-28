"""Goal-directed room steering for the generalist contractor.

Join xy stays the dense target in the Goal room. Across rooms,
``steering_target`` walks the canonical Zebes graph
(``FULL_ROOM_GRAPH_PATH``) via ``shortest_room_path`` and seats on the
first edge's source block. Collision grids stay in ``solid.py``; they are
not the topology graph. Nearest clip-9 is the fallback when the path is
missing or the first edge has no block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from super_metroid.generalist.solid import RoomSolid, potential_xy
from super_metroid.paths import FULL_ROOM_GRAPH_PATH
from super_metroid.ram import BOMBS_MASK, HI_JUMP_MASK, MORPH_BALL_MASK, VARIA_MASK
from super_metroid.rooms import shortest_room_path

FALLBACK_ROUTE_DOORS = 4
ROOM_ROUTE_PX = 4_096.0
TILE_PX = 16

# $09A4 collected-items bits named in ram.py, plus the rest of the ITEM_CAPABILITIES
# layout used by the room graph. Keep the table explicit; do not alias ad-hoc.
_ITEM_BITS: tuple[tuple[int, str], ...] = (
    (VARIA_MASK, "varia_suit"),
    (0x0002, "spring_ball"),
    (MORPH_BALL_MASK, "morph_ball"),
    (0x0008, "screw_attack"),
    (0x0020, "gravity_suit"),
    (HI_JUMP_MASK, "hi_jump"),
    (0x0200, "space_jump"),
    (BOMBS_MASK, "bombs"),
    (0x2000, "speed_booster"),
    (0x4000, "grapple_beam"),
    (0x8000, "xray_scope"),
)
# $09A8 collected-beams bits.
_BEAM_BITS: tuple[tuple[int, str], ...] = (
    (0x0001, "wave_beam"),
    (0x0002, "ice_beam"),
    (0x0004, "spazer"),
    (0x0008, "plasma_beam"),
    (0x1000, "charge_beam"),
)


@dataclass(frozen=True)
class SteeringTarget:
    """Dense-reward target and its remaining room-route depth."""

    x: int
    y: int
    kind: str
    remaining_doors: int
    next_room_id: int | None = None
    route_rooms: tuple[int, ...] = ()


@lru_cache(maxsize=4)
def _load_room_graph(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "edges" not in payload:
        raise ValueError(f"room graph missing edges: {path}")
    return payload


def load_room_graph(path: Path | str | None = None) -> dict[str, Any]:
    """Load the canonical room graph (cached). Tests may pass an in-memory dict instead."""

    graph_path = Path(path) if path is not None else FULL_ROOM_GRAPH_PATH
    return _load_room_graph(str(graph_path))


def capabilities_from_state(state: Any) -> set[str]:
    """Map Samus items/beams/ammo onto ``rooms.capabilities.ITEM_CAPABILITIES`` names."""

    items = int(getattr(state, "collected_items", 0) or 0)
    beams = int(getattr(state, "collected_beams", 0) or 0)
    caps = {name for mask, name in _ITEM_BITS if items & mask}
    caps.update(name for mask, name in _BEAM_BITS if beams & mask)
    if _has_ammo(state, "missiles", "max_missiles"):
        caps.add("missiles")
    if _has_ammo(state, "super_missiles", "max_super_missiles"):
        caps.add("super_missiles")
    if _has_ammo(state, "power_bombs", "max_power_bombs"):
        caps.add("power_bombs")
    return caps


def _has_ammo(state: Any, current_attr: str, max_attr: str) -> bool:
    return int(getattr(state, current_attr, 0) or 0) > 0 or int(
        getattr(state, max_attr, 0) or 0
    ) > 0


def _source_pixel(edge: Mapping[str, Any]) -> tuple[int, int] | None:
    source = edge.get("source")
    if not isinstance(source, Mapping):
        return None
    block = source.get("block")
    if not isinstance(block, (list, tuple)) or len(block) < 2:
        return None
    try:
        return int(block[0]) * TILE_PX + 8, int(block[1]) * TILE_PX + 8
    except (TypeError, ValueError):
        return None


def _fallback_target(
    state: Any, goal: Any, solid: RoomSolid | None, room_id: int
) -> SteeringTarget:
    x, y = potential_xy(state, goal, solid)
    return SteeringTarget(
        x=x,
        y=y,
        kind="nearest_door",
        remaining_doors=FALLBACK_ROUTE_DOORS,
        route_rooms=(room_id,),
    )


def steering_target(
    state: Any,
    goal: Any,
    solid: RoomSolid | None,
    *,
    graph: Mapping[str, Any] | None = None,
    capabilities: Iterable[str] | None = None,
) -> SteeringTarget:
    """Choose Join xy, the first Goal-directed door, or a local fallback."""

    room_id = int(getattr(state, "room_id", getattr(state, "room", 0)) or 0)
    goal_room_id = int(getattr(goal, "room_id", 0) or 0)
    if room_id == goal_room_id:
        return SteeringTarget(
            x=int(getattr(goal, "x", 0) or 0),
            y=int(getattr(goal, "y", 0) or 0),
            kind="join",
            remaining_doors=0,
            route_rooms=(room_id,),
        )

    room_graph = graph if graph is not None else load_room_graph()
    caps = (
        set(capabilities)
        if capabilities is not None
        else capabilities_from_state(state)
    )
    route = shortest_room_path(room_graph, room_id, goal_room_id, caps)
    if route:
        first = route[0]
        seat = _source_pixel(first)
        if seat is not None:
            return SteeringTarget(
                x=seat[0],
                y=seat[1],
                kind="goal_door",
                remaining_doors=len(route),
                next_room_id=int(first["target"]["roomId"]),
                route_rooms=(
                    room_id,
                    *(int(edge["target"]["roomId"]) for edge in route),
                ),
            )
    return _fallback_target(state, goal, solid, room_id)


def steering_distance(state: Any, target: SteeringTarget) -> float:
    """Monotone room-route potential plus local Euclidean distance.

    Editor rooms in this corpus are less than 4096 px across.  Removing one
    remaining door therefore rewards a correct room transition instead of
    applying the old clipped -1 spike when room-local coordinates reset.
    """

    dx = float(int(getattr(state, "samus_x", 0) or 0) - int(target.x))
    dy = float(int(getattr(state, "samus_y", 0) or 0) - int(target.y))
    local = (dx * dx + dy * dy) ** 0.5
    return local + float(target.remaining_doors) * ROOM_ROUTE_PX


__all__ = [
    "FALLBACK_ROUTE_DOORS",
    "ROOM_ROUTE_PX",
    "SteeringTarget",
    "capabilities_from_state",
    "load_room_graph",
    "steering_distance",
    "steering_target",
]
