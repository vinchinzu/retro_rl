"""Early Brinstar room graph (start → morph → planned first missiles).

Map cells are (map_x, map_y) from system RAM $50/$4F. Verified path:
start (3,14) → (2,14) → morph room (1,14).

East of start (probe 2026-07-27): (3,14) → (4,14) → (5,14). The blue door
at the east end of (5,14) has not yet been opened under script; first missiles
remain a planned milestone after that door / shaft route.
"""

from __future__ import annotations

from retro_harness.adventure.graph import (
    GraphEdge,
    GraphNode,
    ProgressionMilestone,
    RouteGraph,
    RouteLeg,
    apply_milestones,
)

from metroid.ram import (
    EAST_CORRIDOR_MAP_X,
    EAST_CORRIDOR_MAP_Y,
    MORPH_MAP_X,
    MORPH_MAP_Y,
    START_MAP_X,
    START_MAP_Y,
)


def map_node_id(map_x: int, map_y: int) -> str:
    return f"b_{map_x:02x}_{map_y:02x}"


NODE_START = map_node_id(START_MAP_X, START_MAP_Y)
NODE_MORPH = map_node_id(MORPH_MAP_X, MORPH_MAP_Y)
NODE_EAST_DOOR = map_node_id(EAST_CORRIDOR_MAP_X, EAST_CORRIDOR_MAP_Y)
# Placeholder until the missile room map cell is probe-verified.
NODE_FIRST_MISSILES = "b_first_missiles"

# Probe-verified corridor west of start to Maru Mari.
_EARLY_PATH: tuple[tuple[int, int], ...] = (
    (START_MAP_X, START_MAP_Y),  # (3, 14)
    (2, 14),
    (MORPH_MAP_X, MORPH_MAP_Y),  # (1, 14)
)

# Probe-visited cells east of start (Level1 pure RIGHT + combat).
_EAST_PATH: tuple[tuple[int, int], ...] = (
    (START_MAP_X, START_MAP_Y),  # (3, 14)
    (4, 14),
    (EAST_CORRIDOR_MAP_X, EAST_CORRIDOR_MAP_Y),  # (5, 14)
)


def _direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    ax, ay = a
    bx, by = b
    if by < ay:
        return "UP"
    if by > ay:
        return "DOWN"
    if bx < ax:
        return "LEFT"
    if bx > ax:
        return "RIGHT"
    return ""


def build_early_brinstar_graph(
    path: tuple[tuple[int, int], ...] | None = None,
) -> RouteGraph:
    """Capability-aware graph for early Brinstar (morph + east probe)."""
    west = path if path is not None else _EARLY_PATH
    # Union west morph path with east corridor cells.
    seen: set[tuple[int, int]] = set()
    cells: list[tuple[int, int]] = []
    for cell in (*west, *_EAST_PATH):
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)

    nodes = [
        GraphNode(
            node_id=map_node_id(x, y),
            name=f"Brinstar ({x},{y})",
            area="brinstar",
            tags=frozenset({"brinstar", "map_cell"}),
            meta={"map_x": x, "map_y": y},
        )
        for x, y in cells
    ]
    for x, y, tag in (
        (START_MAP_X, START_MAP_Y, "start"),
        (MORPH_MAP_X, MORPH_MAP_Y, "morph"),
        (EAST_CORRIDOR_MAP_X, EAST_CORRIDOR_MAP_Y, "east_door"),
    ):
        nid = map_node_id(x, y)
        if not any(n.node_id == nid for n in nodes):
            nodes.append(
                GraphNode(
                    node_id=nid,
                    name=f"Brinstar ({x},{y})",
                    area="brinstar",
                    tags=frozenset({"brinstar", "map_cell", tag}),
                    meta={"map_x": x, "map_y": y},
                )
            )
    nodes.append(
        GraphNode(
            node_id=NODE_FIRST_MISSILES,
            name="First missile expansion",
            area="brinstar",
            tags=frozenset({"brinstar", "item", "missiles", "planned"}),
            meta={"stop": "is_missiles_obtained"},
        )
    )

    edges: list[GraphEdge] = []

    def _add_bidirectional(
        a: tuple[int, int],
        b: tuple[int, int],
        *,
        forward_verification: str,
        reverse_requires: frozenset[str] = frozenset(),
        provenance: str = "early_brinstar_probe",
    ) -> None:
        src = map_node_id(*a)
        dst = map_node_id(*b)
        edges.append(
            GraphEdge(
                source_id=src,
                target_id=dst,
                direction=_direction(a, b),
                verification=forward_verification,
                provenance=provenance,
                meta={"from": list(a), "to": list(b)},
            )
        )
        edges.append(
            GraphEdge(
                source_id=dst,
                target_id=src,
                direction=_direction(b, a),
                requires=reverse_requires,
                verification="planned",
                provenance=provenance,
                meta={"from": list(b), "to": list(a)},
            )
        )

    for a, b in zip(west, west[1:]):
        _add_bidirectional(
            a,
            b,
            forward_verification=(
                "continuous" if b == (MORPH_MAP_X, MORPH_MAP_Y) else "planned"
            ),
            reverse_requires=(
                frozenset({"morph_ball"})
                if a == (MORPH_MAP_X, MORPH_MAP_Y)
                else frozenset()
            ),
        )
    for a, b in zip(_EAST_PATH, _EAST_PATH[1:]):
        _add_bidirectional(
            a,
            b,
            forward_verification="continuous",
            provenance="east_corridor_probe",
        )
    # Planned: open (5,14) blue door / shaft route → first missiles.
    edges.append(
        GraphEdge(
            source_id=NODE_EAST_DOOR,
            target_id=NODE_FIRST_MISSILES,
            direction="RIGHT",
            requires=frozenset(),
            verification="planned",
            provenance="walkthrough_first_missiles",
            meta={"note": "blue door + west/east shaft route still WIP"},
        )
    )
    return RouteGraph(nodes, edges)


EARLY_BRINSTAR_GRAPH = build_early_brinstar_graph()

EARLY_MILESTONES = (
    ProgressionMilestone(
        "brinstar_control",
        "Controllable Brinstar start",
        node_id=NODE_START,
        timeout_frames=4_000,
        policy_id="power_on_boot",
        goal="level1_ready",
    ),
    ProgressionMilestone(
        "morph_ball",
        "Maru Mari (Morph Ball) collected",
        node_id=NODE_MORPH,
        acquires=frozenset({"morph_ball"}),
        timeout_frames=4_000,
        policy_id="morph_ball_segment",
        goal="morph_obtained",
    ),
    ProgressionMilestone(
        "first_missiles",
        "First missile expansion (capacity > 0)",
        node_id=NODE_FIRST_MISSILES,
        requires=frozenset({"morph_ball"}),
        acquires=frozenset({"missiles"}),
        timeout_frames=12_000,
        policy_id="first_missiles_segment",
        goal="missiles_obtained",
    ),
)


def morph_route_legs() -> tuple[RouteLeg, ...]:
    mid = map_node_id(2, 14)
    return (
        RouteLeg(
            "start_to_west",
            NODE_START,
            mid,
            goal="west_corridor",
        ),
        RouteLeg(
            "west_to_morph",
            mid,
            NODE_MORPH,
            acquires=frozenset({"morph_ball"}),
            goal="morph_obtained",
        ),
    )


def missiles_route_legs() -> tuple[RouteLeg, ...]:
    """Planned morph → start → east door → first missiles legs."""
    mid = map_node_id(2, 14)
    mid_east = map_node_id(4, 14)
    return (
        RouteLeg(
            "morph_to_mid",
            NODE_MORPH,
            mid,
            requires=frozenset({"morph_ball"}),
            goal="west_return",
        ),
        RouteLeg(
            "mid_to_start",
            mid,
            NODE_START,
            requires=frozenset({"morph_ball"}),
            goal="return_start",
        ),
        RouteLeg(
            "start_to_mid_east",
            NODE_START,
            mid_east,
            requires=frozenset({"morph_ball"}),
            goal="east_corridor",
        ),
        RouteLeg(
            "mid_east_to_door",
            mid_east,
            NODE_EAST_DOOR,
            requires=frozenset({"morph_ball"}),
            goal="east_door",
        ),
        RouteLeg(
            "east_door_to_missiles",
            NODE_EAST_DOOR,
            NODE_FIRST_MISSILES,
            requires=frozenset({"morph_ball"}),
            acquires=frozenset({"missiles"}),
            goal="missiles_obtained",
        ),
    )


def validate_early_milestones() -> frozenset[str]:
    caps, _ = apply_milestones(EARLY_MILESTONES)
    return caps
