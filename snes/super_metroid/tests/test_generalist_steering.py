"""Goal-door routing and monotone room potential (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from super_metroid.generalist.goals import Goal
from super_metroid.generalist.solid import (
    editor_rooms_dir,
    load_room_solid,
    room_solid_from_collision,
)
from super_metroid.generalist.steering import (
    FALLBACK_ROUTE_DOORS,
    ROOM_ROUTE_PX,
    capabilities_from_state,
    load_room_graph,
    steering_distance,
    steering_target,
)
from super_metroid.paths import FULL_ROOM_GRAPH_PATH
from super_metroid.ram import BOMBS_MASK, MORPH_BALL_MASK


def _state(room_id: int, x: int = 24, y: int = 24, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(room_id=room_id, samus_x=x, samus_y=y, **kwargs)


def _edge(
    source: int,
    target: int,
    block: list[int] | None = None,
    *,
    requires: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source": {"roomId": source, "block": block},
        "target": {"roomId": target},
        "requires": list(requires or ()),
        "impossible": False,
    }


def test_steering_selects_first_exit_on_goal_route() -> None:
    graph = {
        "edges": [
            _edge(0x1000, 0x2000, [10, 20]),
            _edge(0x2000, 0x3000, [30, 40]),
        ]
    }
    solid = room_solid_from_collision(0x1000, [[0, 9], [0, 0]])

    target = steering_target(
        _state(0x1000),
        Goal("next", 0x3000, 500, 600),
        solid,
        graph=graph,
    )

    assert target.kind == "goal_door"
    assert (target.x, target.y) == (10 * 16 + 8, 20 * 16 + 8)
    assert target.next_room_id == 0x2000
    assert target.remaining_doors == 2
    assert target.route_rooms == (0x1000, 0x2000, 0x3000)


def test_steering_long_route_is_goal_door_without_a_door_cap() -> None:
    graph = {
        "edges": [
            _edge(0x1000, 0x2000, [1, 1]),
            _edge(0x2000, 0x3000, [1, 1]),
            _edge(0x3000, 0x4000, [1, 1]),
            _edge(0x4000, 0x5000, [2, 3]),
        ]
    }
    solid = room_solid_from_collision(0x1000, [[0, 9], [0, 0]])

    target = steering_target(
        _state(0x1000),
        Goal("next", 0x5000, 500, 600),
        solid,
        graph=graph,
    )

    assert target.kind == "goal_door"
    assert (target.x, target.y) == (1 * 16 + 8, 1 * 16 + 8)
    assert target.next_room_id == 0x2000
    assert target.remaining_doors == 4
    assert target.route_rooms == (0x1000, 0x2000, 0x3000, 0x4000, 0x5000)


def test_steering_falls_back_when_path_or_block_is_missing() -> None:
    solid = room_solid_from_collision(0x1000, [[0, 9], [0, 0]])
    disconnected = steering_target(
        _state(0x1000),
        Goal("next", 0x5000, 500, 600),
        solid,
        graph={"edges": [_edge(0x1000, 0x2000, [1, 1])]},
    )
    assert disconnected.kind == "nearest_door"
    assert (disconnected.x, disconnected.y) == (24, 8)
    assert disconnected.remaining_doors == FALLBACK_ROUTE_DOORS

    no_block = steering_target(
        _state(0x1000),
        Goal("next", 0x2000, 500, 600),
        solid,
        graph={"edges": [_edge(0x1000, 0x2000, None)]},
    )
    assert no_block.kind == "nearest_door"
    assert (no_block.x, no_block.y) == (24, 8)


def test_steering_skips_gated_edges_without_capabilities() -> None:
    graph = {
        "edges": [_edge(0x1000, 0x2000, [10, 20], requires=["missiles"])]
    }
    solid = room_solid_from_collision(0x1000, [[0, 9], [0, 0]])
    goal = Goal("next", 0x2000, 500, 600)

    blocked = steering_target(_state(0x1000), goal, solid, graph=graph)
    assert blocked.kind == "nearest_door"

    armed = steering_target(
        _state(0x1000, missiles=5),
        goal,
        solid,
        graph=graph,
    )
    assert armed.kind == "goal_door"
    assert armed.next_room_id == 0x2000


def test_capabilities_from_state_maps_ram_item_masks() -> None:
    caps = capabilities_from_state(
        SimpleNamespace(
            collected_items=MORPH_BALL_MASK | BOMBS_MASK | 0x2000,
            collected_beams=0x0002,
            missiles=0,
            max_missiles=5,
            super_missiles=0,
            max_super_missiles=0,
            power_bombs=0,
            max_power_bombs=0,
        )
    )
    assert {"morph_ball", "bombs", "speed_booster", "ice_beam", "missiles"} <= caps
    assert "super_missiles" not in caps


def test_room_transition_reduces_route_potential() -> None:
    graph = {
        "edges": [
            _edge(0x1000, 0x2000, [10, 10]),
            _edge(0x2000, 0x3000, [20, 20]),
        ]
    }
    solid_a = room_solid_from_collision(0x1000, [[0, 9], [0, 0]])
    solid_b = room_solid_from_collision(0x2000, [[0, 9], [0, 0]])
    goal = Goal("next", 0x3000, 500, 600)
    before_state = _state(0x1000, 160, 160)
    after_state = _state(0x2000, 24, 24)
    before = steering_target(before_state, goal, solid_a, graph=graph)
    after = steering_target(after_state, goal, solid_b, graph=graph)

    assert before.remaining_doors == 2
    assert after.remaining_doors == 1
    assert steering_distance(before_state, before) - steering_distance(
        after_state, after
    ) > ROOM_ROUTE_PX / 2


def test_goal_room_target_remains_join_coordinates() -> None:
    state = _state(0x3000, 450, 550)
    goal = Goal("next", 0x3000, 500, 600)
    target = steering_target(state, goal, None, graph={"edges": []})

    assert target.kind == "join"
    assert (target.x, target.y) == (500, 600)
    assert target.remaining_doors == 0


def test_real_construction_route_chooses_bottom_left_not_entry() -> None:
    if not FULL_ROOM_GRAPH_PATH.is_file():
        pytest.skip("canonical room graph missing")
    root = editor_rooms_dir()
    if root is None:
        pytest.skip("snes_editor navigation export missing")
    solid = load_room_solid(0x9F11, root)
    assert solid is not None
    entry = _state(0x9F11, 24, 136)

    target = steering_target(
        entry,
        Goal("first_missile", 0xA107, 85, 139),
        solid,
        graph=load_room_graph(),
    )

    assert target.kind == "goal_door"
    assert target.next_room_id == 0xA107
    assert (target.x, target.y) == (8, 376)
    assert solid.nearest_door(entry.samus_x, entry.samus_y) != (
        target.x,
        target.y,
    )


def test_real_green_pirates_route_chooses_bottom_left() -> None:
    if not FULL_ROOM_GRAPH_PATH.is_file():
        pytest.skip("canonical room graph missing")
    root = editor_rooms_dir()
    if root is None:
        pytest.skip("snes_editor navigation export missing")
    solid = load_room_solid(0x99BD, root)
    assert solid is not None
    middle = _state(0x99BD, 224, 1163)

    target = steering_target(
        middle,
        Goal("green_elevator", 0x9938, 126, 139),
        solid,
        graph=load_room_graph(),
    )

    assert target.kind == "goal_door"
    assert target.next_room_id == 0x9969
    assert (target.x, target.y) == (8, 1656)
    assert solid.nearest_door(middle.samus_x, middle.samus_y) != (
        target.x,
        target.y,
    )
