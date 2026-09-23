"""Overworld lattice: ROM walkability, routes, the shot escape, the pond fairy.

Behaviour only. Each test builds the geometry it needs and asserts where Link
goes or what he presses, never a reason string alone.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon.tilemap import (
    ADDR_FIRST_UNWALKABLE,
    ADDR_ROOM_TILE_MAP,
    TILE_ROWS,
    WRAM_BASE,
    WRAM_RAM_OFFSET,
    ow_walkable_nodes,
)
from zelda_i.overworld.common import _sim_walk, shot_escape
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.heart_farm import POND_EDGE_Y, PondFairyController
from zelda_i.overworld.path import OverworldPathController
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot
from zelda_i.walk.physics import lattice_route, lattice_step

ROCK = 0x90
BOX = (0, 240, 61, 221)


def _ram(rock_cols: dict[int, range] | None = None, extra: dict[tuple[int, int], int] | None = None):
    ram = np.zeros(10240, dtype=np.uint8)
    ram[ADDR_FIRST_UNWALKABLE] = 0x89
    base = WRAM_RAM_OFFSET + ADDR_ROOM_TILE_MAP - WRAM_BASE
    for col, rows in (rock_cols or {}).items():
        for row in rows:
            ram[base + col * TILE_ROWS + row] = ROCK
    for (col, row), tile in (extra or {}).items():
        ram[base + col * TILE_ROWS + row] = tile
    return ram


def _all_nodes() -> frozenset[tuple[int, int]]:
    return ow_walkable_nodes(_ram())


def test_a_rock_tile_blocks_both_nodes_whose_feet_cover_it() -> None:
    # Tile column 10 (x 80..87), feet row of y=141 is tile row 11.
    nodes = ow_walkable_nodes(_ram({10: range(11, 12)}))
    assert (72, 141) not in nodes  # feet on cols 9 and 10
    assert (80, 141) not in nodes  # feet on cols 10 and 11
    assert (64, 141) in nodes and (88, 141) in nodes
    assert (80, 133) in nodes  # the row above is clear


def test_walkable_extra_tiles_pass_even_past_the_threshold() -> None:
    nodes = ow_walkable_nodes(_ram(extra={(10, 11): 0x8D, (11, 11): 0xDF}))
    assert (80, 141) in nodes


def test_route_goes_through_the_only_gap_with_few_corners() -> None:
    # A wall at tile column 15 (x 120..127) for every row except the feet row
    # of y=165, so the only crossing is on that row.
    wall = [r for r in range(TILE_ROWS) if r != (165 + 11 - 64) // 8]
    nodes = ow_walkable_nodes(_ram({15: wall}))
    route = lattice_route(nodes, (40, 101), {(232, 101)})
    assert route is not None
    assert route[-1] == (232, 101)
    assert any(y == 165 for _, y in route)
    assert len(route) <= 4


def test_route_is_none_when_the_goal_is_sealed_off() -> None:
    nodes = ow_walkable_nodes(_ram({15: range(TILE_ROWS)}))
    assert lattice_route(nodes, (40, 101), {(232, 101)}) is None


def test_lattice_step_slides_on_the_free_axis_first() -> None:
    # Four px off the column on a row: slide along the row to it first.
    assert lattice_step(124, 141, (120, 61)) == "LEFT"
    assert lattice_step(120, 141, (120, 61)) == "UP"


def test_lattice_step_lets_the_rom_snap_a_near_column() -> None:
    # Within LATTICE_SNAP_PX the column is the nearest grid line, and UP
    # slides Link onto it (see the sim test below). Walking LEFT instead
    # overshot by 2 px and flipped every frame on L6 0x28.
    assert lattice_step(122, 141, (120, 61)) == "UP"
    assert lattice_step(118, 141, (120, 61)) == "UP"


def test_sim_walk_snaps_to_the_nearest_grid_line_before_turning() -> None:
    path = _sim_walk(115, 141, "UP", 6, None, BOX)
    assert path[0][0] == 113.5 and path[1] == (112.0, 141.0)
    assert path[-1][0] == 112.0 and path[-1][1] < 141


def test_sim_walk_stops_on_the_last_node_before_rock() -> None:
    nodes = frozenset(n for n in _all_nodes() if not (n[0] == 112 and n[1] < 125))
    path = _sim_walk(112, 141, "UP", 20, nodes, BOX)
    assert path[-1] == (112.0, 125.0)


def test_shot_escape_crosses_a_shot_coming_down_the_row() -> None:
    # A shot 60 px east on Link's row, flying west: standing is a hit.
    direction, needed = shot_escape(120, 141, [(180, 141, -1.75, 0.0, 4)], BOX)
    assert needed
    assert direction in ("UP", "DOWN")


def test_shot_escape_takes_the_open_side_when_one_is_rock() -> None:
    nodes = frozenset(n for n in _all_nodes() if not (n[0] == 120 and n[1] < 141))
    direction, needed = shot_escape(
        120, 141, [(180, 141, -1.75, 0.0, 4)], BOX, nodes=nodes
    )
    assert needed and direction == "DOWN"


def test_shot_escape_does_not_run_along_a_diagonal_shot() -> None:
    # 0x7B z1 f=4657: shot down-right of Link flying up-left, UP walled.
    nodes = frozenset(n for n in _all_nodes() if n[1] >= 125)
    direction, needed = shot_escape(
        112, 125, [(129, 152, -1.5, -0.6, 4)], BOX, nodes=nodes
    )
    assert needed and direction != "LEFT"


def test_shot_escape_leaves_the_frame_when_nothing_can_be_hit() -> None:
    direction, needed = shot_escape(120, 141, [(40, 80, -1.75, 0.0, 4)], BOX)
    assert not needed


def _snap(**kw) -> ZeldaSnapshot:
    fields = dict(
        mode=PLAY_MODE, level=0, screen=0x39, next_screen=0x39, link_x=112,
        link_y=189, facing=8, sword=1, bombs=0, rupees=0, keys=0, health=0x20,
        triforce=0, compass=0, dialog_timer=0, colliding_tile=0,
        room_item_id=0, room_all_dead=0, room_obj_count=0, cur_opened_doors=0,
        open_doorway_mask=0, heart_partial=0x80,
        objects=(ZeldaObject(slot=0, type_id=0, x=112, y=189, facing=8, hp=0, state=0),),
    )
    fields.update(kw)
    return ZeldaSnapshot(**fields)


def test_pond_walks_onto_the_edge_row_then_waits_for_the_fill() -> None:
    ctl = PondFairyController()
    assert list(ctl.step(_snap()).action) != list(ctl.step(_snap(link_x=120)).action)
    at_edge = _snap(link_x=120, link_y=POND_EDGE_Y)
    for _ in range(40):
        ctl.step(at_edge)
    assert not ctl.success and not ctl.failed
    halted = (ZeldaObject(slot=0, type_id=0, x=120, y=POND_EDGE_Y, facing=8, hp=0, state=0x40),)
    full = dict(link_x=120, link_y=POND_EDGE_Y, health=0x22, heart_partial=0xFF)
    ctl.step(_snap(objects=halted, **full))
    assert not ctl.success  # the ROM still holds Link
    ctl.step(_snap(**full))
    assert ctl.success


def test_pond_refuses_a_screen_without_a_pond() -> None:
    ctl = PondFairyController()
    ctl.step(_snap(screen=0x38))
    assert ctl.failed and not ctl.success


def test_geo_goal_falls_back_to_the_nearest_open_edge_row() -> None:
    # East edge open only on y=141; the hop asks for 165 (0x79's case).
    nodes = frozenset(n for n in _all_nodes() if n[0] < 232 or n[1] == 141)
    goals = OverworldPathController._geo_goals(
        ScreenHop(0x7A, "RIGHT", align_y=165), nodes
    )
    assert goals == {(240, 141)}
