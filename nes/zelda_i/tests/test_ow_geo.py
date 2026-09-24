"""The overworld lattice rung (``OverworldPathController._rung_geo``).

It declines when the fall-through align-then-push would already walk to the
hop's edge on floor. That judgement must model the fall-through as it is: a
horizontal hop with no ``align_y`` pushes along Link's own row.
"""

from __future__ import annotations

from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController


def _nodes_0x5b_south_west() -> frozenset[tuple[int, int]]:
    """0x5B's west side from the live $6530 lattice (BFS_5B, 2026-09-23):
    rows 85..189 run to the west edge; rows 197..213 are only the x=48..80
    column Link climbs out of 0x6B on."""
    open_rows = {(x, y) for x in range(0, 88, 8) for y in range(85, 190, 8)}
    column = {(x, y) for x in range(48, 88, 8) for y in range(197, 214, 8)}
    return frozenset(open_rows | column)


def test_unaligned_left_hop_is_not_clear_from_a_row_that_is_rock_west() -> None:
    """2026-09-23 chain: back on 0x5B at (48, 205) for the 0x5A LEFT hop,
    the rung declined (row 189 reaches the edge), the fall-through pushed
    LEFT on row 205 into rock, and ``unstick_wait`` held 9750 frames."""
    nodes = _nodes_0x5b_south_west()
    hop = ScreenHop(0x5A, "LEFT")
    goals = {n for n in nodes if n[0] == 0}
    assert not OverworldPathController._geo_direct_clear(hop, nodes, (48, 205), goals)
    # The same hop from an open row is still left to the fall-through.
    assert OverworldPathController._geo_direct_clear(hop, nodes, (48, 181), goals)


def test_aligned_left_hop_is_clear_when_the_align_row_reaches_the_edge() -> None:
    """With ``align_y`` the fall-through does move to that row first."""
    nodes = _nodes_0x5b_south_west()
    hop = ScreenHop(0x5A, "LEFT", align_y=189)
    goals = {n for n in nodes if n[0] == 0 and n[1] == 189}
    assert OverworldPathController._geo_direct_clear(hop, nodes, (48, 205), goals)
