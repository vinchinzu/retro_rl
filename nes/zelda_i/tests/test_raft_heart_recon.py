"""Offline checks for the rr-ps7.4.1 raft_heart partial recon (no ROM).

The live hop chain itself can only be re-verified against the emulator;
these tests just pin the grid arithmetic and the shape of the recorded
recon data so the module cannot silently drift from what was measured.
"""

from __future__ import annotations

from zelda_i.anchors import SCREEN_RAFT_HEART_DOCK, SCREEN_RAFT_HEART_ISLAND
from zelda_i.overworld.graph import screen_to_grid
from zelda_i.overworld.raft_heart import (
    raft_heart_recon_report,
)


def test_dock_and_island_share_a_column_one_row_apart():
    dock_col, dock_row = screen_to_grid(SCREEN_RAFT_HEART_DOCK)
    island_col, island_row = screen_to_grid(SCREEN_RAFT_HEART_ISLAND)
    assert dock_col == island_col == 15
    assert dock_row == 3
    assert island_row == dock_row - 1


def test_row7_from_start_hypothesis_lands_on_dock():
    # Doc hypothesis (OVERWORLD_DOORS.md): 0x77 start, x8 east, x4 north.
    start_col, start_row = screen_to_grid(0x77)
    assert start_col == 7 and start_row == 7
    east8 = ((start_col + 8) & 0xF) | (start_row << 4)
    assert east8 == 0x7F
    north4 = (east8 & 0xF) | ((start_row - 4) << 4)
    assert north4 == SCREEN_RAFT_HEART_DOCK


def test_recon_report_discloses_partial_status():
    report = raft_heart_recon_report()
    assert report["status"] == "partial_live_recon"
    assert report["route_eligible"] is False
    assert "heart_container_touch" in report["not_verified"]
