"""Offline L4 planner consumer over ALTTP escape capability edges."""

from __future__ import annotations

from alttp.opening_route.escape_graph import (
    CAP_FIGHTER_SWORD,
    CAP_LAMP,
    N_CASTLE_GROUNDS,
    N_ROOM_50,
    N_ROOM_61,
)
from alttp.opening_route.l4_plan import plan_escape, plan_escape_summary
from retro_harness.adventure.planner import PlanStatus


def test_plan_escape_grounds_to_room_50_with_lamp() -> None:
    """Natural house exit (lamp) should unlock continuous tip via sword acquire."""
    result = plan_escape(N_CASTLE_GROUNDS, N_ROOM_50, capabilities={CAP_LAMP})
    assert result.status is PlanStatus.FOUND
    assert result.found is True
    edge_ids = [e.edge_id for e in result.path]
    assert "hole_to_sword" in edge_ids
    assert "pocket_to_main_hall" in edge_ids
    assert CAP_FIGHTER_SWORD in {
        str(c) for c in result.final_progression.capabilities
    }
    # Continuous tip is room_50.
    assert result.final_progression.node == N_ROOM_50


def test_plan_escape_without_lamp_blocked_on_sewers_not_required_to_tip() -> None:
    """Empty inventory still reaches tip; sword is acquired on-path."""
    result = plan_escape(N_CASTLE_GROUNDS, N_ROOM_50, capabilities=frozenset())
    assert result.status is PlanStatus.FOUND
    assert any(e.edge_id == "hole_to_sword" for e in result.path)


def test_plan_escape_summary_record() -> None:
    summary = plan_escape_summary(N_CASTLE_GROUNDS, N_ROOM_61)
    assert summary["game"] == "alttp"
    assert summary["found"] is True
    assert summary["status"] == "FOUND"
    assert "pocket_to_main_hall" in summary["path_edge_ids"]
