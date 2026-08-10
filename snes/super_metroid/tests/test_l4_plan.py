"""Offline L4 planner consumer over Super Metroid Morph edges."""

from __future__ import annotations

from retro_harness.adventure.planner import PlanStatus
from super_metroid.progression.l4_plan import (
    ROOM_CERES_ELEVATOR,
    ROOM_LANDING_SITE,
    ROOM_MORPH,
    plan_morph,
    plan_morph_summary,
)
from super_metroid.routes.kpdr.room_ids import ROOM_PARLOR


def test_plan_morph_ceres_to_morph() -> None:
    result = plan_morph(ROOM_CERES_ELEVATOR, ROOM_MORPH)
    assert result.status is PlanStatus.FOUND
    assert result.found is True
    assert result.final_progression.node == ROOM_MORPH
    assert len(result.path) >= 3
    # Final edge lands in Morph Ball room.
    assert result.path[-1].target_id == ROOM_MORPH


def test_plan_morph_landing_to_parlor() -> None:
    result = plan_morph(ROOM_LANDING_SITE, ROOM_PARLOR)
    assert result.status is PlanStatus.FOUND
    assert result.path
    assert result.final_progression.node == ROOM_PARLOR


def test_plan_morph_summary_record() -> None:
    summary = plan_morph_summary(ROOM_CERES_ELEVATOR, ROOM_MORPH)
    assert summary["game"] == "super_metroid"
    assert summary["subgraph"] == "morph"
    assert summary["found"] is True
    assert summary["status"] == "FOUND"
    assert summary["path_edge_ids"]
