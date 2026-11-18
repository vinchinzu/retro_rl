"""Offline autobot board. The plan file is the fixture."""

from __future__ import annotations

from alttp.scripts.dungeon_board import load_rows, main


def test_twelve_dungeons_are_eight_steps_and_do_not_boot() -> None:
    rows = load_rows()
    assert len(rows) == 12
    assert all(len(row.steps) == 8 for row in rows)
    assert all(not row.may_boot for row in rows)
    assert main([]) == 0


def test_eastern_hall_stays_unbooted() -> None:
    eastern = next(row for row in load_rows() if row.slug == "eastern")
    assert eastern.next_open is not None
    assert eastern.next_open.ap_id == "AP-105"
    assert eastern.next_open.mark == " "
    assert any(step.ap_id == "AP-101" and step.mark == "!" for step in eastern.steps)
    gate = eastern.gate()
    assert "0xB9" in gate
    assert "Do not boot the hall" in gate
    assert "Do not walk the chain" not in gate


def test_desert_tablet_and_kholdstare_stay_claimed() -> None:
    rows = {row.slug: row for row in load_rows()}
    desert = rows["desert"]
    ice = rows["ice"]
    assert any(step.ap_id == "AP-113" and step.mark == "~" for step in desert.steps)
    assert any(step.ap_id == "AP-168" and step.mark == "~" for step in ice.steps)
    assert "Do not steal" in desert.gate()
    assert "Do not send another stand" in ice.gate()
