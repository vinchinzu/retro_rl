"""The Clean ladder must stay honest about its own evidence."""

from __future__ import annotations

from pathlib import Path

from zelda_i.spine.clean_tip import (
    CLEAN_LADDER,
    Blocker,
    Rung,
    blocked,
    by_blocker,
    next_open,
    render,
    tip,
    tool_for,
)

_PKG_ROOT = Path(__file__).resolve().parents[1]


def test_ids_are_unique() -> None:
    ids = [step.id for step in CLEAN_LADDER]
    assert len(ids) == len(set(ids))


def test_every_cited_residual_exists() -> None:
    """A row may not point at a document that is not in the tree."""
    missing = [
        step.residual
        for step in CLEAN_LADDER
        if step.residual and not (_PKG_ROOT / step.residual).is_file()
    ]
    assert not missing, missing


def test_blocked_rows_name_a_tool() -> None:
    """The point of a blocker class is that it says what to reach for."""
    for step in blocked():
        assert tool_for(step.blocker), step.id
        assert step.room, step.id


def test_blocked_rows_are_never_spine_green() -> None:
    for step in blocked():
        assert step.rung < Rung.SPINE_GREEN, step.id
        assert not step.proven


def test_tip_is_none_while_the_power_on_run_is_red() -> None:
    """M5 measured red 2026-09-11 (2/2) at clear33_key.

    A tip is a ROM claim. While the first row is red there is no tip, and
    `tip()` must say so rather than naming a row the ROM does not support.
    """
    assert tip() is None
    assert all(step.rung < Rung.SPINE_GREEN for step in CLEAN_LADDER)


def test_next_open_is_the_first_unproven_row() -> None:
    nxt = next_open()
    assert nxt is not None
    assert nxt.id == "l1_tf"
    assert nxt.open
    assert nxt.room == "L1 0x23"


def test_render_handles_a_missing_tip() -> None:
    assert "clean tip: NONE" in render()


def test_shared_blocker_classes_are_visible() -> None:
    groups = by_blocker()
    assert Blocker.SHOT_UNDODGEABLE in groups
    assert Blocker.FIRING_LINE in groups
    assert Blocker.BODY_UNDODGEABLE in groups
    # The two stand-line lanes are answered by the same threat tool.
    assert "off_line_step" in tool_for(Blocker.SHOT_UNDODGEABLE)
    assert "in_firing_line" in tool_for(Blocker.FIRING_LINE)
    assert "MIN_DODGE_BODY" in tool_for(Blocker.BODY_UNDODGEABLE)


def test_render_mentions_the_next_hop_and_blockers() -> None:
    text = render()
    assert "next open: l1_tf" in text
    assert "blockers by class:" in text
