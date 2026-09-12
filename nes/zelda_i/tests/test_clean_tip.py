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


def test_tip_is_the_last_contiguous_green() -> None:
    """M5 Clean is L1 only; nothing behind it may claim spine-green."""
    assert tip().id == "l1_tf"
    assert CLEAN_LADDER[0].rung is Rung.SPINE_GREEN
    assert all(step.rung < Rung.SPINE_GREEN for step in CLEAN_LADDER[1:])


def test_next_open_is_the_first_unproven_row() -> None:
    nxt = next_open()
    assert nxt is not None
    assert nxt.id == "l1_exit_ow_l2"
    assert nxt.open


def test_shared_blocker_classes_are_visible() -> None:
    groups = by_blocker()
    assert Blocker.SHOT_UNDODGEABLE in groups
    assert Blocker.FIRING_LINE in groups
    # The two stand-line lanes are answered by the same threat tool.
    assert "off_line_step" in tool_for(Blocker.SHOT_UNDODGEABLE)
    assert "in_firing_line" in tool_for(Blocker.FIRING_LINE)


def test_render_mentions_the_tip_and_next_hop() -> None:
    text = render()
    assert "clean tip: l1_tf" in text
    assert "next open: l1_exit_ow_l2" in text
    assert "blockers by class:" in text
