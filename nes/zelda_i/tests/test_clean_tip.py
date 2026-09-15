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


def test_blocked_rows_name_a_tool_and_a_place_to_look() -> None:
    """The point of a blocker class is that it says what to reach for.

    An ``invalid_pin`` row's locus is the fixture, not a room, so either
    field satisfies it — but a row that names neither is unactionable.
    """
    for step in blocked():
        assert tool_for(step.blocker), step.id
        assert step.room or step.pin, step.id


def test_blocked_rows_are_never_spine_green() -> None:
    for step in blocked():
        assert step.rung < Rung.SPINE_GREEN, step.id
        assert not step.proven


def test_tip_is_l1_tf_after_natural_entry_triforce() -> None:
    """M5 measured 2/2 2026-09-14, triforce=0x01, end 18909."""
    top = tip()
    assert top is not None
    assert top.id == "l1_tf"
    assert top.rung is Rung.SPINE_GREEN
    assert top.blocker is Blocker.NONE


def test_next_open_is_the_first_unproven_row() -> None:
    nxt = next_open()
    assert nxt is not None
    assert nxt.id == "pre_l1"
    assert nxt.open
    assert nxt.blocker is Blocker.INVENTORY_GAP


def test_render_names_the_l1_tip() -> None:
    assert "clean tip: l1_tf" in render()


def test_shared_blocker_classes_are_visible() -> None:
    """Rows move between classes; the class -> tool mapping is the contract."""
    groups = by_blocker()
    assert groups, "a ladder with no blocked row has nothing to dispatch"
    # The two stand-line lanes are answered by the same threat tool.
    assert "off_line_step" in tool_for(Blocker.SHOT_UNDODGEABLE)
    assert "in_firing_line" in tool_for(Blocker.FIRING_LINE)
    assert "MIN_DODGE_BODY" in tool_for(Blocker.BODY_UNDODGEABLE)
    assert all(tool_for(blocker) for blocker in groups)


def test_invalid_pin_rows_quote_the_byte() -> None:
    """A row that blames its fixture must say which byte is wrong."""
    for step in blocked():
        if step.blocker is Blocker.INVALID_PIN:
            assert "$066F" in step.pin, step.id


def test_render_mentions_the_next_hop_and_blockers() -> None:
    text = render()
    assert "next open: pre_l1" in text
    assert "blockers by class:" in text


def test_invalid_pin_is_its_own_blocker_class() -> None:
    """Not INVENTORY_GAP: that pin is poorer than a real arrival, this one
    holds a state the ROM cannot produce."""
    from zelda_i.spine.clean_tip import Blocker, tool_for

    assert Blocker.INVALID_PIN.value == "invalid_pin"
    tool = tool_for(Blocker.INVALID_PIN)
    assert "lo <= hi" in tool
    assert "capture_level6_entrance_fixture" in tool
