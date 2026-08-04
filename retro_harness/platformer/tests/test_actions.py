"""Tests for actions and BK2 extraction (no ROM needed)."""

from retro_harness.platformer.actions import (
    DEFAULT_PLATFORMER_ACTIONS,
    NUM_BUTTONS,
    action_index_to_buttons,
    buttons_to_action_index,
)


def test_action_table_dimensions():
    assert len(DEFAULT_PLATFORMER_ACTIONS) == 14
    for action in DEFAULT_PLATFORMER_ACTIONS:
        assert len(action) == NUM_BUTTONS
        assert all(b in (0, 1) for b in action)


def test_nothing_action():
    buttons = action_index_to_buttons(0)
    assert all(b == 0 for b in buttons)


def test_round_trip():
    """Every action index maps to buttons that map back to the same index."""
    for idx in range(len(DEFAULT_PLATFORMER_ACTIONS)):
        buttons = action_index_to_buttons(idx)
        recovered = buttons_to_action_index(buttons)
        assert recovered == idx, f"Action {idx} round-trip failed: got {recovered}"


def test_invalid_index_returns_nothing():
    buttons = action_index_to_buttons(999)
    assert all(b == 0 for b in buttons)


def test_custom_action_table():
    custom = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    buttons = action_index_to_buttons(1, action_table=custom)
    assert buttons[1] == 1  # Y button
    assert buttons[0] == 0
