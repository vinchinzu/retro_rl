"""Tests for the canonical named-button action API."""

from __future__ import annotations

import numpy as np
import pytest

from retro_harness.actions import (
    ActionBuilder,
    action_names,
    buttons_multi,
    idle_action_multi,
    indexed_action,
    snes_action,
)


def test_named_and_keyword_actions_share_one_layout() -> None:
    named = snes_action("RIGHT", "Y")
    keyword = snes_action(right=True, y=True, a=False)

    assert named == keyword
    assert action_names(named) == ("Y", "RIGHT")


def test_numpy_action_uses_requested_dtype() -> None:
    action = snes_action("A", dtype=np.int32)

    assert isinstance(action, np.ndarray)
    assert action.dtype == np.int32
    assert action[8] == 1


def test_indexed_action_validates_action_space() -> None:
    assert indexed_action([0, 7], action_size=8) == [1, 0, 0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="outside action size"):
        indexed_action([8], action_size=8)


def test_multiplayer_compatibility_helpers() -> None:
    action = buttons_multi(("A",), ("LEFT",))

    assert len(action) == 24
    assert action[8] == 1
    assert action[12 + 6] == 1
    assert idle_action_multi(players=3) == [0] * 36


def test_action_builder_delegates_to_named_builder() -> None:
    assert ActionBuilder().press("START", "A").build() == snes_action("START", "A")
