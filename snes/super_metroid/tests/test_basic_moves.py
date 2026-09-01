"""ROM-free tests for basic Map Rando movement builders."""

from __future__ import annotations

import inspect

from super_metroid.routes.skills.basic_moves import (
    down_grab,
    ledge_grab,
    ledge_grab_action,
    ledge_grab_buttons,
    shoot_up,
    shoot_up_action,
)
from super_metroid.takeoff import spin_jump


def test_shoot_up_is_vertical_beam_not_shoulder() -> None:
    assert shoot_up_action() == ("UP", "X")
    src = inspect.getsource(shoot_up)
    assert "shoot_up_action()" in src
    assert '"R"' not in src
    assert '"L"' not in src


def test_ledge_grab_is_shoulder_unspin_not_down() -> None:
    assert ledge_grab_buttons("RIGHT") == ("RIGHT", "L")
    assert ledge_grab_buttons("LEFT", aim="UP") == ("LEFT", "R")
    assert ledge_grab_buttons() == ("L",)
    grab = ledge_grab_action("RIGHT", over_ledge=True)
    assert grab == ("RIGHT", "L")
    assert "A" not in grab
    assert "DOWN" not in grab
    assert "B" not in grab
    air = ledge_grab_action("RIGHT", over_ledge=False)
    assert air == spin_jump("RIGHT")
    assert "A" in air and "B" in air
    src = inspect.getsource(ledge_grab)
    assert "ledge_grab_buttons" in src
    down_src = inspect.getsource(down_grab)
    assert '"DOWN"' in down_src
    assert '"L"' not in down_src
