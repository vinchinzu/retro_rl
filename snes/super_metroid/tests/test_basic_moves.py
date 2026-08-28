"""ROM-free tests for basic Map Rando movement builders."""

from __future__ import annotations

import inspect

from super_metroid.routes.skills.basic_moves import shoot_up, shoot_up_action


def test_shoot_up_is_vertical_beam_not_shoulder() -> None:
    assert shoot_up_action() == ("UP", "X")
    src = inspect.getsource(shoot_up)
    assert "shoot_up_action()" in src
    assert '"R"' not in src
    assert '"L"' not in src
