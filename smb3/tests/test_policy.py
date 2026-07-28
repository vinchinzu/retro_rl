from __future__ import annotations

from smb3.policy import (
    DEFAULT_LEVEL1_POLICY,
    Level1Policy,
    enter_level1_script,
    load_action_indices,
)


def test_level1_policy_file_loads() -> None:
    actions = load_action_indices()
    assert len(actions) > 500
    assert DEFAULT_LEVEL1_POLICY.is_file()
    policy = Level1Policy.from_file()
    assert len(policy) == len(actions)
    first = policy.tick()
    assert first.reason.startswith("a")
    assert len(first.action) == 9


def test_enter_level1_script_has_move_and_confirm() -> None:
    script = enter_level1_script()
    reasons = {frame.reason for frame in script}
    assert "map_right" in reasons
    assert "map_up" in reasons
    assert "map_enter" in reasons
