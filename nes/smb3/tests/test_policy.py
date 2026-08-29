from __future__ import annotations

from smb3.policy import (
    DEFAULT_LEVEL1_POLICY,
    DEFAULT_LEVEL2_POLICY,
    STAGES,
    Level1Policy,
    enter_level1_script,
    enter_level2_script,
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


def test_enter_level2_script_is_two_right_hops() -> None:
    script = enter_level2_script()
    reasons = [frame.reason for frame in script]
    assert reasons.count("map_right") == 16
    assert "map_enter" in set(reasons)
    assert "map_up" not in set(reasons)
    assert STAGES["1-2"].enter is enter_level2_script
    assert STAGES["1-2"].policy_file == DEFAULT_LEVEL2_POLICY


def test_smb3_1_2_level_config_matches_stage() -> None:
    import smb3.platformer_levels  # noqa: F401
    from retro_harness.platformer.level_config import get_level_config

    cfg = get_level_config("smb3_1_2")
    assert cfg.start_state == STAGES["1-2"].start_state
    assert cfg.completion_ram_key == "goal_auto"


def test_level2_policy_file_loads() -> None:
    actions = load_action_indices(DEFAULT_LEVEL2_POLICY)
    assert len(actions) > 500
    assert DEFAULT_LEVEL2_POLICY.is_file()
    policy = Level1Policy.from_file(DEFAULT_LEVEL2_POLICY)
    assert len(policy) == len(actions)
    first = policy.tick()
    assert first.reason.startswith("a")
    assert len(first.action) == 9
