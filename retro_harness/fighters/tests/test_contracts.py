"""Fighter PPO contract consumer tests."""

from __future__ import annotations

from retro_harness.fighters.contracts import build_fighter_contracts


def test_fighter_contract_binds_actions_rewards_and_wrapper_order(tmp_path) -> None:
    integration = tmp_path / "custom_integrations" / "Fixture-Snes"
    integration.mkdir(parents=True)
    (integration / "rom.sfc").write_bytes(b"rom")
    (integration / "Fight.state").write_bytes(b"state")
    bundle = build_fighter_contracts(
        game_id="Fixture-Snes",
        game_dir=tmp_path,
        state="Fight",
        action_maps=({}, {6: 1}, {7: 1}),
        reward_weights={"damage": 1.0, "time": -0.001},
        frame_skip=4,
        frame_stack=4,
        direct_ram=True,
        monitor=True,
    )

    assert bundle.environment.action_space_size == 3
    assert bundle.observation.fields[0].shape == (4, 84, 84)
    assert [wrapper.name for wrapper in bundle.wrappers.stack] == [
        "stable_retro.RetroEnv",
        "DirectRAMReader",
        "FrameSkip",
        "GrayscaleResize",
        "FightingEnv",
        "DiscreteAction",
        "FrameStack",
        "stable_baselines3.Monitor",
    ]
