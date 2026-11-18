"""ROM eval for the run layer: boot, guarded play, tape replay.

``uv run pytest snes/tmnt_iv/tests/test_rom.py -m rom`` (needs the ROM;
the Sewer case also needs the local ``LiveHardStage3`` pin).
"""

from __future__ import annotations

import pytest

from tmnt_iv.paths import INTEGRATION_DIR


@pytest.mark.rom
@pytest.mark.rom_smoke
@pytest.mark.parametrize("turtle", ["mike", "raph"])
def test_power_on_boot_reaches_hard_play_with_the_chosen_turtle(turtle: str) -> None:
    from retro_harness.env import make_env, reset_obs
    from retro_harness.segment_runner import configure_headless
    from tmnt_iv.menus import TURTLE_CHAR, TURTLE_HP, hard_boot_action
    from tmnt_iv.paths import GAME, GAME_DIR
    from tmnt_iv.ram import parse_game_state

    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        reset_obs(env)
        for frame in range(3600):
            state = parse_game_state(env.get_ram(), frame=frame)
            if frame > 1490 and state.extras["menu"] == 6 and state.player_x > 0 and state.health:
                break
            env.step(hard_boot_action(frame, turtle))
        assert state.extras["char_id"] == TURTLE_CHAR[turtle]
        assert state.health == TURTLE_HP[turtle]
        assert state.extras["difficulty"] == 2
    finally:
        env.close()


@pytest.mark.rom
def test_guard_clears_sewer_clean_and_the_tape_replays(tmp_path) -> None:
    if not (INTEGRATION_DIR / "LiveHardStage3.state").exists():
        pytest.skip("local pin LiveHardStage3 missing")
    from tmnt_iv.scripts.run import run
    from tmnt_iv.tape import load_tape

    live = run(
        entry_state="LiveHardStage3",
        until_stage=4,
        turtle="leo",
        report_path=tmp_path / "sewer.json",
        max_frames=20_000,
    )
    assert live["success"], live["failure"]
    assert live["life_losses"] == 0 and live["state_loads"] == 0
    assert live["emergency_hp_writes"] == 0 and live["iframe_writes"] == 0
    assert live["rollout_restores"] > 0 and live["lookahead"]["desyncs"] == 0

    replay = run(
        entry_state="LiveHardStage3",
        until_stage=4,
        turtle="leo",
        tape=load_tape(live["tape"]),
        report_path=tmp_path / "sewer_replay.json",
        max_frames=20_000,
    )
    assert replay["success"] and replay["desync_frame"] is None
    assert replay["frames"] == live["frames"]
    assert replay["damage"] == live["damage"]
    assert replay["rollout_restores"] == 0
