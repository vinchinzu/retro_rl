"""ROM-free checks for the shared TMNT IV Clean (heal=none) suite."""

from __future__ import annotations

import inspect

from retro_harness.ram_state import GameMode, GameState
from tmnt_iv.clean_suite import (
    CLEAN_SPECS,
    STAGE1_CLEAN,
    STAGE2_CLEAN,
    STAGE3_CLEAN,
    CleanProbeSpec,
    is_live_alleycat,
    is_live_big_apple,
    is_live_sewer,
)
from tmnt_iv.scripts import probe_stage1_clean, probe_stage2_clean, probe_stage3_clean


def _playing(
    *,
    stage: int,
    health: int = 80,
    player_x: int = 100,
    event: int = 0x0A,
    mode: GameMode = GameMode.PLAYING,
) -> GameState:
    return GameState(
        frame=1,
        mode=mode,
        stage=stage,
        health=health,
        player_x=player_x,
        extras={"event": event},
    )


def test_specs_exist_for_stage_bytes_0_1_2() -> None:
    assert set(CLEAN_SPECS) == {0, 1, 2}
    for byte, spec in CLEAN_SPECS.items():
        assert isinstance(spec, CleanProbeSpec)
        assert spec.stage_byte == byte
        assert spec.stop_stage_gt == byte
        assert spec.suite_states
        assert spec.evidence_dir == f"stage{byte + 1}_clean_track"
        assert callable(spec.is_live)


def test_suite_state_lists_match_current_tuples() -> None:
    assert STAGE1_CLEAN.suite_states == ("Stage1", "Stage1_BeforeBoss", "Boss")
    assert STAGE2_CLEAN.suite_states == (
        "Stage2",
        "Stage2_Clear_w17_cam27882",
        "Boss2",
    )
    assert STAGE3_CLEAN.suite_states == ("LiveHardStage3", "Boss3", "Stage3")
    assert STAGE3_CLEAN.default_state == "LiveHardStage3"


def test_extra_entry_flags() -> None:
    assert STAGE1_CLEAN.extra_entry == "power_on"
    assert STAGE2_CLEAN.extra_entry == "from_stage1_clear"
    assert STAGE3_CLEAN.extra_entry == "from_stage2_clear"
    assert STAGE1_CLEAN.boss_entry_hp_key == "baxter_entry_hp"
    assert STAGE3_CLEAN.detect_game_over
    assert STAGE3_CLEAN.strict_advance


def test_wrappers_export_run_clean_probe() -> None:
    for mod in (probe_stage1_clean, probe_stage2_clean, probe_stage3_clean):
        assert callable(mod.run_clean_probe)
        assert callable(mod.run_suite)
        assert callable(mod.main)
        assert "run_clean_probe" in mod.__all__


def test_wrapper_signatures_keep_stage_flags() -> None:
    s1 = inspect.signature(probe_stage1_clean.run_clean_probe)
    assert s1.parameters["state_name"].default == "Stage1"
    assert s1.parameters["max_frames"].default == 20000
    assert s1.parameters["stop_stage_gt"].default == 0
    assert "power_on" in s1.parameters

    s2 = inspect.signature(probe_stage2_clean.run_clean_probe)
    assert s2.parameters["state_name"].default == "Stage2"
    assert s2.parameters["stop_stage_gt"].default == 1
    assert "from_stage1_clear" in s2.parameters

    s3 = inspect.signature(probe_stage3_clean.run_clean_probe)
    assert s3.parameters["state_name"].default == "LiveHardStage3"
    assert s3.parameters["stop_stage_gt"].default == 2
    assert "from_stage2_clear" in s3.parameters


def test_live_predicates_distinguish_stages() -> None:
    apple = _playing(stage=0)
    alley = _playing(stage=1)
    sewer = _playing(stage=2)
    assert is_live_big_apple(apple) and not is_live_big_apple(alley)
    assert is_live_alleycat(alley) and not is_live_alleycat(apple)
    assert is_live_sewer(sewer) and not is_live_sewer(alley)
    assert STAGE1_CLEAN.is_live(apple) and not STAGE1_CLEAN.is_live(sewer)
    assert STAGE2_CLEAN.is_live(alley)
    assert STAGE3_CLEAN.is_live(sewer)

    low_hp = _playing(stage=0, health=30)
    assert not is_live_big_apple(low_hp)
    assert is_live_alleycat(_playing(stage=1, health=30))

    assert not is_live_big_apple(_playing(stage=0, player_x=0))
    assert not is_live_sewer(_playing(stage=2, event=0x00))
    assert not is_live_alleycat(_playing(stage=1, mode=GameMode.TITLE))
