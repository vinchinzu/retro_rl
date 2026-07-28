"""Integration: autobot 1-1 clear (isolated + natural-entry)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from smb.paths import INTEGRATION_V0_DIR
from smb.policy import DEFAULT_1_1_SEED


def _has_real_stable_retro() -> bool:
    try:
        import stable_retro as retro
    except ImportError:
        return False
    return hasattr(getattr(retro, "data", None), "Integrations") and hasattr(
        retro.data.Integrations, "CUSTOM"
    )


pytestmark = pytest.mark.skipif(
    not _has_real_stable_retro()
    or not (INTEGRATION_V0_DIR / "Level1_1.state").exists()
    or not (INTEGRATION_V0_DIR / "rom.nes").exists()
    or not DEFAULT_1_1_SEED.exists(),
    reason="real stable_retro + SMB v0 Level1_1 / ROM / seed required",
)


def test_run_1_1_isolated_clear(tmp_path: Path) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from smb.scripts.run_1_1 import run_1_1

    report = run_1_1(
        state_name="Level1_1",
        natural_entry=False,
        max_frames=4000,
        out_dir=tmp_path,
        save_clear=False,
        tag="pytest_1_1",
    )
    assert report["success"] is True
    assert report["outcome"] == "success"
    assert report["max_player_x"] >= 2500
    assert report["frames"] > 1500
    assert report["final"]["level_id"] != 0
    assert report["natural_entry"] is False


def test_run_1_1_natural_entry_clear(tmp_path: Path) -> None:
    """M4: power-on boot + 1-frame settle + seed clears 1-1."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from smb.scripts.run_1_1 import NATURAL_SETTLE_FRAMES, run_1_1

    report = run_1_1(
        natural_entry=True,
        max_frames=4000,
        out_dir=tmp_path,
        save_clear=False,
        tag="pytest_1_1_natural",
        natural_settle_frames=NATURAL_SETTLE_FRAMES,
    )
    assert report["success"] is True
    assert report["outcome"] == "success"
    assert report["natural_entry"] is True
    assert report["boot_frames"] > 200
    assert report["settle_frames"] == NATURAL_SETTLE_FRAMES
    assert report["max_player_x"] >= 2500
    assert report["final"]["level_id"] != 0


def test_warp_chain_segment_12_reaches_world_4(tmp_path: Path) -> None:
    """1-2 secret warp segment: mid-level state + seed → World 4."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from smb.paths import INTEGRATION_V0_DIR
    from smb.policy import DEFAULT_1_2_WARP_SEED
    from smb.scripts.run_warp_chain import run_warp_chain

    if not (INTEGRATION_V0_DIR / "Level1_2_WarpMid.state").exists():
        pytest.skip("Level1_2_WarpMid.state missing")
    if not DEFAULT_1_2_WARP_SEED.exists():
        pytest.skip("1-2 warp seed missing")

    report = run_warp_chain(
        mode="segment-12",
        out_dir=tmp_path,
        save_clear=False,
        tag="pytest_seg12",
    )
    assert report["success"] is True
    assert report["outcome"] == "world_4"
    assert report["final"]["world"] == 3
    assert report["final"]["reached_world_4"] is True


def test_warp_chain_natural_1_1_then_w4(tmp_path: Path) -> None:
    """Power-on → 1-1 → mid-1-2 warp load → World 4."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from smb.paths import INTEGRATION_V0_DIR
    from smb.scripts.run_warp_chain import run_warp_chain

    if not (INTEGRATION_V0_DIR / "Level1_2_WarpMid.state").exists():
        pytest.skip("Level1_2_WarpMid.state missing")

    report = run_warp_chain(
        mode="chain",
        out_dir=tmp_path,
        save_clear=False,
        tag="pytest_chain_w4",
    )
    assert report["success"] is True
    assert report["stages"]["1-1"]["success"] is True
    assert report["final"]["world"] == 3


def test_warp_suffix_reaches_stable_8_4_ending(tmp_path: Path) -> None:
    """M5: one continuous controller suffix clears 1-2 through 8-4."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from smb.policy import DEFAULT_WARP_SUFFIX_SEED
    from smb.scripts.run_warp_finish import run_warp_finish

    if not DEFAULT_WARP_SUFFIX_SEED.exists():
        pytest.skip("folded warp suffix seed missing")

    report = run_warp_finish(
        mode="suffix",
        out_dir=tmp_path,
        tag="pytest_warp_suffix",
    )
    suffix = report["stages"]["continuous_suffix"]
    assert report["success"] is True
    assert report["outcome"] == "ending"
    assert report["exits_completed"] == 7
    assert suffix["state_loads_during_suffix"] == 0
    assert suffix["policy_frames"] == 19_963
    assert suffix["ending_settle_frames"] == 120
    assert [row["exit_id"] for row in suffix["milestones"]] == [
        "1-2",
        "4-1",
        "4-2",
        "8-1",
        "8-2",
        "8-3",
        "8-4",
    ]
    assert suffix["final"]["oper_mode"] == 2
    assert suffix["final"]["lives"] == suffix["start"]["lives"]


def test_continuous_level1_1_to_ending_no_mid_splice(tmp_path: Path) -> None:
    """Level1_1 + settle + seed clears all eight exits with zero mid loads."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from smb.policy import CONTINUOUS_SETTLE_FRAMES, DEFAULT_CONTINUOUS_SEED
    from smb.scripts.run_warp_finish import run_warp_finish

    if not DEFAULT_CONTINUOUS_SEED.exists():
        pytest.skip("continuous 1-1→ending seed missing")

    report = run_warp_finish(
        mode="continuous",
        out_dir=tmp_path,
        tag="pytest_warp_continuous",
    )
    stage = report["stages"]["continuous"]
    assert report["success"] is True
    assert report["outcome"] == "ending"
    assert report["exits_completed"] == 8
    assert report["state_loads_during_attempt"] == 0
    assert report["stages"]["settle"]["frames"] == CONTINUOUS_SETTLE_FRAMES
    assert stage["state_loads_during_policy"] == 0
    assert stage["policy_frames"] == 21_731
    assert stage["ending_settle_frames"] == 120
    assert [row["exit_id"] for row in stage["milestones"]] == [
        "1-1",
        "1-2",
        "4-1",
        "4-2",
        "8-1",
        "8-2",
        "8-3",
        "8-4",
    ]
    assert stage["final"]["oper_mode"] == 2
    assert stage["final"]["lives"] == stage["start"]["lives"]


def test_poweron_clean_reaches_stable_8_4_ending(tmp_path: Path) -> None:
    """M7: power-on + fixed boot + continuous seed → ending, zero state loads."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from smb.policy import (
        DEFAULT_CONTINUOUS_SEED,
        POWERON_BOOT_FRAMES,
        POWERON_SETTLE_FRAMES,
    )
    from smb.scripts.run_warp_finish import run_warp_finish

    if not DEFAULT_CONTINUOUS_SEED.exists():
        pytest.skip("continuous seed missing")

    report = run_warp_finish(
        mode="poweron",
        out_dir=tmp_path,
        tag="pytest_warp_poweron",
    )
    stage = report["stages"]["continuous"]
    boot = report["stages"]["boot"]
    assert report["success"] is True
    assert report["outcome"] == "ending"
    assert report["benchmark_eligible"] is True
    assert report["intervention"]["class"] == "Clean"
    assert report["exits_completed"] == 8
    assert report["state_loads_during_attempt"] == 0
    assert boot["frames"] == POWERON_BOOT_FRAMES
    assert boot["settle_frames"] == POWERON_SETTLE_FRAMES
    assert stage["state_loads_during_policy"] == 0
    assert stage["policy_frames"] == 21_731
    assert stage["ending_settle_frames"] == 120
    assert [row["exit_id"] for row in stage["milestones"]] == [
        "1-1",
        "1-2",
        "4-1",
        "4-2",
        "8-1",
        "8-2",
        "8-3",
        "8-4",
    ]
    assert stage["final"]["oper_mode"] == 2
    assert stage["final"]["lives"] == stage["start"]["lives"]
