"""Unit + light integration tests for smb.tas 1-1 tooling."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from smb.paths import INTEGRATION_V0_DIR, MODELS_DIR
from smb.policy import DEFAULT_1_1_SEED
from smb.tas.windows import ISOLATED_1_1_WINDOWS, discover_windows, windows_from_labels
from smb.tas.trace import SeedTrace, TraceEvent


def _has_real_stable_retro() -> bool:
    try:
        import stable_retro as retro
    except ImportError:
        return False
    return hasattr(getattr(retro, "data", None), "Integrations") and hasattr(
        retro.data.Integrations, "CUSTOM"
    )


def test_windows_from_labels_static() -> None:
    wins = windows_from_labels(["stairs", "100:200"], seed_len=2000, flag_frame=1300)
    assert len(wins) == 2
    assert wins[0].label == "stairs"
    assert wins[0].end <= 1298
    assert wins[1].start == 100
    assert wins[1].end == 200


def test_discover_windows_from_synthetic_trace() -> None:
    tr = SeedTrace(
        num_frames=1500,
        completed=True,
        flag_frame=1200,
        leave_frame=1900,
        max_player_x=3200,
        wall_slams=[TraceEvent(1100, "wall_slam", 2960, 100, "prev_xs=40")],
        stalls=[{"start": 400, "length": 40, "x": 900, "reason": "no_progress"}],
        xs_zero_runs=[{"start": 1090, "length": 20, "x": 2960}],
    )
    wins = discover_windows(tr, seed_len=1500, max_windows=10)
    labels = {w.label for w in wins}
    assert "stairs" in labels or any(w.start >= 1000 for w in wins)
    assert all(w.end <= 1198 for w in wins)  # pre-flag clamp


def test_isolated_windows_nonempty() -> None:
    assert len(ISOLATED_1_1_WINDOWS) >= 3
    assert any(w.label == "stairs" for w in ISOLATED_1_1_WINDOWS)


@pytest.mark.skipif(
    not _has_real_stable_retro()
    or not (INTEGRATION_V0_DIR / "Level1_1.state").exists()
    or not (INTEGRATION_V0_DIR / "rom.nes").exists()
    or not DEFAULT_1_1_SEED.exists(),
    reason="real stable_retro + SMB v0 Level1_1 / ROM / seed required",
)
def test_trace_clear_seed_completes() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from retro_harness.platformer.frame_tools import load_raw_frames
    from smb.tas.pipeline import ensure_completing_seed

    frames = load_raw_frames(DEFAULT_1_1_SEED)
    padded, tr = ensure_completing_seed(frames)
    assert tr.completed is True
    assert tr.flag_frame is not None
    assert tr.leave_frame is not None
    assert tr.leave_frame <= len(padded)
    assert tr.max_player_x >= 2500


@pytest.mark.skipif(
    not _has_real_stable_retro()
    or not (INTEGRATION_V0_DIR / "Level1_1.state").exists()
    or not DEFAULT_1_1_SEED.exists(),
    reason="real stable_retro + seed required",
)
def test_save_nes9_roundtrip(tmp_path: Path) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from retro_harness.platformer.frame_tools import load_raw_frames
    from smb.tas.pipeline import save_nes9_seed

    frames = load_raw_frames(DEFAULT_1_1_SEED)[:100]
    out = tmp_path / "tiny.json"
    save_nes9_seed(out, frames, metadata={"note": "test"})
    back = load_raw_frames(out)
    assert len(back) == 100
    assert back[0][:9] == [int(b) for b in frames[0][:9]]
