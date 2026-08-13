"""Live Level1_1 residual profiles (requires ROM + stable-retro)."""

from __future__ import annotations

import os

import pytest

from smb.paths import INTEGRATION_V0_DIR
from smb.residual_harness import measure_segment


def _has_real_stable_retro() -> bool:
    try:
        import stable_retro as retro
    except ImportError:
        return False
    return hasattr(getattr(retro, "data", None), "Integrations") and hasattr(
        retro.data.Integrations, "CUSTOM"
    )


pytestmark = [
    pytest.mark.rom,
    pytest.mark.skipif(
        not _has_real_stable_retro()
        or not (INTEGRATION_V0_DIR / "Level1_1.state").exists()
        or not (INTEGRATION_V0_DIR / "rom.nes").exists(),
        reason="real stable_retro + SMB v0 Level1_1 / ROM required",
    ),
]


_SHORT_TAPES = ("idle", "walk", "jump", "run_jump")
_LAND_TAPES = ("jump_to_land", "run_jump_to_land")
_AIR_X_TAPES = ("run_then_jump",)


@pytest.mark.parametrize("name", _SHORT_TAPES + _LAND_TAPES + _AIR_X_TAPES)
def test_segment_produces_measured_profile(name: str) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    result = measure_segment(name, run_emulator=True)
    assert result.emu_obs is not None
    assert result.profile.unmeasured is False
    assert result.horizon == len(result.emu_obs)
    assert result.approx_obs[0].x == result.emu_obs[0].x
    assert result.approx_obs[0].y == result.emu_obs[0].y


def test_short_tapes_hold_pixels_and_subpixels() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    for name in _SHORT_TAPES + _LAND_TAPES + _AIR_X_TAPES:
        result = measure_segment(name, run_emulator=True)
        assert result.profile.fd_pi is None, name
        assert result.profile.fd_sigma is None, name
        assert result.profile.can_keep_as_search_model() is True, name
