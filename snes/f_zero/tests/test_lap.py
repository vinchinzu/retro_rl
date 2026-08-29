from __future__ import annotations

from pathlib import Path

import pytest

from f_zero.paths import GAME_DIR, INTEGRATION_DIR, MUTE_CITY_STATE

_ROM = INTEGRATION_DIR / "rom.sfc"
_STATE = INTEGRATION_DIR / f"{MUTE_CITY_STATE}.state"


def _rom_ready() -> bool:
    try:
        return _STATE.is_file() and _ROM.resolve().is_file()
    except OSError:
        return False


pytestmark = pytest.mark.rom


@pytest.mark.skipif(not _rom_ready(), reason="F-Zero ROM or MuteCity.state missing")
def test_mute_city_centerline_completes_one_lap() -> None:
    from f_zero.scripts.run_mute_city_lap import run_mute_city_lap

    report = run_mute_city_lap(
        max_frames=4500,
        out_dir=Path(GAME_DIR) / "recordings" / "mute_city_lap_test",
    )
    assert report["success"] is True
    assert int(report["laps"]) >= 1
    assert report["crashed"] is False
    assert int(report["frames"]) < 4500
