from __future__ import annotations

import pytest

from magical_quest.paths import GAME, GAME_DIR, STAGE1_STATE
from magical_quest.policy import run_first_room
from retro_harness.env import get_available_states


@pytest.mark.rom
@pytest.mark.rom_smoke
def test_right_reaches_first_door_alive(tmp_path) -> None:
    if STAGE1_STATE not in get_available_states(GAME, GAME_DIR):
        pytest.skip("Stage1.state is missing")
    results = [
        run_first_room(out_dir=tmp_path / f"attempt_{index}")
        for index in range(2)
    ]
    assert all(report["success"] for report in results)
    assert all(int(report["end_health"]) > 0 for report in results)
    assert all(bool(report["at_first_door"]) for report in results)
    assert all(int(report["frames"]) <= 900 for report in results)
