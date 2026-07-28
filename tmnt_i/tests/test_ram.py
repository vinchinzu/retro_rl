from __future__ import annotations

import numpy as np

from tmnt_i.ram import is_level1_ready, parse_game_state


def test_parse_game_state_menu_by_default() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    state = parse_game_state(ram, frame=0)
    assert state.extras["ram_map_partial"] is True
