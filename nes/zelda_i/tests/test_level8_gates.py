"""L8 ``RoomHopSpec`` rows: structural invariants only.

Phase 2.3 folded eight near-identical L8 interior gate controllers into rows
of ``zelda_i.dungeon.door_hop.RoomHopController``.  Behaviour of each row is
covered by the per-room tests (west/south/east/stairs/cellar/passage/
triforce), which drive the factories; this file only checks what every row
must satisfy for the shared engine to run it.
"""

from __future__ import annotations

import pytest

from zelda_i.dungeon.door_hop import RoomHopSpec
from zelda_i.level8.cellar import CELLAR_RETURN_GATE
from zelda_i.level8.passage import PASSAGE_2F_GATE
from zelda_i.level8.path import (
    EAST_3E_GATE,
    SOUTH_1E_GATE,
    SOUTH_2E_GATE,
    WEST_1F_GATE,
)
from zelda_i.level8.stairs import STAIRS_3F_GATE
from zelda_i.level8.triforce import NORTH_3C_GATE

ALL_GATES: tuple[RoomHopSpec, ...] = (
    WEST_1F_GATE,
    SOUTH_1E_GATE,
    SOUTH_2E_GATE,
    EAST_3E_GATE,
    STAIRS_3F_GATE,
    NORTH_3C_GATE,
    CELLAR_RETURN_GATE,
    PASSAGE_2F_GATE,
)


@pytest.mark.parametrize("gate", ALL_GATES, ids=[g.spec_id for g in ALL_GATES])
def test_every_gate_is_level8_and_has_exactly_one_policy(gate: RoomHopSpec) -> None:
    assert gate.level == 8
    assert (gate.step is None) != (gate.policy_fn is None)
