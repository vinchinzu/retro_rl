"""Unit tests for the HIO search helpers (no emulator required)."""

from __future__ import annotations

import numpy as np

from hals_golf.core.ram import (
    WRAM_HOLE_INDEX,
    WRAM_LIE_TYPE,
    WRAM_REST_DISTANCE,
    WRAM_STROKE_COUNT,
)
from hals_golf.runtime.hio_search import CandidateResult, situation_from_ram
from hals_golf.tasks.shot_policy import ShotIntent


def test_situation_from_ram_reads_wram_bytes() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    ram[WRAM_HOLE_INDEX] = 3  # zero-based -> hole 4
    ram[WRAM_STROKE_COUNT] = 0
    ram[WRAM_REST_DISTANCE] = 200 & 0xFF
    ram[WRAM_REST_DISTANCE + 1] = 200 >> 8
    ram[WRAM_LIE_TYPE] = 1
    situation = situation_from_ram(ram, default_power=40)
    assert situation.hole == 4
    assert situation.strokes == 0
    assert situation.rest == 200
    assert situation.lie == 1
    assert situation.default_power == 40


def test_candidate_result_marks_hole_in_one() -> None:
    intent = ShotIntent(
        power=42,
        aim=0,
        club_downs=0,
        require_rest_change=False,
        complete_on_rest_zero=True,
    )
    hit = CandidateResult(
        index=0,
        intent=intent,
        start_rest=380,
        end_rest=0,
        end_strokes=1,
        frames=900,
        status="success",
    )
    miss = CandidateResult(
        index=1,
        intent=intent,
        start_rest=380,
        end_rest=120,
        end_strokes=1,
        frames=900,
        status="success",
    )
    assert hit.hole_in_one
    assert hit.rest_delta == 380
    assert not miss.hole_in_one
    assert miss.rest_delta == 260
