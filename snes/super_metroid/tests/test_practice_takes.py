"""ROM-free tests for practice_takes segments + tape coverage rows."""

from __future__ import annotations

import numpy as np

from super_metroid.plm import (
    ADDR_PLM_BLOCK,
    ADDR_PLM_ID,
    ADDR_PLM_INST,
    ADDR_ROOM_WIDTH,
    ADDR_SAMUS_PROJ_TYPE,
    ADDR_SAMUS_PROJ_X,
    ADDR_SAMUS_PROJ_Y,
    SHOT_BLOCK_PLM_IDS,
    coverage_trace,
    plms_from_compact,
    shot_block_spawns,
)
from super_metroid.scripts.record.practice_takes import SEGMENTS


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = (value >> 8) & 0xFF


def test_ws_main_to_attic_segment_from_hop1_pin() -> None:
    seg = SEGMENTS["ws-main-to-attic"]
    assert seg.start == "ws-main"
    assert seg.pure_hop == "ws-main-to-attic"
    assert seg.pure_source_rel == "scratch/post_ws_basement_to_main.state"
    assert seg.no_guide_default is True


def test_post_phantoon_to_gravity_segment_from_defeated_pin() -> None:
    seg = SEGMENTS["post-phantoon-to-gravity"]
    assert seg.start == "post-phantoon"
    assert seg.pure_hop is None
    assert seg.pure_source_rel == "scratch/post_phantoon_defeated.state"
    assert seg.no_guide_default is True
    assert "gravity_path_human" in seg.description


def test_coverage_trace_empty_ram() -> None:
    assert coverage_trace(None) == {"enemies": [], "plms": [], "projs": []}
    assert coverage_trace(np.zeros(8, dtype=np.uint8))["plms"] == []


def test_coverage_trace_plm_and_projectile() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _put_u16(ram, ADDR_ROOM_WIDTH, 16)
    _put_u16(ram, ADDR_PLM_ID, 0xB091)
    _put_u16(ram, ADDR_PLM_BLOCK, 16 * 10 + 5)
    _put_u16(ram, ADDR_PLM_INST, 0xAABB)
    _put_u16(ram, ADDR_SAMUS_PROJ_TYPE, 0x0005)
    _put_u16(ram, ADDR_SAMUS_PROJ_X, 1180)
    _put_u16(ram, ADDR_SAMUS_PROJ_Y, 1800)
    cov = coverage_trace(ram)
    assert cov["plms"] == [[0, 0xB091, 5 * 16 + 8, 10 * 16 + 8, 0xAABB]]
    assert cov["projs"] == [[0, 0x0005, 1180, 1800]]
    assert cov["enemies"] == []


def test_shot_block_spawns_from_take02_lip_rows() -> None:
    """Take02 f305→f306: UP+X at (1223,1860) p3 spawns 0xD080 + 0xD074."""
    before = plms_from_compact(
        [
            [32, 0xC842, 968, 2264, 48788],
            [33, 0xC848, 552, 3288, 48893],
            [34, 0xEEDB, 72, 2856, 57550],
        ]
    )
    after = plms_from_compact(
        [
            [30, 0xD080, 904, 3576, 0],
            [31, 0xD074, 872, 3608, 0],
            [32, 0xC842, 968, 2264, 48788],
            [33, 0xC848, 552, 3288, 48893],
            [34, 0xEEDB, 72, 2856, 57550],
        ]
    )
    assert 0xD080 in SHOT_BLOCK_PLM_IDS
    assert shot_block_spawns((), after) == ()
    spawned = shot_block_spawns(before, after)
    assert [row["id"] for row in spawned] == [0xD080, 0xD074]
    assert shot_block_spawns(after, after) == ()
    reused = plms_from_compact(
        [
            [32, 0xD080, 904, 3576, 0],
            [33, 0xC848, 552, 3288, 48893],
            [34, 0xEEDB, 72, 2856, 57550],
        ]
    )
    assert [row["id"] for row in shot_block_spawns(before, reused)] == [0xD080]
