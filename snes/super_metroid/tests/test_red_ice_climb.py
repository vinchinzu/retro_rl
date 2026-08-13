from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from super_metroid.paths import GAME_DIR
from super_metroid.routes.kpdr.k5.red_ice_climb import (
    BOTTOM_FLOOR,
    LOWER_RIPPER_1,
    RIPPER_ID,
    can_attach_bottom_edge,
    checkpoint_supported,
    read_rippers,
)


def _state(**overrides):
    values = {
        "room_id": 0xA253,
        "samus_x": 90,
        "samus_y": 2351,
        "velocity_y": 0,
        "vertical_direction": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Env:
    def __init__(self, ram: np.ndarray) -> None:
        self._ram = ram

    def get_ram(self) -> np.ndarray:
        return self._ram


def _write_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = (value >> 8) & 0xFF


def test_checkpoint_requires_grounded_state() -> None:
    assert BOTTOM_FLOOR.matches(_state(samus_y=2443, samus_x=120))
    assert LOWER_RIPPER_1.matches(_state())
    assert not LOWER_RIPPER_1.matches(_state(vertical_direction=2))
    assert not LOWER_RIPPER_1.matches(_state(samus_y=2300))


def test_bottom_edge_attach_requires_equipped_ice_and_hi_jump() -> None:
    ready = _state(
        samus_y=2443,
        samus_x=120,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_bottom_edge(ready)
    assert not can_attach_bottom_edge(_state(**{**vars(ready), "equipped_beams": 0x1005}))
    assert not can_attach_bottom_edge(_state(**{**vars(ready), "equipped_items": 0x3005}))


def test_frozen_support_is_part_of_checkpoint_truth() -> None:
    ram = np.zeros(0x20000, dtype=np.uint8)
    base = 0x0F78 + 5 * 0x40
    _write_u16(ram, base, RIPPER_ID)
    _write_u16(ram, base + 0x02, 92)
    _write_u16(ram, base + 0x06, 2376)
    _write_u16(ram, base + 0x26, 180)
    env = _Env(ram)

    assert read_rippers(env)[0].slot == 5
    assert checkpoint_supported(env, _state(), LOWER_RIPPER_1)

    _write_u16(ram, base + 0x26, 0)
    assert not checkpoint_supported(env, _state(), LOWER_RIPPER_1)


def test_checkpoint_plan_has_one_verified_edge_and_planned_recovery_tree() -> None:
    path = (
        GAME_DIR
        / "routes"
        / "kpdr"
        / "data"
        / "red_tower_ice_checkpoint_plan.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = {row["id"]: row for row in data["checkpoints"]}
    edges = {row["id"]: row for row in data["edges"]}

    assert data["kind"] == "super_metroid_checkpoint_room_plan"
    assert data["roomIdHex"] == "0xA253"
    assert len(checkpoints) >= 20
    assert edges["bottom_to_lower_ripper_1"]["status"] == "verified_phase_sweep"
    assert edges["lower_ripper_1_to_2"]["status"] == "planned"
    assert data["recovery"]
