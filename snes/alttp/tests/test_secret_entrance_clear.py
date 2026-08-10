"""Offline tests for post-sword → secret-entrance clear helpers."""

from __future__ import annotations

import numpy as np

from alttp.opening_route.secret_entrance_clear import (
    MAP_ID,
    STAIRS_ALIGN_X,
    STAIRS_DOOR_LABEL,
    approach_south_chamber,
    ensure_sword_control,
    evaluate_acceptance,
    evaluate_diagnostics,
    left_secret_entrance,
    run_from_sword,
    south_chamber_waypoints,
)
from alttp.ram import (
    DARK_WORLD_FLAG,
    EQUIP_SWORD,
    FOLLOWER,
    INDOORS,
    LINK_ACTION,
    LINK_X,
    LINK_Y,
    MODULE,
    ROOM_ID,
    SECRET_PASSAGE_ROOM,
    SUBMODULE,
    read_snapshot,
    wram_index,
    zelda_rescued_accepted,
)
from alttp.room_map import load_room_map


def _ram(writes: dict[int, int], *, size: int = 0x20000) -> np.ndarray:
    ram = np.zeros(size, dtype=np.uint8)
    for addr, value in writes.items():
        if addr < len(ram):
            ram[addr] = value & 0xFF
    return ram


def _passage(**extra: int) -> dict[int, int]:
    base = {
        MODULE: 0x07,
        SUBMODULE: 0x00,
        INDOORS: 1,
        DARK_WORLD_FLAG: 0,
        ROOM_ID: SECRET_PASSAGE_ROOM,
        wram_index(EQUIP_SWORD): 1,
        LINK_X: 2803 & 0xFF,
        LINK_X + 1: (2803 >> 8) & 0xFF,
        LINK_Y: 2680 & 0xFF,
        LINK_Y + 1: (2680 >> 8) & 0xFF,
    }
    base.update(extra)
    return base


def _set_xy(writes: dict[int, int], x: int, y: int) -> None:
    writes[LINK_X] = x & 0xFF
    writes[LINK_X + 1] = (x >> 8) & 0xFF
    writes[LINK_Y] = y & 0xFF
    writes[LINK_Y + 1] = (y >> 8) & 0xFF


def test_map_stairs_door_path() -> None:
    m = load_room_map(MAP_ID)
    door = m.door(STAIRS_DOOR_LABEL)
    assert door is not None
    assert door.outdoors is True
    assert "south_chamber" in door.path
    assert "stairs_align" in door.path
    wps = south_chamber_waypoints()
    assert len(wps) >= 1
    assert wps[-1].label == "south_chamber"


def test_evaluate_acceptance_zelda() -> None:
    writes = _passage()
    writes[wram_index(FOLLOWER)] = 1
    snap = read_snapshot(_ram(writes))
    acc = evaluate_acceptance(snap)
    diag = evaluate_diagnostics(snap)
    assert "zelda_follower" not in acc
    assert diag["zelda_follower"] is True
    assert acc["fighter_sword_ram"] is True
    assert acc["left_secret_entrance"] is False
    assert zelda_rescued_accepted(snap) is True


def test_left_secret_entrance_outdoors() -> None:
    writes = _passage()
    writes[INDOORS] = 0
    snap = read_snapshot(_ram(writes))
    assert left_secret_entrance(snap) is True
    acc = evaluate_acceptance(snap)
    assert acc["left_secret_entrance"] is True
    assert acc["in_secret_passage"] is False


def test_stairs_align_constant() -> None:
    assert STAIRS_ALIGN_X == 2672
    door = load_room_map(MAP_ID).door(STAIRS_DOOR_LABEL)
    assert door is not None
    assert door.approach_xy[0] == STAIRS_ALIGN_X


def test_hold_up_flag_blocks_acceptance_clear() -> None:
    writes = _passage()
    writes[LINK_ACTION] = 21
    snap = read_snapshot(_ram(writes))
    acc = evaluate_acceptance(snap)
    assert acc["hold_up_cleared"] is False
    assert snap.is_hold_up_item is True


class _FakeEm:
    def __init__(self) -> None:
        self._state = b"s0"

    def get_state(self) -> bytes:
        return self._state

    def set_state(self, state: bytes) -> None:
        self._state = state


class _FakeSwordEnv:
    """Simulates D-pad walking so map waypoints can be reached."""

    # SNES multi-hot: UP=4 DOWN=5 LEFT=6 RIGHT=7
    _UP, _DOWN, _LEFT, _RIGHT = 4, 5, 6, 7

    def __init__(self) -> None:
        self.em = _FakeEm()
        self.steps = 0
        self._writes = _passage()
        self._x = 2803
        self._y = 2680

    def get_ram(self) -> np.ndarray:
        return _ram(self._writes)

    def step(self, action: object) -> None:
        self.steps += 1
        # Clear hold-up immediately if present.
        self._writes[LINK_ACTION] = 0
        arr = np.asarray(action).reshape(-1)
        step = 3
        if len(arr) > self._LEFT and arr[self._LEFT] > 0:
            self._x -= step
        if len(arr) > self._RIGHT and arr[self._RIGHT] > 0:
            self._x += step
        if len(arr) > self._UP and arr[self._UP] > 0:
            self._y -= step
        if len(arr) > self._DOWN and arr[self._DOWN] > 0:
            self._y += step
        _set_xy(self._writes, self._x, self._y)


class _FakeExitEnv(_FakeSwordEnv):
    """After stairs approach, DOWN push transitions outdoors."""

    def step(self, action: object) -> None:
        super().step(action)
        arr = np.asarray(action).reshape(-1)
        # Once near stairs_align and holding DOWN, leave the secret entrance.
        near_stairs = abs(self._x - 2672) <= 12 and abs(self._y - 2916) <= 12
        holding_down = len(arr) > self._DOWN and arr[self._DOWN] > 0
        if near_stairs and holding_down and self.steps > 40:
            self._writes[INDOORS] = 0
            self._writes[MODULE] = 0x09
            self._x, self._y = 2248, 1755
            _set_xy(self._writes, self._x, self._y)


def test_run_from_sword_controller_smoke() -> None:
    env = _FakeSwordEnv()
    result = run_from_sword(
        env,
        source="state_load_dev",
        phases=[ensure_sword_control, approach_south_chamber],
    )
    report = result.to_report()
    assert report["kind"] == "alttp_secret_entrance_clear_report"
    assert report["development_only"] is True
    assert result.ok is False  # no exit / Zelda in this smoke
    assert any(p["phase"] == "ensure_sword_control" for p in report["phases"])


def test_run_from_sword_exit_ok() -> None:
    env = _FakeExitEnv()
    result = run_from_sword(env, source="state_load_dev")
    assert result.acceptance["left_secret_entrance"] is True
    assert result.ok is True
    assert result.phase == "secret_entrance_exited"
