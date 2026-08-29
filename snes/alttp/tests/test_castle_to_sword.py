"""Offline tests for castle→secret-entrance→sword route predicates/controller."""

from __future__ import annotations

import numpy as np

from alttp.opening_route.castle_to_sword import (
    BUSH_LIFT_CANDIDATES,
    HOLE_DOOR_LABEL,
    MAP_ID,
    SECRET_HOLE_ENTRY_SCRIPT,
    evaluate_acceptance,
    hole_approach_waypoints,
    run_from_castle_grounds,
)
from alttp.ram import (
    DARK_WORLD_FLAG,
    EQUIP_SWORD,
    HYRULE_CASTLE_SCREEN,
    INDOORS,
    MODULE,
    ROOM_ID,
    SCREEN_ID,
    SECRET_HOLE_WORLD_X,
    SECRET_HOLE_WORLD_Y,
    SECRET_PASSAGE_ROOM,
    SUBMODULE,
    WRAM_IDX,
    castle_entry_accepted,
    has_fighter_sword,
    in_secret_passage,
    read_snapshot,
    read_sword_level,
    secret_passage_accepted,
    snapshot_to_diag,
    uncle_sword_event_accepted,
    wram_index,
)


def _ram(writes: dict[int, int], *, size: int = 0x20000) -> np.ndarray:
    ram = np.zeros(size, dtype=np.uint8)
    for addr, value in writes.items():
        if addr < len(ram):
            ram[addr] = value & 0xFF
    return ram


def _playable_outdoors(**extra: int) -> dict[int, int]:
    base = {
        MODULE: 0x09,
        SUBMODULE: 0x00,
        SCREEN_ID: HYRULE_CASTLE_SCREEN,
        INDOORS: 0,
        DARK_WORLD_FLAG: 0,
    }
    base.update(extra)
    return base


def test_wram_index_high_sword() -> None:
    assert wram_index(EQUIP_SWORD) == WRAM_IDX + EQUIP_SWORD
    assert wram_index(0x10) == 0x10


def test_sword_level_high_wram() -> None:
    ram = _ram({wram_index(EQUIP_SWORD): 1})
    assert read_sword_level(ram) == 1
    snap = read_snapshot(ram)
    assert snap.has_fighter_sword is True
    assert uncle_sword_event_accepted(snap) is True


def test_no_sword_default() -> None:
    snap = read_snapshot(_ram(_playable_outdoors()))
    assert snap.sword_level == 0
    assert snap.has_fighter_sword is False
    assert uncle_sword_event_accepted(snap) is False


def test_secret_passage_acceptance() -> None:
    ram = _ram(
        {
            MODULE: 0x07,
            SUBMODULE: 0x00,
            INDOORS: 1,
            DARK_WORLD_FLAG: 0,
            ROOM_ID: SECRET_PASSAGE_ROOM,
            ROOM_ID + 1: 0,
        }
    )
    # room is u16 little-endian at ROOM_ID
    ram[ROOM_ID] = SECRET_PASSAGE_ROOM & 0xFF
    ram[ROOM_ID + 1] = 0
    snap = read_snapshot(ram)
    assert snap.in_secret_passage is True
    assert secret_passage_accepted(snap) is True
    assert castle_entry_accepted(snap) is True


def test_near_secret_hole() -> None:
    # link x/y are u16 at LINK_X/LINK_Y
    from alttp.ram import LINK_X, LINK_Y

    writes = _playable_outdoors()
    writes[LINK_X] = SECRET_HOLE_WORLD_X & 0xFF
    writes[LINK_X + 1] = (SECRET_HOLE_WORLD_X >> 8) & 0xFF
    writes[LINK_Y] = SECRET_HOLE_WORLD_Y & 0xFF
    writes[LINK_Y + 1] = (SECRET_HOLE_WORLD_Y >> 8) & 0xFF
    snap = read_snapshot(_ram(writes))
    assert snap.near_secret_hole is True
    assert snap.on_castle_grounds is True


def test_evaluate_acceptance_flags() -> None:
    from alttp.ram import LINK_X, LINK_Y

    writes = _playable_outdoors()
    writes[LINK_X] = SECRET_HOLE_WORLD_X & 0xFF
    writes[LINK_X + 1] = (SECRET_HOLE_WORLD_X >> 8) & 0xFF
    writes[LINK_Y] = SECRET_HOLE_WORLD_Y & 0xFF
    writes[LINK_Y + 1] = (SECRET_HOLE_WORLD_Y >> 8) & 0xFF
    snap = read_snapshot(_ram(writes))
    acc = evaluate_acceptance(snap)
    assert acc["near_secret_hole"] is True
    assert acc["on_castle_grounds"] is True
    assert acc["fighter_sword_ram"] is False
    assert acc["castle_entry"] is False


def test_snapshot_to_diag_keys() -> None:
    snap = read_snapshot(_ram(_playable_outdoors()))
    diag = snapshot_to_diag(snap)
    for key in (
        "game_mode",
        "room_base_id",
        "indoors",
        "sword_level",
        "near_secret_hole",
        "has_fighter_sword",
    ):
        assert key in diag


def test_env_predicates_use_get_ram() -> None:
    class _Env:
        def __init__(self, ram: np.ndarray) -> None:
            self._ram = ram

        def get_ram(self) -> np.ndarray:
            return self._ram

    # Indoors passage
    ram = _ram(
        {
            MODULE: 0x07,
            SUBMODULE: 0x00,
            INDOORS: 1,
            DARK_WORLD_FLAG: 0,
            ROOM_ID: SECRET_PASSAGE_ROOM,
        }
    )
    env = _Env(ram)
    assert in_secret_passage(env) is True

    ram2 = _ram({wram_index(EQUIP_SWORD): 1})
    assert has_fighter_sword(_Env(ram2)) is True


def test_approach_script_nonempty_and_candidates() -> None:
    from alttp.room_map import load_room_map

    m = load_room_map(MAP_ID)
    door = m.door(HOLE_DOOR_LABEL)
    assert door is not None
    assert door.approach_xy == (2430, 1704)
    wps = hole_approach_waypoints()
    assert len(wps) >= 8
    assert wps[-1].label in {"secret_hole_approach", f"{HOLE_DOOR_LABEL}_approach"}
    assert (wps[-1].x, wps[-1].y) == door.approach_xy
    assert m.point("secret_hole") is not None
    assert m.point("grounds_spawn") is not None
    assert len(BUSH_LIFT_CANDIDATES) >= 3
    assert SECRET_HOLE_ENTRY_SCRIPT[0][0] == ("UP",)
    assert any(buttons == ("A",) for buttons, _ in SECRET_HOLE_ENTRY_SCRIPT)
    assert BUSH_LIFT_CANDIDATES[0] == SECRET_HOLE_ENTRY_SCRIPT


class _FakeEm:
    def __init__(self, state: bytes = b"state0") -> None:
        self._state = state

    def get_state(self) -> bytes:
        return self._state

    def set_state(self, state: bytes) -> None:
        self._state = state


class _FakeRouteEnv:
    """Minimal env: stays on castle grounds; never enters indoors."""

    def __init__(self) -> None:
        self.em = _FakeEm()
        self.steps = 0
        from alttp.ram import LINK_X, LINK_Y

        self._writes = _playable_outdoors()
        # Spawn-ish coords from measured castle grounds
        self._writes[LINK_X] = 2386 & 0xFF
        self._writes[LINK_X + 1] = (2386 >> 8) & 0xFF
        self._writes[LINK_Y] = 2528 & 0xFF
        self._writes[LINK_Y + 1] = (2528 >> 8) & 0xFF

    def get_ram(self) -> np.ndarray:
        return _ram(self._writes)

    def step(self, _action: object) -> None:
        self.steps += 1
        # After enough steps, fake arrival at the map hole-approach tile.
        if self.steps > 40:
            from alttp.ram import LINK_X, LINK_Y
            from alttp.opening_route.castle_to_sword import hole_door

            ax, ay = hole_door().approach_xy
            self._writes[LINK_X] = ax & 0xFF
            self._writes[LINK_X + 1] = (ax >> 8) & 0xFF
            self._writes[LINK_Y] = ay & 0xFF
            self._writes[LINK_Y + 1] = (ay >> 8) & 0xFF


def test_run_from_castle_grounds_controller_smoke() -> None:
    env = _FakeRouteEnv()
    result = run_from_castle_grounds(
        env, source="state_load_dev", include_entry=True, include_uncle=False
    )
    assert result.source == "state_load_dev"
    assert result.development_only if hasattr(result, "development_only") else True
    report = result.to_report()
    assert report["kind"] == "alttp_castle_to_sword_report"
    assert "acceptance" in report
    assert "blocker" in report
    assert any(p["phase"] == "approach_secret_hole" for p in report["phases"])
    # Fake env never gets sword.
    assert result.ok is False
    assert report["development_only"] is True
