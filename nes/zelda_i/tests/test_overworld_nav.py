from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.dungeon import ids as dungeon_ids
from zelda_i.dungeon.ids import RUPEE_DROP_OBJECT_TYPE, RUPEE_DROP_STATE
from zelda_i.overworld.nav import (
    NavPhase,
    OverworldToLevel1Controller,
    level1_entrance_success,
)
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_STATE,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    ADDR_SWORD,
    PLAY_MODE,
    SCREEN_LEVEL1_ENTRANCE,
    SCREEN_START,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_START)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 160)
    ram[ADDR_HEALTH] = fields.get("health", 0x33)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    return ram


def test_level1_entrance_success_dungeon() -> None:
    assert level1_entrance_success(_ram(level=1, screen=0x73), require_dungeon=True)
    assert not level1_entrance_success(
        _ram(level=0, screen=SCREEN_LEVEL1_ENTRANCE, sword=1),
        require_dungeon=True,
    )


def test_controller_advances_on_screen_78() -> None:
    from zelda_i.ram import read_snapshot

    ctrl = OverworldToLevel1Controller()
    # Force a step on 77 first so phase is set
    ctrl.step(read_snapshot(_ram(screen=SCREEN_START, x=120, y=140, sword=1)))
    snap = read_snapshot(_ram(screen=0x78, x=20, y=140, sword=1))
    ctrl.step(snap)
    assert ctrl.phase is NavPhase.NORTH_78


def _ram_drop(
    *,
    screen: int,
    x: int,
    y: int,
    drop_x: int,
    drop_y: int,
    **fields: int,
) -> np.ndarray:
    ram = _ram(screen=screen, x=x, y=y, **fields)
    ram[ADDR_OBJ_TYPE + 1] = RUPEE_DROP_OBJECT_TYPE
    ram[ADDR_OBJ_STATE + 1] = RUPEE_DROP_STATE
    ram[ADDR_LINK_X + 1] = drop_x
    ram[ADDR_LINK_Y + 1] = drop_y
    return ram


def test_screen_78_scoops_nearby_rupee_with_need_rupees_zero() -> None:
    from zelda_i.ram import read_snapshot

    ctrl = OverworldToLevel1Controller()
    act = ctrl.step(
        read_snapshot(_ram_drop(screen=0x78, x=48, y=140, drop_x=80, drop_y=140))
    )
    assert ctrl.need_rupees == 0
    assert act.reason == "scoop_rupee"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert ctrl.phase is NavPhase.NORTH_78


def test_screen_77_far_west_drop_does_not_steal_east_exit() -> None:
    from zelda_i.ram import read_snapshot

    ctrl = OverworldToLevel1Controller()
    act = ctrl.step(
        read_snapshot(
            _ram_drop(screen=SCREEN_START, x=120, y=140, drop_x=16, drop_y=200)
        )
    )
    assert "scoop" not in act.reason
    assert ctrl.phase is NavPhase.EAST_77


def test_link_death_fails_closed() -> None:
    from zelda_i.ram import read_snapshot

    ctrl = OverworldToLevel1Controller()
    act = ctrl.step(read_snapshot(_ram(mode=17, screen=0x38, x=128, y=141)))
    assert act.reason == "link_death"
    assert ctrl.phase is NavPhase.FAILED


def test_low_hearts_divert_into_farm() -> None:
    from zelda_i.ram import read_snapshot

    ctrl = OverworldToLevel1Controller()
    assert ctrl.farm_below_hearts == 2
    ctrl.step(read_snapshot(_ram(screen=0x78, x=48, y=140, health=0x30)))
    assert ctrl.farm_attempts == 1
    assert any(note.startswith("farm_start_78") for note in ctrl.notes)


def test_start_screen_does_not_farm_without_prey() -> None:
    from zelda_i.ram import read_snapshot

    ctrl = OverworldToLevel1Controller()
    ctrl.step(read_snapshot(_ram(screen=SCREEN_START, x=120, y=140, health=0x30)))
    assert ctrl.farm_attempts == 0
    assert ctrl._farm is None


def test_screen_78_scoops_a_heart_twenty_px_ahead(monkeypatch) -> None:
    from zelda_i.overworld import nav as nav_mod
    from zelda_i.ram import read_snapshot

    heart_type = int(getattr(dungeon_ids, "HEART_DROP_OBJECT_TYPE", 0xFE))
    heart_state = int(getattr(dungeon_ids, "HEART_DROP_STATE", 0x22))
    monkeypatch.setattr(nav_mod, "HEART_FAIRY_DROP_TYPES", frozenset({heart_type}))
    ram = _ram(screen=0x78, x=48, y=140, health=0x32)
    ram[ADDR_OBJ_TYPE + 1] = heart_type
    ram[ADDR_OBJ_STATE + 1] = heart_state
    ram[ADDR_LINK_X + 1] = 48
    ram[ADDR_LINK_Y + 1] = 120
    ctrl = OverworldToLevel1Controller(farm_below_hearts=0)
    act = ctrl.step(read_snapshot(ram))
    assert ctrl.farm_attempts == 0
    assert act.reason == "scoop_heart"
    assert list(act.action) == list(nes_action("UP"))
    assert ctrl.phase is NavPhase.NORTH_78
