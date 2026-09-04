"""L7 0x1C FORCED_DIGDOGGER whistle-shrink + north 0x0C (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from zelda_i.dungeon.behaviors import DIGDOGGER_SHRUNK_TYPE, DIGDOGGER_TYPE
from zelda_i.level7.digdogger import (
    DEST,
    ROOM,
    WHISTLE_B_SLOT,
    WHISTLE_STAND,
    DigdoggerPhase,
    Level7ForcedDigdoggerController,
    make_level7_forced_digdogger_controller,
)
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 7)
    ram[ADDR_SCREEN] = fields.get("screen", ROOM)
    ram[ADDR_LINK_X] = fields.get("x", 16)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_SELECTED_ITEM] = fields.get("selected", 4)
    return ram


def _plant(
    ram: np.ndarray,
    slot: int,
    type_id: int,
    *,
    hp: int = 240,
    x: int = 132,
    y: int = 141,
) -> None:
    ram[ADDR_OBJ_TYPE + slot] = type_id
    ram[ADDR_OBJ_HP + slot] = hp
    ram[ADDR_LINK_X + slot] = x
    ram[ADDR_LINK_Y + slot] = y


def _env(ram: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(get_ram=lambda: ram)


def _bound(ram: np.ndarray) -> Level7ForcedDigdoggerController:
    controller = make_level7_forced_digdogger_controller()
    controller.bind_env(_env(ram))
    return controller


def _buttons(action) -> list[str]:
    return sorted(
        name
        for name, idx in NES_BUTTON_NAME_TO_INDEX.items()
        if idx is not None and int(action.action[idx])
    )


def test_public_constants() -> None:
    assert ROOM == 0x1C
    assert DEST == 0x0C
    assert WHISTLE_B_SLOT == 5
    assert WHISTLE_STAND == (120, 141)


def test_walks_toward_stand() -> None:
    ram = _ram(x=16, y=141, selected=4)
    _plant(ram, 1, DIGDOGGER_TYPE)
    controller = _bound(ram)
    action = controller.step(read_snapshot(ram))
    assert _buttons(action) == ["RIGHT"]
    ram[ADDR_LINK_Y] = 125
    action = controller.step(read_snapshot(ram))
    assert _buttons(action) == ["DOWN"]


def test_pause_select_emits_start_not_a_poke() -> None:
    ram = _ram(x=120, y=141, selected=4)
    _plant(ram, 1, DIGDOGGER_TYPE)
    before = int(ram[ADDR_SELECTED_ITEM])
    controller = _bound(ram)
    action = None
    for _ in range(40):
        action = controller.step(read_snapshot(ram))
        if _buttons(action) == ["START"]:
            break
        assert int(ram[ADDR_SELECTED_ITEM]) == before
    assert action is not None
    assert _buttons(action) == ["START"]
    assert action.reason == "pause_open"
    assert int(ram[ADDR_SELECTED_ITEM]) == before
    assert controller.report()["writes"] == 0
    assert controller.report()["normal_pause_input"] is True


def test_twelve_b_after_selected_equals_5() -> None:
    ram = _ram(x=120, y=141, selected=5)
    _plant(ram, 1, DIGDOGGER_TYPE)
    controller = _bound(ram)
    seen_b = 0
    for _ in range(40):
        before = ram.copy()
        action = controller.step(read_snapshot(ram))
        assert np.array_equal(ram, before)
        if _buttons(action) == ["B"]:
            seen_b += 1
        elif seen_b:
            break
    assert seen_b == 12
    assert int(ram[ADDR_SELECTED_ITEM]) == 5


def test_success_on_dest_0x0c_after_shrink() -> None:
    ram = _ram(x=120, y=141, selected=5)
    _plant(ram, 1, DIGDOGGER_TYPE, hp=240, x=132, y=141)
    controller = _bound(ram)
    for _ in range(40):
        action = controller.step(read_snapshot(ram))
        if _buttons(action) == ["B"]:
            break
    _plant(ram, 1, DIGDOGGER_SHRUNK_TYPE, hp=128, x=132, y=141)
    for _ in range(20):
        controller.step(read_snapshot(ram))
        if controller.shrunk:
            break
    assert controller.shrunk
    ram[ADDR_OBJ_TYPE + 1] = 0
    ram[ADDR_OBJ_HP + 1] = 0
    for _ in range(50):
        controller.step(read_snapshot(ram))
        if controller.killed:
            break
    assert controller.killed
    ram[ADDR_SCREEN] = DEST
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 205
    controller.step(read_snapshot(ram))
    assert controller.success
    assert not controller.failed
    leftover = controller.report()["leftover"]
    assert leftover is not None
    assert leftover["screen"] == DEST
    assert leftover["mode"] == PLAY_MODE


def test_already_in_dest_without_shrink_fails() -> None:
    ram = _ram(x=120, y=205, screen=DEST, selected=5)
    controller = _bound(ram)
    controller.step(read_snapshot(ram))
    assert controller.failed
    assert not controller.success
    assert "dest_without_kill_edge" in controller.notes


def test_fail_on_death() -> None:
    ram = _ram(x=16, y=141, mode=17)
    _plant(ram, 1, DIGDOGGER_TYPE)
    controller = _bound(ram)
    controller.step(read_snapshot(ram))
    assert controller.failed
    assert not controller.success
    assert "death" in controller.notes


def test_north_exit_holds_up_past_the_door_plane() -> None:
    ram = _ram(x=120, y=89, selected=5)
    controller = _bound(ram)
    controller.saw_boss = True
    controller.shrunk = True
    controller.killed = True
    controller.phase = DigdoggerPhase.EXIT
    action = controller.step(read_snapshot(ram))
    assert not controller.failed
    assert _buttons(action) == ["UP"]


def test_fail_if_leave_0x1c_to_non_dest_dungeon_room() -> None:
    ram = _ram(x=16, y=141, selected=4)
    _plant(ram, 1, DIGDOGGER_TYPE)
    controller = _bound(ram)
    controller.step(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x1B
    ram[ADDR_LINK_X] = 224
    controller.step(read_snapshot(ram))
    assert controller.failed
    assert not controller.success
    assert any("0x1b" in note for note in controller.notes)


def test_factory_never_shares_instances() -> None:
    assert make_level7_forced_digdogger_controller() is not (
        make_level7_forced_digdogger_controller()
    )


def test_report_route_eligible_is_false() -> None:
    controller = make_level7_forced_digdogger_controller()
    report = controller.report()
    assert report["route_eligible"] is False
    assert report["writes"] == 0
    assert report["spec_id"] == "level7_forced_digdogger"
    assert report["live_room"] == "0x1C"
    assert report["dest"] == "0x0C"
