"""Shared pause-select machine. Fake RAM. Never writes ``$0656``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from zelda_i.dungeon.pause_select import (
    B_SLOT_BOMBS,
    B_SLOT_RECORDER,
    CLOSE_SETTLE_FRAMES,
    CURSOR_SETTLE_FRAMES,
    OPEN_SETTLE_FRAMES,
    PauseSelectController,
    PauseSelectPhase,
)
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    read_snapshot,
)


def _ram(*, selected: int = 1, mode: int = PLAY_MODE) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_SCREEN] = 0x42
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 141
    ram[ADDR_SELECTED_ITEM] = selected
    return ram


def _bound(want: int = B_SLOT_RECORDER, **fields: int) -> tuple[
    PauseSelectController, np.ndarray
]:
    ram = _ram(**fields)
    ctl = PauseSelectController(want=want)
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    return ctl, ram


def _buttons(action) -> list[str]:
    return sorted(
        name
        for name, idx in NES_BUTTON_NAME_TO_INDEX.items()
        if idx is not None and int(action.action[idx])
    )


def _step(ctl: PauseSelectController, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "pause-select must not write RAM"
    return act


def test_already_on_slot_skips_the_menu() -> None:
    ctl, ram = _bound(want=B_SLOT_RECORDER, selected=B_SLOT_RECORDER)
    act = _step(ctl, ram)
    assert ctl.success
    assert ctl.skipped
    assert not ctl.failed
    assert _buttons(act) == []
    assert "recorder_already_selected" in ctl.notes
    assert ctl.drive(read_snapshot(ram)) is None


def test_right_is_not_the_first_frame_after_start() -> None:
    ctl, ram = _bound(want=B_SLOT_RECORDER, selected=1)
    act = _step(ctl, ram)
    assert _buttons(act) == ["START"]
    assert act.reason == "pause_open"
    assert ctl.phase is PauseSelectPhase.OPEN_SETTLE
    for _ in range(OPEN_SETTLE_FRAMES):
        act = _step(ctl, ram)
        assert _buttons(act) == []
        assert "B" not in _buttons(act)
        assert "RIGHT" not in _buttons(act)
    act = _step(ctl, ram)
    assert _buttons(act) == ["RIGHT"]
    assert act.reason == "pause_next_item"


def test_noop_right_does_not_count_as_a_cursor_move() -> None:
    ctl, ram = _bound(want=B_SLOT_RECORDER, selected=1)
    _step(ctl, ram)
    for _ in range(OPEN_SETTLE_FRAMES):
        _step(ctl, ram)
    _step(ctl, ram)  # RIGHT, selected still 1
    assert ctl.cursor_moves == 0
    for _ in range(CURSOR_SETTLE_FRAMES):
        _step(ctl, ram)
    assert ctl.cursor_moves == 0
    act = _step(ctl, ram)
    assert _buttons(act) == ["RIGHT"]
    assert ctl.cursor_moves == 0


def test_b_forbidden_until_close_settle_completes() -> None:
    ctl, ram = _bound(want=B_SLOT_RECORDER, selected=1)
    assert _buttons(_step(ctl, ram)) == ["START"]
    for _ in range(OPEN_SETTLE_FRAMES):
        assert "B" not in _buttons(_step(ctl, ram))
    assert _buttons(_step(ctl, ram)) == ["RIGHT"]
    ram[ADDR_SELECTED_ITEM] = B_SLOT_RECORDER
    for _ in range(CURSOR_SETTLE_FRAMES):
        assert "B" not in _buttons(_step(ctl, ram))
    act = _step(ctl, ram)
    assert _buttons(act) == []
    act = _step(ctl, ram)
    assert _buttons(act) == ["START"]
    assert act.reason == "pause_close"
    for _ in range(CLOSE_SETTLE_FRAMES - 1):
        act = _step(ctl, ram)
        assert _buttons(act) == []
        assert not ctl.success
    act = _step(ctl, ram)
    assert ctl.success
    assert not ctl.failed
    assert _buttons(act) == []
    assert "B" not in _buttons(act)
    assert ctl.drive(read_snapshot(ram)) is None
    assert "recorder_selected_naturally" in ctl.notes


def test_env_not_bound_fails() -> None:
    ctl = PauseSelectController(want=B_SLOT_BOMBS)
    act = ctl.step(read_snapshot(_ram(selected=5)))
    assert ctl.failed
    assert act.reason == "pause_select_env_not_bound"
