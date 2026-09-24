"""L7 0x1C FORCED_DIGDOGGER whistle-shrink + north 0x0C (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from zelda_i.dungeon.behaviors import DIGDOGGER_SHRUNK_TYPE, DIGDOGGER_TYPE
from zelda_i.level7.digdogger import (
    DEST,
    ROOM,
    DigdoggerPhase,
    Level7ForcedDigdoggerController,
    make_level7_forced_digdogger_controller,
)
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 7,
    "screen": ROOM,
    "x": 16,
    "y": 141,
    "selected": 4,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


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


def test_walks_toward_stand() -> None:
    ram = _ram(x=16, y=141, selected=4)
    _plant(ram, 1, DIGDOGGER_TYPE)
    controller = _bound(ram)
    action = controller.step(read_snapshot(ram))
    assert _buttons(action) == ["RIGHT"]
    ram[ADDR_LINK_Y] = 125
    action = controller.step(read_snapshot(ram))
    assert _buttons(action) == ["DOWN"]


def test_boundary_coords_bind_leftover_relative_stand() -> None:
    """West / east / off-row leftovers walk to the door-row stand; dest is RAM."""
    west = _ram(x=16, y=141, selected=5)
    _plant(west, 1, DIGDOGGER_TYPE)
    ctl = _bound(west)
    assert _buttons(ctl.step(read_snapshot(west))) == ["RIGHT"]
    assert ctl.stand == (120, 141)
    assert ctl.phase is DigdoggerPhase.WALK

    east = _ram(x=200, y=141, selected=5)
    _plant(east, 1, DIGDOGGER_TYPE)
    ctl = _bound(east)
    assert _buttons(ctl.step(read_snapshot(east))) == ["LEFT"]
    assert ctl.stand == (120, 141)

    off = _ram(x=120, y=125, selected=5)
    _plant(off, 1, DIGDOGGER_TYPE)
    ctl = _bound(off)
    assert _buttons(ctl.step(read_snapshot(off))) == ["DOWN"]
    assert ctl.stand == (120, 141)
    assert "stand_settle" not in ctl.notes


def test_at_stand_skips_idle_settle_and_opens_pause() -> None:
    ram = _ram(x=120, y=141, selected=4)
    _plant(ram, 1, DIGDOGGER_TYPE)
    before = int(ram[ADDR_SELECTED_ITEM])
    controller = _bound(ram)
    action = controller.step(read_snapshot(ram))
    assert _buttons(action) == ["START"]
    assert action.reason == "pause_open"
    assert controller.phase is DigdoggerPhase.SELECT
    assert "stand_settle" not in {action.reason, *controller.notes}
    assert int(ram[ADDR_SELECTED_ITEM]) == before
    assert controller.report()["writes"] == 0
    assert controller.report()["normal_pause_input"] is True


def test_blows_until_ram_shrinks_not_twelve_b() -> None:
    ram = _ram(x=120, y=141, selected=5)
    _plant(ram, 1, DIGDOGGER_TYPE)
    controller = _bound(ram)
    seen_b = 0
    for _ in range(20):
        before = ram.copy()
        action = controller.step(read_snapshot(ram))
        assert np.array_equal(ram, before)
        if _buttons(action) == ["B"]:
            seen_b += 1
        else:
            break
    assert seen_b >= 1
    assert controller.phase is DigdoggerPhase.BLOW
    _plant(ram, 1, DIGDOGGER_SHRUNK_TYPE, hp=128, x=132, y=141)
    action = controller.step(read_snapshot(ram))
    assert controller.shrunk
    assert controller.phase is DigdoggerPhase.SWORD
    assert "B" not in _buttons(action)
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
    assert "link_death" in controller.notes


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


def test_empty_slot_frame_does_not_exit_before_shrunk_minis() -> None:
    """Whistle despawns 0x38 before 0x18 spawns: stay in BLOW, transition to SWORD on 0x18."""
    ram = _ram(x=120, y=141, selected=5)
    _plant(ram, 1, DIGDOGGER_TYPE, hp=240, x=132, y=141)
    controller = _bound(ram)

    # Start BLOW (B button pulsed on rising edge)
    action = controller.step(read_snapshot(ram))
    assert controller.phase is DigdoggerPhase.BLOW
    assert _buttons(action) == ["B"]

    # Clear slot for one frame: large 0x38 despawned, shrunk 0x18 not yet spawned
    ram[ADDR_OBJ_TYPE + 1] = 0
    ram[ADDR_OBJ_HP + 1] = 0
    action = controller.step(read_snapshot(ram))
    assert controller.phase is not DigdoggerPhase.EXIT
    assert controller.phase is DigdoggerPhase.BLOW
    assert not controller.killed
    assert not controller.shrunk
    assert action.reason == "whistle_wait"

    # Mini Digdogger spawns (0x18, hp > 0): controller transitions to SWORD
    _plant(ram, 1, DIGDOGGER_SHRUNK_TYPE, hp=128, x=132, y=141)
    action = controller.step(read_snapshot(ram))
    assert controller.phase is DigdoggerPhase.SWORD
    assert controller.shrunk
    assert not controller.killed


def test_blow_wait_frames_timeout_exits_if_large_gone() -> None:
    """If large Digdogger is gone and no minis spawn for BLOW_WAIT_FRAMES, exit north."""
    ram = _ram(x=120, y=141, selected=5)
    _plant(ram, 1, DIGDOGGER_TYPE, hp=240, x=132, y=141)
    controller = _bound(ram)
    controller.step(read_snapshot(ram))
    assert controller.phase is DigdoggerPhase.BLOW

    # Clear slot
    ram[ADDR_OBJ_TYPE + 1] = 0
    ram[ADDR_OBJ_HP + 1] = 0

    # Advance until just before BLOW_WAIT_FRAMES
    for _ in range(239):
        controller.step(read_snapshot(ram))
        assert controller.phase is DigdoggerPhase.BLOW
        assert not controller.killed

    # At BLOW_WAIT_FRAMES, large_gone_after_blow triggers EXIT
    action = controller.step(read_snapshot(ram))
    assert controller.phase is DigdoggerPhase.EXIT
    assert controller.killed
    assert controller.shrunk


def test_blow_wait_frames_timeout_retries_if_large_still_present() -> None:
    """If large Digdogger remains present after BLOW_WAIT_FRAMES, retry whistle."""
    ram = _ram(x=120, y=141, selected=5)
    _plant(ram, 1, DIGDOGGER_TYPE, hp=240, x=132, y=141)
    controller = _bound(ram)
    controller.step(read_snapshot(ram))
    assert controller.phase is DigdoggerPhase.BLOW
    assert controller.blow_attempts == 1

    # Large 0x38 remains present throughout BLOW_WAIT_FRAMES
    for _ in range(239):
        action = controller.step(read_snapshot(ram))
        assert controller.phase is DigdoggerPhase.BLOW
        assert not controller.killed
        assert action.reason == "whistle_wait"

    # At BLOW_WAIT_FRAMES with large still present, retry blow
    action = controller.step(read_snapshot(ram))
    assert controller.phase is DigdoggerPhase.BLOW
    assert controller.blow_attempts == 2
    assert _buttons(action) == ["B"]
    assert "whistle_retry" in controller.notes
