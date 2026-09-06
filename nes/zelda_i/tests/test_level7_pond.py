"""L7 Demon pond whistle-drain + 0x79 entry (`level7.pond`).

Fake RAM. ``OW_L7Pond`` is whistle=0 (optional ``@pytest.mark.rom`` fail-closed).
No 0x42+whistle=1 pin exists; live 2/2 drain success is skipped.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from zelda_i.dungeon.pause_select import CLOSE_SETTLE_FRAMES
from zelda_i.level7.pond import (
    BLOW_STAND,
    BLOW_PRESSES,
    DEST,
    DEST_XY,
    POND_SCREEN,
    POND_STAIR_TILE,
    SOUTH_SHORE,
    STAIR_CANDIDATES,
    STAIRS_XY,
    STUCK_FRAMES,
    WHISTLE_B_SLOT,
    Level7PondDrainController,
    PondPhase,
    make_pond_drain_controller,
)
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_POND_PIN = "OW_L7Pond"
_POND_NATURAL_PIN = "OW_L7PondNatural"

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 0,
    "screen": POND_SCREEN,
    "x": SOUTH_SHORE[0],
    "y": SOUTH_SHORE[1],
    "whistle": 1,
    "selected": WHISTLE_B_SLOT,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _env(ram: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(get_ram=lambda: ram)


def _bound(**fields: int) -> tuple[Level7PondDrainController, np.ndarray]:
    ram = _ram(**fields)
    ctl = make_pond_drain_controller()
    ctl.bind_env(_env(ram))
    return ctl, ram


def _buttons(action) -> list[str]:
    return sorted(
        name
        for name, idx in NES_BUTTON_NAME_TO_INDEX.items()
        if idx is not None and int(action.action[idx])
    )


def _step(ctl: Level7PondDrainController, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "pond controller must not write RAM"
    return act


def test_public_constants() -> None:
    assert WHISTLE_B_SLOT == 5
    assert POND_SCREEN == 0x42
    assert DEST == 0x79
    assert DEST_XY == (120, 205)
    assert BLOW_STAND == (128, 189)
    assert STAIRS_XY == (96, 144)
    assert POND_STAIR_TILE == 0x70
    assert SOUTH_SHORE == (128, 221)
    assert STAIR_CANDIDATES[0] == STAIRS_XY


def test_whistle_zero_fails_immediately() -> None:
    ctl, ram = _bound(whistle=0, selected=1)
    act = _step(ctl, ram)
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "pond_requires_whistle"
    assert _buttons(act) == []
    assert int(ram[ADDR_WHISTLE]) == 0
    assert int(ram[ADDR_SELECTED_ITEM]) == 1


def test_env_not_bound_fails() -> None:
    ctl = make_pond_drain_controller()
    act = ctl.step(read_snapshot(_ram(whistle=1)))
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "pond_drain_env_not_bound"


def test_pause_select_presses_start_and_does_not_poke() -> None:
    ctl, ram = _bound(whistle=1, selected=1, x=100, y=205)
    before_sel = int(ram[ADDR_SELECTED_ITEM])
    before_wh = int(ram[ADDR_WHISTLE])
    act = _step(ctl, ram)
    assert _buttons(act) == ["START"]
    assert act.reason == "pause_open"
    assert int(ram[ADDR_SELECTED_ITEM]) == before_sel
    assert int(ram[ADDR_WHISTLE]) == before_wh
    assert ctl.report()["writes"] == 0
    assert ctl.report()["normal_pause_input"] is True
    assert ctl.phase is PondPhase.SELECT
    assert "RIGHT" not in _buttons(act)
    assert "UP" not in _buttons(act)


def test_pause_select_cycles_right_then_closes() -> None:
    ctl, ram = _bound(whistle=1, selected=1, x=BLOW_STAND[0], y=BLOW_STAND[1])
    assert _buttons(_step(ctl, ram)) == ["START"]
    for _ in range(20):
        act = _step(ctl, ram)
        assert _buttons(act) == []
    act = _step(ctl, ram)
    assert _buttons(act) == ["RIGHT"]
    assert int(ram[ADDR_SELECTED_ITEM]) == 1
    ram[ADDR_SELECTED_ITEM] = WHISTLE_B_SLOT
    for _ in range(8):
        _step(ctl, ram)
    act = _step(ctl, ram)
    assert _buttons(act) == []
    act = _step(ctl, ram)
    assert _buttons(act) == ["START"]
    assert act.reason == "pause_close"
    for _ in range(CLOSE_SETTLE_FRAMES - 1):
        _step(ctl, ram)
    act = _step(ctl, ram)
    assert ctl.phase is PondPhase.STAND_SETTLE
    assert _buttons(act) == []
    assert int(ram[ADDR_SELECTED_ITEM]) == WHISTLE_B_SLOT
    assert int(ram[ADDR_WHISTLE]) == 1


def test_already_selected_slot_5_skips_the_menu() -> None:
    ctl, ram = _bound(whistle=1, selected=WHISTLE_B_SLOT)
    act = _step(ctl, ram)
    assert _buttons(act) == ["UP"]
    assert act.reason == "stand_y"
    assert ctl.phase is PondPhase.WALK
    assert "START" not in _buttons(act)
    assert "recorder_already_selected" in ctl.notes


def test_walks_toward_blow_stand() -> None:
    ctl, ram = _bound(
        whistle=1, selected=WHISTLE_B_SLOT, x=100, y=BLOW_STAND[1]
    )
    assert _buttons(_step(ctl, ram)) == ["RIGHT"]
    ram[ADDR_LINK_X] = 140
    assert _buttons(_step(ctl, ram)) == ["LEFT"]
    ram[ADDR_LINK_X] = BLOW_STAND[0]
    ram[ADDR_LINK_Y] = SOUTH_SHORE[1]
    assert _buttons(_step(ctl, ram)) == ["UP"]


def test_twelve_b_after_selected_equals_5() -> None:
    ctl, ram = _bound(
        whistle=1,
        selected=WHISTLE_B_SLOT,
        x=BLOW_STAND[0],
        y=BLOW_STAND[1],
    )
    seen_b = 0
    for _ in range(40):
        act = _step(ctl, ram)
        if _buttons(act) == ["B"]:
            seen_b += 1
        elif seen_b:
            break
    assert seen_b == BLOW_PRESSES
    assert int(ram[ADDR_SELECTED_ITEM]) == WHISTLE_B_SLOT
    assert int(ram[ADDR_WHISTLE]) == 1
    assert ctl.phase is PondPhase.BLOW_WAIT


def test_success_rising_edge_into_l7_0x79() -> None:
    ctl, ram = _bound(whistle=1, selected=WHISTLE_B_SLOT)
    _step(ctl, ram)
    assert not ctl.success
    assert ctl.screen_in == POND_SCREEN
    assert ctl.level_in == 0
    ram[ADDR_LEVEL] = 7
    ram[ADDR_SCREEN] = DEST
    ram[ADDR_LINK_X] = DEST_XY[0]
    ram[ADDR_LINK_Y] = DEST_XY[1]
    _step(ctl, ram)
    assert ctl.success
    assert not ctl.failed
    assert "entered_0x79" in ctl.notes
    leftover = ctl.report()["leftover"]
    assert leftover is not None
    assert leftover["screen"] == DEST
    assert leftover["mode"] == PLAY_MODE
    assert leftover["level"] == 7


def test_already_in_0x79_without_drain_never_greens() -> None:
    ctl, ram = _bound(
        whistle=1,
        selected=WHISTLE_B_SLOT,
        level=7,
        screen=DEST,
        x=DEST_XY[0],
        y=DEST_XY[1],
    )
    act = _step(ctl, ram)
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "already_in_entry_without_drain"
    assert ctl.screen_in == DEST
    assert ctl.level_in == 7


def test_leaving_0x42_for_a_non_entry_screen_fails() -> None:
    ctl, ram = _bound(whistle=1, selected=WHISTLE_B_SLOT)
    _step(ctl, ram)
    ram[ADDR_SCREEN] = 0x52
    act = _step(ctl, ram)
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "left_ow_0x52"


def test_death_fails() -> None:
    ctl, ram = _bound(whistle=1, selected=WHISTLE_B_SLOT, mode=17)
    act = _step(ctl, ram)
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "death"


def test_occupancy_miss_on_stairs_halts() -> None:
    ctl, ram = _bound(
        whistle=1,
        selected=WHISTLE_B_SLOT,
        x=BLOW_STAND[0],
        y=BLOW_STAND[1],
    )
    ctl.phase = PondPhase.STAIRS
    ctl.blew = True
    act = None
    for _ in range(STUCK_FRAMES + 4):
        act = _step(ctl, ram)
        if ctl.failed:
            break
    assert ctl.failed
    assert not ctl.success
    assert act is not None
    assert act.reason == "occupancy_miss"


def test_first_stairs_cell_miss_seeks_probe_candidate() -> None:
    from zelda_i.level7.pond import STAIR_DWELL_FRAMES

    ctl, ram = _bound(
        whistle=1,
        selected=WHISTLE_B_SLOT,
        x=STAIRS_XY[0],
        y=STAIRS_XY[1],
    )
    ctl.phase = PondPhase.STAIRS
    ctl.blew = True
    for _ in range(STAIR_DWELL_FRAMES):
        act = _step(ctl, ram)
        assert not ctl.failed
        assert _buttons(act) == ["UP"]
    act = _step(ctl, ram)
    assert not ctl.failed
    assert ctl.stair_index == 1
    assert STAIR_CANDIDATES[1] == (104, 144)
    assert act.reason == "stairs_seek_x"
    assert _buttons(act) == ["RIGHT"]


def test_factory_never_shares_instances() -> None:
    assert make_pond_drain_controller() is not make_pond_drain_controller()


def test_report_writes_zero_route_eligible_false() -> None:
    report = make_pond_drain_controller().report()
    assert report["route_eligible"] is False
    assert report["writes"] == 0
    assert report["normal_pause_input"] is True
    assert report["evidence"] == "fixture-live"
    assert report["spec_id"] == "level7_pond_drain_entry"
    assert report["dest"] == "0x79"
    assert report["pond_screen"] == "0x42"
    assert report["stairs_xy"] == [96, 144]
    assert report["blow_stand"] == [128, 189]


def test_no_occupancy_walker_and_no_whistle_poke() -> None:
    import zelda_i.level7.pond as mod

    import inspect

    drain_src = inspect.getsource(Level7PondDrainController)
    assert "OccupancyWalker" not in drain_src
    assert "mem_write" not in drain_src
    text = open(mod.__file__, encoding="utf-8").read()
    assert "poke_whistle" not in text
    tree = ast.parse(inspect.getsource(Level7PondDrainController))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                joined = ast.unparse(target)
                assert "ADDR_SELECTED_ITEM" not in joined
                assert "ADDR_WHISTLE" not in joined


def _pond_pin_ready() -> bool:
    from retro_harness.env import state_path
    from zelda_i.paths import GAME, GAME_DIR, SHARED_ROM_ZIP

    return SHARED_ROM_ZIP.is_file() and state_path(
        GAME_DIR, GAME, _POND_PIN
    ).is_file()


@pytest.mark.rom
@pytest.mark.skipif(not _pond_pin_ready(), reason="Zelda I ROM or OW_L7Pond pin missing")
def test_live_ow_l7pond_fails_closed_without_whistle() -> None:
    """Pin OW_L7Pond is whistle=0; drain must fail closed (no poke)."""
    from retro_harness.env import make_env, reset_obs
    from retro_harness.nes import nes_idle_action
    from retro_harness.segment_runner import configure_headless
    from zelda_i.paths import GAME, GAME_DIR
    from zelda_i.ram import read_u8

    configure_headless()
    env = make_env(GAME, _POND_PIN, GAME_DIR, render_mode="rgb_array")
    ctl = make_pond_drain_controller()
    ctl.bind_env(env)
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        whistle0 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        selected0 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
        assert whistle0 == 0
        snap = read_snapshot(env.get_ram())
        action = ctl.step(snap)
        env.step(action.action)
        selected1 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
        whistle1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    finally:
        env.close()
    assert ctl.failed
    assert not ctl.success
    assert "pond_requires_whistle" in ctl.notes
    assert whistle1 == 0
    assert selected1 == selected0
    assert ctl.report()["writes"] == 0
    assert ctl.report()["route_eligible"] is False


def _pond_natural_pin_ready() -> bool:
    from retro_harness.env import state_path
    from zelda_i.paths import GAME, GAME_DIR, SHARED_ROM_ZIP

    return SHARED_ROM_ZIP.is_file() and state_path(
        GAME_DIR, GAME, _POND_NATURAL_PIN
    ).is_file()


@pytest.mark.rom
@pytest.mark.skipif(
    not _pond_natural_pin_ready(),
    reason="Zelda I ROM or OW_L7PondNatural pin missing",
)
def test_live_drain_2of2_from_naturally_arrived_pond_pin() -> None:
    """OW_L7PondNatural: Recorder naturally owned, arrived on 0x42 with zero
    pokes (scratch/pond/capture_pond_natural_pin.py). Drain must land byte-
    identically in L7 play 0x79 twice in a row, writes=0 both trials.
    """
    from retro_harness.env import make_env, reset_obs
    from retro_harness.nes import nes_idle_action
    from retro_harness.segment_runner import configure_headless
    from zelda_i.paths import GAME, GAME_DIR
    from zelda_i.runner import make_assist

    configure_headless()
    leftovers: list[dict] = []
    for trial in range(2):
        assist = make_assist(True)
        env = make_env(GAME, _POND_NATURAL_PIN, GAME_DIR, render_mode="rgb_array")
        ctl = make_pond_drain_controller()
        ctl.bind_env(env)
        try:
            reset_obs(env)
            for _ in range(2):
                env.step(nes_idle_action())
            whistle0 = int(env.get_ram()[ADDR_WHISTLE])
            assert whistle0 >= 1
            f = 0
            while f < ctl.max_frames + 10:
                snap = read_snapshot(env.get_ram())
                action = ctl.step(snap)
                env.step(action.action)
                if assist is not None:
                    assist.apply_env(env, frame=f)
                f += 1
                if ctl.success or ctl.failed:
                    break
            end = read_snapshot(env.get_ram())
            report = ctl.report()
        finally:
            env.close()
        assert ctl.success, f"trial {trial} {ctl.report()}"
        assert not ctl.failed
        assert end.level == 7
        assert end.mode == PLAY_MODE
        assert not end.transitioning
        assert int(end.screen) == DEST
        assert (int(end.link_x), int(end.link_y)) == DEST_XY
        assert report["route_eligible"] is False
        assert report["writes"] == 0
        assert assist.telemetry.progression_writes == 0
        assert assist.telemetry.capacity_writes == 0
        leftovers.append(report["leftover"])
    assert leftovers[0] == leftovers[1], "trials must be byte-identical"
