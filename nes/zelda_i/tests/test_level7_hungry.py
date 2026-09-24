"""L7 Hungry Goriya feed + north leave (`level7.hungry`).

Fake RAM. Live pin is ``Level7Interior28ReconFixture`` (optional ``@pytest.mark.rom``).
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from zelda_i.dungeon.pause_select import CLOSE_SETTLE_FRAMES
from zelda_i.level7.hungry import (
    DEST,
    DOOR_X,
    FEED_Y,
    FOOD_B_SLOT,
    HungryPhase,
    Level7HungryGoriyaController,
    ROOM,
    make_level7_hungry_goriya_controller,
)
from zelda_i.ram import (
    ADDR_FOOD,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_FIXTURE = "Level7Interior28ReconFixture"

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 7,
    "screen": ROOM,
    "x": 120,
    "y": 205,
    "food": 1,
    "selected": FOOD_B_SLOT,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _env(ram: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(get_ram=lambda: ram)


def _bound(**fields: int) -> tuple[Level7HungryGoriyaController, np.ndarray]:
    ram = _ram(**fields)
    ctl = make_level7_hungry_goriya_controller()
    ctl.bind_env(_env(ram))
    return ctl, ram


def _buttons(action) -> list[str]:
    return sorted(
        name
        for name, idx in NES_BUTTON_NAME_TO_INDEX.items()
        if idx is not None and int(action.action[idx])
    )


def _step(ctl: Level7HungryGoriyaController, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "hungry controller must not write RAM"
    return act


def test_food_less_than_one_fails_immediately() -> None:
    ctl, ram = _bound(food=0, selected=1)
    act = _step(ctl, ram)
    assert ctl.failed
    assert not ctl.success
    assert not ctl.food_consumed
    assert act.reason == "hungry_goriya_requires_food"
    assert _buttons(act) == []
    leftover = ctl.report()["leftover"]
    assert leftover is not None
    assert leftover["reason"] == "hungry_goriya_requires_food"
    assert leftover["food"] == 0
    assert leftover["screen"] == ROOM
    assert int(ram[ADDR_FOOD]) == 0


def test_clean_survival_false_never_writes_addr_food() -> None:
    """allow_pokes=False / survival=False must not poke ADDR_FOOD."""
    from types import SimpleNamespace

    from zelda_i.level7.entry import NaturalBaitPurchaseController
    from zelda_i.level7.hops import level7_entry_chapter_stages
    from zelda_i.ram import ADDR_FOOD as FOOD_ADDR

    ram = _ram(food=0)
    ram[FOOD_ADDR] = 0
    calls: list[tuple[int, int]] = []

    class _Mem:
        def assign(self, addr: int, _fmt: str, val: int) -> None:
            calls.append((int(addr), int(val)))
            ram[int(addr)] = int(val) & 0xFF

    env = SimpleNamespace(
        get_ram=lambda: ram,
        unwrapped=SimpleNamespace(data=SimpleNamespace(memory=_Mem())),
    )
    stages = level7_entry_chapter_stages(survival=False)
    bait = stages[3][1]
    assert isinstance(bait, NaturalBaitPurchaseController)
    bait.bind_env(env)
    bait.step(read_snapshot(ram))
    assert calls == []
    assert ram[FOOD_ADDR] == 0
    assert bait.report()["writes"] == 0


def test_pause_select_presses_start_and_does_not_poke() -> None:
    ctl, ram = _bound(food=1, selected=1, x=100, y=205)
    before = int(ram[ADDR_SELECTED_ITEM])
    act = _step(ctl, ram)
    assert _buttons(act) == ["START"]
    assert act.reason == "pause_open"
    assert int(ram[ADDR_SELECTED_ITEM]) == before
    assert ctl.report()["writes"] == 0
    assert ctl.report()["normal_pause_input"] is True
    assert ctl.phase is HungryPhase.SELECT
    # Off-center spawn would walk RIGHT if we skipped pause; START wins.
    assert "RIGHT" not in _buttons(act)


def test_pause_select_cycles_right_then_closes() -> None:
    ctl, ram = _bound(food=1, selected=1, x=120, y=205)
    assert _buttons(_step(ctl, ram)) == ["START"]
    for _ in range(20):
        act = _step(ctl, ram)
        assert _buttons(act) == []
    act = _step(ctl, ram)
    assert _buttons(act) == ["RIGHT"]
    assert int(ram[ADDR_SELECTED_ITEM]) == 1
    ram[ADDR_SELECTED_ITEM] = FOOD_B_SLOT
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
    assert ctl.phase is HungryPhase.APPROACH
    assert _buttons(act) == ["UP"]
    assert int(ram[ADDR_SELECTED_ITEM]) == FOOD_B_SLOT
    assert int(ram[ADDR_FOOD]) == 1


def test_walks_toward_x_120() -> None:
    ctl, ram = _bound(food=1, selected=FOOD_B_SLOT, x=100, y=205)
    assert _buttons(_step(ctl, ram)) == ["RIGHT"]
    ram[ADDR_LINK_X] = 140
    assert _buttons(_step(ctl, ram)) == ["LEFT"]
    ram[ADDR_LINK_X] = DOOR_X
    assert _buttons(_step(ctl, ram)) == ["UP"]


def test_b_taps_after_selected_equals_6() -> None:
    ctl, ram = _bound(food=1, selected=FOOD_B_SLOT, x=DOOR_X, y=FEED_Y)
    act = _step(ctl, ram)
    assert ctl.phase is HungryPhase.APPROACH
    assert _buttons(act) == ["B"]
    assert act.reason == "hungry_feed"
    # Still unselected bait: pause START, never B.
    other, ram2 = _bound(food=1, selected=1, x=DOOR_X, y=FEED_Y)
    assert _buttons(_step(other, ram2)) == ["START"]


def test_success_on_0x18_after_food_falls_to_zero() -> None:
    ctl, ram = _bound(food=1, selected=FOOD_B_SLOT, x=DOOR_X, y=FEED_Y)
    _step(ctl, ram)
    assert not ctl.success
    assert not ctl.food_consumed
    ram[ADDR_FOOD] = 0
    ram[ADDR_SCREEN] = DEST
    ram[ADDR_LINK_Y] = 189
    _step(ctl, ram)
    assert ctl.success
    assert not ctl.failed
    assert ctl.food_consumed
    assert "food_consumed_naturally" in ctl.notes
    assert "fed_and_map_0x18" in ctl.notes


def test_already_food_zero_then_overworld_or_other_room_never_greens() -> None:
    ctl, ram = _bound(food=0, selected=FOOD_B_SLOT, x=DOOR_X, y=FEED_Y)
    act = _step(ctl, ram)
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "hungry_goriya_requires_food"
    ram[ADDR_LEVEL] = 0
    ram[ADDR_SCREEN] = 0x42
    _step(ctl, ram)
    assert ctl.failed
    assert not ctl.success
    assert not ctl.food_consumed

    other, ram2 = _bound(food=0, selected=FOOD_B_SLOT, screen=DEST)
    act = _step(other, ram2)
    assert act.reason == "hungry_goriya_requires_food"
    ram2[ADDR_SCREEN] = 0x38
    _step(other, ram2)
    assert other.failed
    assert not other.success
    assert "hungry_goriya_requires_food" in other.notes


def test_leaving_0x28_for_a_non_map_room_fails() -> None:
    ctl, ram = _bound(food=1, selected=FOOD_B_SLOT, x=DOOR_X, y=205)
    _step(ctl, ram)
    ram[ADDR_SCREEN] = 0x38
    act = _step(ctl, ram)
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "left_room_L7_0x38"


def test_death_fails() -> None:
    ctl, ram = _bound(food=1, selected=FOOD_B_SLOT, mode=17)
    act = _step(ctl, ram)
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "death"


def test_dest_without_watched_food_drop_fails() -> None:
    ctl, ram = _bound(food=1, selected=FOOD_B_SLOT, x=DOOR_X, y=205)
    _step(ctl, ram)
    ram[ADDR_SCREEN] = DEST
    act = _step(ctl, ram)
    assert ctl.failed
    assert not ctl.success
    assert not ctl.food_consumed
    assert act.reason == "dest_without_food_consume"


def test_factory_never_shares_instances() -> None:
    assert make_level7_hungry_goriya_controller() is not (
        make_level7_hungry_goriya_controller()
    )


def test_no_occupancy_walker_and_no_selected_item_poke() -> None:
    import zelda_i.level7.hungry as mod

    assert not hasattr(mod, "OccupancyWalker")
    source = mod.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    tree = ast.parse(text)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                joined = ast.unparse(target)
                assert "ADDR_SELECTED_ITEM" not in joined
                assert "ADDR_FOOD" not in joined
    assert "OccupancyWalker" not in imported
    assert "zelda_i.walk.physics" not in imported
    assert "mem_write" not in imported
    assert "poke_food" not in text


def _live_ready() -> bool:
    from zelda_i.paths import GAME, GAME_DIR, SHARED_ROM_ZIP
    from retro_harness.env import state_path

    return SHARED_ROM_ZIP.is_file() and state_path(
        GAME_DIR, GAME, _FIXTURE
    ).is_file()


@pytest.mark.rom
@pytest.mark.skipif(not _live_ready(), reason="Zelda I ROM or 0x28 fixture missing")
def test_live_feed_from_interior28_recon_fixture() -> None:
    """Pin Level7Interior28ReconFixture; Food 1→0 natural, dest play 0x18."""
    from retro_harness.env import make_env, reset_obs
    from retro_harness.nes import nes_idle_action
    from retro_harness.segment_runner import configure_headless
    from zelda_i.paths import GAME, GAME_DIR
    from zelda_i.ram import read_u8
    from zelda_i.runner import make_assist

    configure_headless()
    for trial in range(2):
        assist = make_assist(True)
        env = make_env(GAME, _FIXTURE, GAME_DIR, render_mode="rgb_array")
        ctl = make_level7_hungry_goriya_controller()
        ctl.bind_env(env)
        try:
            reset_obs(env)
            for _ in range(2):
                env.step(nes_idle_action())
            food0 = int(read_u8(env.get_ram(), ADDR_FOOD))
            selected0 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
            assert food0 >= 1
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
            food1 = int(read_u8(env.get_ram(), ADDR_FOOD))
            selected1 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
            report = ctl.report()
        finally:
            env.close()
        assert ctl.success, f"trial {trial} {ctl.report()}"
        assert not ctl.failed
        assert ctl.food_consumed
        assert food1 == 0
        assert food0 >= 1
        assert end.level == 7
        assert end.mode == PLAY_MODE
        assert not end.transitioning
        assert int(end.screen) == DEST
        assert report["route_eligible"] is False
        assert report["writes"] == 0
        assert assist.telemetry.progression_writes == 0
        assert selected0 != FOOD_B_SLOT or selected1 in {FOOD_B_SLOT, 5}
