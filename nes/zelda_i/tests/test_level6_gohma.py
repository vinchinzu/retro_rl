"""Unit tests for Level 6 Gohma 0x1C (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.dungeon.ids import GOHMA_BLUE_OBJECT_TYPE, GOHMA_OBJECT_TYPE
from zelda_i.level6.gohma import (
    level6_gohma_success,
    make_gohma_controller,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 6)
    ram[ADDR_SCREEN] = fields.get("screen", 0x1C)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 205)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 3)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_ROD] = fields.get("rod", 1)
    ram[ADDR_BOW] = fields.get("bow", 1)
    ram[ADDR_ARROWS] = fields.get("arrows", 1)
    return ram


def _plant_gohma(
    ram: np.ndarray,
    *,
    x: int = 120,
    y: int = 109,
    hp: int = 16,
    type_id: int = GOHMA_OBJECT_TYPE,
) -> None:
    ram[ADDR_OBJ_TYPE + 1] = type_id
    ram[ADDR_OBJ_HP + 1] = hp
    ram[ADDR_LINK_X + 1] = x
    ram[ADDR_LINK_Y + 1] = y


class _AssignMem:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int]] = []

    def assign(self, addr: int, fmt: str, val: int) -> None:
        self.calls.append((addr, fmt, val))


def _env_with_mem(mem: object) -> SimpleNamespace:
    data = SimpleNamespace(memory=mem)
    return SimpleNamespace(unwrapped=SimpleNamespace(data=data))


def test_unarmed_no_bow_fails() -> None:
    ram = _ram(bow=0, arrows=0)
    _plant_gohma(ram)
    ctl = make_gohma_controller()
    ctl.step(read_snapshot(ram))
    assert ctl.failed


def test_poke_writes_arrows_and_b_not_bow() -> None:
    from zelda_i.ram import ADDR_ARROWS as ARROWS
    from zelda_i.ram import ADDR_SELECTED_ITEM

    ram = _ram(bow=1, arrows=0)
    _plant_gohma(ram)
    mem = _AssignMem()
    ctl = make_gohma_controller()
    ctl.bind_env(_env_with_mem(mem))
    ctl.step(read_snapshot(ram))
    assert not ctl.failed
    addrs = [addr for addr, _fmt, _val in mem.calls]
    assert ARROWS in addrs
    assert ADDR_SELECTED_ITEM in addrs
    from zelda_i.ram import ADDR_BOW as BOW

    assert BOW not in addrs
    assert ctl.inventory_assist is not None
    assert ctl.inventory_assist["progression_writes"] == 0
    assert ctl.inventory_assist["bow_writes"] == 0


def test_gohma_success_needs_body_gone_and_arrows() -> None:
    ram = _ram(x=120, y=205, bow=1, arrows=1)
    _plant_gohma(ram)
    assert not level6_gohma_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = 0
    ram[ADDR_OBJ_HP + 1] = 0
    assert level6_gohma_success(read_snapshot(ram))
    ram[ADDR_ARROWS] = 0
    assert not level6_gohma_success(read_snapshot(ram))


def test_south_mouth_column_shoots_same_frame_as_poke() -> None:
    ram = _ram(x=120, y=205, bow=1, arrows=1)
    _plant_gohma(ram, x=128, type_id=GOHMA_BLUE_OBJECT_TYPE)
    ctl = make_gohma_controller()
    ctl.bind_env(_env_with_mem(_AssignMem()))
    action = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert action.reason == "arrow_shot"
    assert action.reason not in ("inland_up", "occupancy_stand", "inland_path")
    assert list(action.action) == list(nes_action("UP", "B"))
    assert ctl.inventory_assist is not None
    assert ctl.arrow_pulses == 1


def test_knockback_y189_still_shoots_in_column() -> None:
    ram = _ram(x=120, y=189, bow=1, arrows=1)
    _plant_gohma(ram, x=128, type_id=GOHMA_BLUE_OBJECT_TYPE)
    ctl = make_gohma_controller()
    action = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert action.reason == "arrow_shot"
    assert action.reason not in ("inland_up", "occupancy_stand")
    assert list(action.action) == list(nes_action("UP", "B"))


def test_spawn_gx128_dx8_shoots() -> None:
    ram = _ram(x=120, y=205, bow=1, arrows=1)
    _plant_gohma(ram, x=128, type_id=GOHMA_BLUE_OBJECT_TYPE)
    ctl = make_gohma_controller()
    action = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert action.reason == "arrow_shot"
    assert list(action.action) == list(nes_action("UP", "B"))


def test_v3_gx141_does_not_shoot_or_align() -> None:
    from retro_harness.nes import nes_idle_action

    ram = _ram(x=120, y=169, bow=1, arrows=1)
    _plant_gohma(ram, x=141, type_id=GOHMA_BLUE_OBJECT_TYPE)
    ctl = make_gohma_controller()
    action = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert action.reason == "column_wait"
    assert action.reason not in ("align_x", "arrow_shot", "inland_up")
    assert list(action.action) == list(nes_idle_action())
    assert ctl.arrow_pulses == 0


def test_north_shutter_one_pulse_does_not_hold_up() -> None:
    from retro_harness.nes import nes_idle_action

    ram = _ram(x=115, y=93, bow=1, arrows=1)
    _plant_gohma(ram, x=120, type_id=GOHMA_BLUE_OBJECT_TYPE)
    ctl = make_gohma_controller()
    shot = ctl.step(read_snapshot(ram))
    assert shot.reason == "arrow_shot"
    wait = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert wait.reason == "shot_wait"
    assert wait.reason not in ("inland_down", "inland_up")
    assert list(wait.action) == list(nes_idle_action())


def test_shot_cooldown_idles_instead_of_walking_north() -> None:
    from retro_harness.nes import nes_idle_action

    ram = _ram(x=120, y=165, bow=1, arrows=1)
    _plant_gohma(ram, x=120, type_id=GOHMA_BLUE_OBJECT_TYPE)
    ctl = make_gohma_controller()
    ctl.bind_env(_env_with_mem(_AssignMem()))
    shot = ctl.step(read_snapshot(ram))
    assert shot.reason == "arrow_shot"
    wait = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert wait.reason == "shot_wait"
    assert list(wait.action) == list(nes_idle_action())


def test_one_pulse_idles_after_cooldown_no_spray() -> None:
    from retro_harness.nes import nes_idle_action

    ram = _ram(x=120, y=165, bow=1, arrows=1)
    _plant_gohma(ram, x=120, type_id=GOHMA_BLUE_OBJECT_TYPE)
    ctl = make_gohma_controller()
    ctl.bind_env(_env_with_mem(_AssignMem()))
    shot = ctl.step(read_snapshot(ram))
    assert shot.reason == "arrow_shot"
    ctl.cooldown = 0
    idle = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert idle.reason == "spawn_wait"
    assert idle.reason != "arrow_shot"
    assert list(idle.action) == list(nes_idle_action())
    assert ctl.arrow_pulses == 1
