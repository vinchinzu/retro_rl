"""Unit tests for Level 6 Gohma 0x1C reactive kill (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.ids import GOHMA_OBJECT_TYPE
from zelda_i.level6.gohma import (
    EYE_ADDR,
    EYE_SHUT,
    FACE_NORTH,
    STAND_Y,
    level6_gohma_success,
    make_clean_gohma_controller,
    make_gohma_controller,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROD,
    ADDR_RUPEES,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x1000, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 6)
    ram[ADDR_SCREEN] = fields.get("screen", 0x1C)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 205)
    ram[ADDR_LINK_FACING] = fields.get("facing", FACE_NORTH)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 2)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_ROD] = fields.get("rod", 1)
    ram[ADDR_BOW] = fields.get("bow", 1)
    ram[ADDR_ARROWS] = fields.get("arrows", 1)
    ram[ADDR_RUPEES] = fields.get("rupees", 43)
    ram[EYE_ADDR] = fields.get("eye", EYE_SHUT)
    return ram


def _plant_gohma(
    ram: np.ndarray,
    *,
    x: int = 128,
    y: int = 112,
    hp: int = 32,
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


def _env(ram: np.ndarray, mem: object | None = None) -> SimpleNamespace:
    mem = mem if mem is not None else _AssignMem()
    return SimpleNamespace(
        get_ram=lambda: ram,
        unwrapped=SimpleNamespace(data=SimpleNamespace(memory=mem)),
    )


def _bound(ram: np.ndarray, mem: object | None = None):
    ctl = make_gohma_controller()
    ctl.bind_env(_env(ram, mem))
    return ctl


def test_unarmed_no_bow_fails() -> None:
    ram = _ram(bow=0, arrows=0)
    _plant_gohma(ram)
    ctl = _bound(ram)
    ctl.step(read_snapshot(ram))
    assert ctl.failed


def test_clean_no_poke_fails_closed_without_natural_arrows() -> None:
    ram = _ram(bow=1, arrows=0)
    _plant_gohma(ram)
    mem = _AssignMem()
    ctl = make_clean_gohma_controller()
    ctl.bind_env(_env(ram, mem))
    ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert ctl.poke_arrows is False
    assert ADDR_ARROWS not in [addr for addr, _fmt, _val in mem.calls]
    assert ADDR_SELECTED_ITEM not in [addr for addr, _fmt, _val in mem.calls]
    assert ctl.inventory_assist is None


def test_clean_suffix_gohma_does_not_poke_survival_suffix_does() -> None:
    from zelda_i.level6.spine_suffix import l6_suffix_hops

    clean = next(h for h in l6_suffix_hops(poke_arrows=False) if h.through == "level6-gohma")
    survival = next(h for h in l6_suffix_hops() if h.through == "level6-gohma")
    assert clean.stages()[-1][1].poke_arrows is False
    assert survival.stages()[-1][1].poke_arrows is True


def test_entrance_tf_skips_ow_and_forbids_pokes() -> None:
    import inspect

    from zelda_i.level6.spine import run_level6_from_entrance

    src = inspect.getsource(run_level6_from_entrance)
    assert "level6-entry" in src
    assert "allow_pokes = False" in src
    assert "poke_arrows=poke_arrows" in src
    assert "poke_arrows: bool = False" in src


def test_poke_writes_arrows_and_b_not_bow() -> None:
    ram = _ram(bow=1, arrows=0)
    _plant_gohma(ram)
    mem = _AssignMem()
    ctl = _bound(ram, mem)
    ctl.step(read_snapshot(ram))
    assert not ctl.failed
    addrs = [addr for addr, _fmt, _val in mem.calls]
    assert ADDR_ARROWS in addrs
    assert ADDR_SELECTED_ITEM in addrs
    assert ADDR_BOW not in addrs
    assert ctl.inventory_assist is not None
    assert ctl.inventory_assist["progression_writes"] == 0
    assert ctl.inventory_assist["bow_writes"] == 0


def test_gohma_success_needs_body_gone_and_arrows() -> None:
    ram = _ram(x=120, y=165, bow=1, arrows=1)
    _plant_gohma(ram)
    assert not level6_gohma_success(read_snapshot(ram))
    ram[ADDR_OBJ_TYPE + 1] = 0
    ram[ADDR_OBJ_HP + 1] = 0
    assert level6_gohma_success(read_snapshot(ram))
    ram[ADDR_ARROWS] = 0
    assert not level6_gohma_success(read_snapshot(ram))


def test_doorway_climbs_straight_up() -> None:
    ram = _ram(x=120, y=205, arrows=1)
    _plant_gohma(ram, x=150)
    ctl = _bound(ram)
    action = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert action.reason == "climb"
    assert list(action.action) == list(nes_action("UP"))


def test_on_line_strafes_toward_body() -> None:
    ram = _ram(x=120, y=STAND_Y, arrows=1)
    _plant_gohma(ram, x=160)  # far right
    ctl = _bound(ram)
    action = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert action.reason == "strafe"
    assert list(action.action) == list(nes_action("RIGHT"))


def test_shut_eye_waits_when_aligned() -> None:
    ram = _ram(x=128, y=STAND_Y, arrows=1, eye=EYE_SHUT)
    _plant_gohma(ram, x=128)
    ctl = _bound(ram)
    action = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert action.reason == "eye_wait"
    assert list(action.action) == list(nes_idle_action())
    assert ctl.arrow_pulses == 0


def test_open_eye_edge_faces_north_then_fires() -> None:
    ram = _ram(x=128, y=STAND_Y, arrows=1, eye=0x70, facing=0x02)
    _plant_gohma(ram, x=128)
    ctl = _bound(ram)
    turn = ctl.step(read_snapshot(ram))
    assert turn.reason == "face_up"
    assert list(turn.action) == list(nes_action("UP"))
    assert ctl.arrow_pulses == 0

    ram[ADDR_LINK_FACING] = FACE_NORTH
    shot = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert shot.reason == "arrow_shot"
    assert list(shot.action) == list(nes_action("UP", "B"))
    assert ctl.arrow_pulses == 1


def test_one_shot_then_cooldown_no_spray() -> None:
    ram = _ram(x=128, y=STAND_Y, arrows=1, eye=0x70, facing=FACE_NORTH)
    _plant_gohma(ram, x=128)
    ctl = _bound(ram)
    shot = ctl.step(read_snapshot(ram))
    assert shot.reason == "arrow_shot"
    nxt = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert nxt.reason == "cooldown"
    assert list(nxt.action) == list(nes_idle_action())
    assert ctl.arrow_pulses == 1


def test_out_of_ammo_fails_when_aligned_on_open_eye() -> None:
    ram = _ram(x=128, y=STAND_Y, arrows=1, rupees=0, eye=0x70, facing=FACE_NORTH)
    _plant_gohma(ram, x=128)
    ctl = _bound(ram)
    action = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert action.reason in ("out_of_ammo",)


def test_static_open_eye_fires_once_not_every_frame() -> None:
    """One shot on the first rising edge, then no re-fire while the eye
    stays statically open (no new edge, cooldown then eye_wait)."""
    ram = _ram(x=128, y=STAND_Y, arrows=1, eye=0x70, facing=FACE_NORTH)
    _plant_gohma(ram, x=128)
    ctl = _bound(ram)
    reasons = [ctl.step(read_snapshot(ram)).reason for _ in range(80)]
    assert reasons.count("arrow_shot") == 1
    assert ctl.arrow_pulses == 1
    assert reasons[-1] in ("eye_wait", "cooldown")
