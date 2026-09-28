"""0x16 Patra blade kill: plan set, done contract, and the guard hand-off. No emulator."""

from __future__ import annotations

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.dungeon.shot_guard import ROM_CHECKED, ShotGuard
from zelda_i.level9.patra import (
    OBJ_PATRA_2,
    OBJ_PATRA_EYE_2,
    PATRA_16_ROOM,
    PATRA_16_SPAWN_FRAMES,
    BLADE_REACH,
    BLADE_STAGE,
    Level9Patra16Controller,
    PatraBlade,
    _blade_plans,
    patra_hp,
)
from zelda_i.ram import (
    PLAY_MODE,
    WORLD_FLAG_ITEM,
    ZeldaObject,
    ZeldaSnapshot,
    read_snapshot,
    room_flags_addr,
)


def _obj(type_id: int, x: int, y: int, *, slot: int, hp: int) -> ZeldaObject:
    return ZeldaObject(slot=slot, type_id=type_id, x=x, y=y, facing=0, hp=hp, state=0)


def _snap(link=(32, 141), *objects: ZeldaObject, screen: int = PATRA_16_ROOM) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=PLAY_MODE, level=9, screen=screen, next_screen=screen,
        link_x=link[0], link_y=link[1], facing=8, sword=3, bombs=2, rupees=2, keys=0,
        health=0xFF, triforce=0xFF, compass=0, dialog_timer=0, colliding_tile=0,
        room_item_id=0, room_all_dead=0, room_obj_count=len(objects), cur_opened_doors=0,
        open_doorway_mask=0, objects=objects,
    )


class _Env:
    def __init__(self) -> None:
        self.ram = np.zeros(0x10000, dtype=np.uint8)

    def get_ram(self):
        return self.ram

    def take_item(self) -> None:
        self.ram[room_flags_addr(9) + PATRA_16_ROOM] |= WORLD_FLAG_ITEM


def test_blade_faces_only_toward_parts_in_reach() -> None:
    near = _obj(OBJ_PATRA_EYE_2, 32 + 20, 141 - 10, slot=2, hp=0x60)
    far = _obj(OBJ_PATRA_EYE_2, 32 + BLADE_REACH + 20, 141 + 30, slot=3, hp=0x60)
    faces = {face for _, _, face in _blade_plans(_snap((32, 141)), (near, far))}
    assert faces == {"RIGHT", "UP"}
    assert _blade_plans(_snap((32, 141)), (far,)) == []


def test_blade_plans_swing_earliest_first_and_turn_before_a() -> None:
    eye = _obj(OBJ_PATRA_EYE_2, 60, 141, slot=2, hp=0x60)
    plans = _blade_plans(_snap((32, 141)), (eye,))
    assert [p[0] for p in plans] == sorted(p[0] for p in plans)
    first_at, presses, face = plans[0]
    assert presses == [face, face] and first_at == 2


def test_blade_stages_on_a_lattice_node_off_the_body() -> None:
    body = _obj(OBJ_PATRA_2, 128, 125, slot=1, hp=0xB0)
    x, y = PatraBlade()._stage(_snap((32, 125)), body)
    assert x % 8 == 0 and y % 8 == 5
    assert abs(abs(x - 128) - BLADE_STAGE) <= 8 and y == 125


def test_patra_hp_counts_the_body_and_its_eyes_only() -> None:
    snap = _snap(
        (32, 141),
        _obj(OBJ_PATRA_2, 150, 120, slot=1, hp=0xB0),
        _obj(OBJ_PATRA_EYE_2, 150, 96, slot=2, hp=0x40),
        _obj(0x23, 60, 60, slot=3, hp=0x10),
    )
    assert patra_hp(snap) == 0xB0 + 0x40


def test_patra16_is_done_once_the_room_item_is_taken() -> None:
    env = _Env()
    ctl = Level9Patra16Controller()
    ctl.bind_env(env)
    assert not ctl.arrived(_snap())
    env.take_item()
    assert ctl.arrived(_snap())


def test_patra16_fails_closed_off_its_room() -> None:
    ctl = Level9Patra16Controller()
    ctl.bind_env(_Env())
    act = ctl.step(_snap(screen=0x06))
    assert ctl.failed and "patra16_left_0x06" in ctl.notes
    assert act.action == nes_idle_action()


def test_patra16_trusts_an_empty_room_only_after_the_spawn_window() -> None:
    env = _Env()
    ctl = Level9Patra16Controller()
    ctl.bind_env(env)
    for _ in range(PATRA_16_SPAWN_FRAMES):
        ctl.step(_snap())
        assert not ctl._cleared
    ctl.step(_snap())
    assert ctl._cleared


def test_guard_passes_a_rom_checked_frame_through() -> None:
    ram = np.zeros(0x10000, dtype=np.uint8)
    ram[0x12], ram[0x10], ram[0xEB] = 5, 9, 0x16
    ram[0x70], ram[0x84], ram[0x98] = 128, 141, 8
    ram[0x350], ram[0x71], ram[0x85], ram[0x99] = 0x58, 80, 141, 1
    ram[0xAD], ram[0x486] = 0x10, 0x20
    guard = ShotGuard()
    plain = FrameAction(nes_idle_action(), "hold_lane")
    assert guard.filter(read_snapshot(ram), ram, plain).reason.startswith("guard_")
    checked = FrameAction(nes_idle_action(), f"prefix_hop_7_patra16_{ROM_CHECKED}rod_wait")
    assert ShotGuard().filter(read_snapshot(ram), ram, checked) is checked
