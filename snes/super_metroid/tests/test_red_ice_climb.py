from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from super_metroid.paths import GAME_DIR
from super_metroid.routes.kpdr.k5.red_ice_climb import (
    BOTTOM_FLOOR,
    LOWER_RIPPER_1,
    LOWER_RIPPER_2,
    LOWER_RIPPER_3,
    LOWER_RIPPER_4,
    MID_FLOOR,
    RIPPER_ID,
    TUNNEL_FLOOR,
    THIN_SEAT,
    RedIceBottomEdgeRunner,
    can_attach_bottom_edge,
    can_attach_ripper1_edge,
    can_attach_ripper2_edge,
    can_attach_ripper3_edge,
    can_attach_ripper4_edge,
    can_attach_tunnel_edge,
    can_attach_mid_floor_edge,
    checkpoint_supported,
    read_rippers,
)
from super_metroid.routes.kpdr.k5.red_ice_r1_to_r2 import (
    POLICY_ID as R12_POLICY,
    RedIceRipper12EdgeRunner,
)
from super_metroid.routes.kpdr.k5.red_ice_r2_to_r3 import (
    POLICY_ID as R23_POLICY,
    RedIceRipper23EdgeRunner,
)
from super_metroid.routes.kpdr.k5.red_ice_r3_to_r4 import (
    POLICY_ID as R34_POLICY,
    RedIceRipper34EdgeRunner,
)
from super_metroid.routes.kpdr.k5.red_ice_r4_to_tunnel import (
    POLICY_ID as R4TUN_POLICY,
    RedIceRipper4TunnelEdgeRunner,
)


def _state(**overrides):
    values = {
        "room_id": 0xA253,
        "samus_x": 90,
        "samus_y": 2351,
        "velocity_y": 0,
        "vertical_direction": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Env:
    def __init__(self, ram: np.ndarray) -> None:
        self._ram = ram

    def get_ram(self) -> np.ndarray:
        return self._ram


def _write_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = (value >> 8) & 0xFF


def test_checkpoint_requires_grounded_state() -> None:
    assert BOTTOM_FLOOR.matches(_state(samus_y=2443, samus_x=120))
    assert LOWER_RIPPER_1.matches(_state())
    assert not LOWER_RIPPER_1.matches(_state(vertical_direction=2))
    assert not LOWER_RIPPER_1.matches(_state(samus_y=2300))
    assert LOWER_RIPPER_2.matches(_state(samus_x=125, samus_y=2255))
    assert not LOWER_RIPPER_2.matches(_state(samus_x=125, samus_y=2351))
    assert LOWER_RIPPER_3.matches(_state(samus_x=110, samus_y=2159))
    assert not LOWER_RIPPER_3.matches(_state(samus_x=110, samus_y=2255))
    assert LOWER_RIPPER_4.matches(_state(samus_x=156, samus_y=2023))
    assert not LOWER_RIPPER_4.matches(_state(samus_x=156, samus_y=2159))
    assert TUNNEL_FLOOR.matches(_state(samus_x=104, samus_y=1883))
    assert not TUNNEL_FLOOR.matches(_state(samus_x=155, samus_y=1883))
    assert not TUNNEL_FLOOR.matches(_state(samus_x=104, samus_y=2023))
    assert MID_FLOOR.matches(_state(samus_x=142, samus_y=1625))
    assert not MID_FLOOR.matches(_state(samus_x=142, samus_y=1591))
    assert THIN_SEAT.matches(_state(samus_x=89, samus_y=587))
    assert not THIN_SEAT.matches(_state(samus_x=166, samus_y=587))


def test_bottom_edge_attach_requires_equipped_ice_and_hi_jump() -> None:
    ready = _state(
        samus_y=2443,
        samus_x=120,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_bottom_edge(ready)
    assert not can_attach_bottom_edge(_state(**{**vars(ready), "equipped_beams": 0x1005}))
    assert not can_attach_bottom_edge(_state(**{**vars(ready), "equipped_items": 0x3005}))
    r1 = _state(
        samus_x=101,
        samus_y=2351,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_ripper1_edge(r1)
    assert not can_attach_ripper1_edge(ready)
    r2 = _state(
        samus_x=125,
        samus_y=2255,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_ripper2_edge(r2)
    assert not can_attach_ripper2_edge(r1)
    assert not can_attach_ripper2_edge(ready)
    r3 = _state(
        samus_x=140,
        samus_y=2159,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_ripper3_edge(r3)
    assert not can_attach_ripper3_edge(r2)
    assert not can_attach_ripper3_edge(ready)
    r4 = _state(
        samus_x=155,
        samus_y=2023,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_ripper4_edge(r4)
    assert not can_attach_ripper4_edge(r3)
    assert not can_attach_ripper4_edge(ready)
    tunnel = _state(
        samus_x=107,
        samus_y=1883,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_tunnel_edge(tunnel)
    assert not can_attach_tunnel_edge(r4)
    mid = _state(
        samus_x=142,
        samus_y=1625,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_mid_floor_edge(mid)
    assert not can_attach_mid_floor_edge(tunnel)


def test_frozen_support_is_part_of_checkpoint_truth() -> None:
    ram = np.zeros(0x20000, dtype=np.uint8)
    base = 0x0F78 + 5 * 0x40
    _write_u16(ram, base, RIPPER_ID)
    _write_u16(ram, base + 0x02, 92)
    _write_u16(ram, base + 0x06, 2376)
    _write_u16(ram, base + 0x26, 180)
    env = _Env(ram)

    assert read_rippers(env)[0].slot == 5
    assert checkpoint_supported(env, _state(), LOWER_RIPPER_1)

    _write_u16(ram, base + 0x26, 0)
    assert not checkpoint_supported(env, _state(), LOWER_RIPPER_1)


def test_checkpoint_plan_has_one_verified_edge_and_planned_recovery_tree() -> None:
    path = (
        GAME_DIR
        / "routes"
        / "kpdr"
        / "data"
        / "red_tower_ice_checkpoint_plan.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = {row["id"]: row for row in data["checkpoints"]}
    edges = {row["id"]: row for row in data["edges"]}

    assert data["kind"] == "super_metroid_checkpoint_room_plan"
    assert data["roomIdHex"] == "0xA253"
    assert len(checkpoints) >= 20
    assert edges["bottom_to_lower_ripper_1"]["status"] == "verified_phase_sweep"
    assert edges["lower_ripper_1_to_2"]["status"] == "verified_dual_from_p165_r1"
    assert edges["lower_ripper_2_to_3"]["status"] == "verified_dual_from_p165_r2"
    assert edges["lower_ripper_3_to_4"]["status"] == "verified_dual_from_p165_r3"
    assert edges["lower_ripper_4_to_tunnel"]["status"] == "verified_dual_from_p165_r4"
    assert edges["tunnel_to_mid_floor"]["status"] == "verified_dual_from_p165_tunnel"
    assert edges["mid_floor_to_thin_seat"]["status"] == "verified_dual_from_p165_mid"
    assert data["recovery"]


def test_r12_acquire_shot_has_no_walk() -> None:
    """LEFT/RIGHT on frozen r1 walks off. Offset freeze is UP+X only."""
    from retro_harness.actions import buttons

    ram = np.zeros(0x20000, dtype=np.uint8)
    base1 = 0x0F78 + 5 * 0x40
    _write_u16(ram, base1, RIPPER_ID)
    _write_u16(ram, base1 + 0x02, 101)
    _write_u16(ram, base1 + 0x06, 2376)
    _write_u16(ram, base1 + 0x26, 180)
    base2 = 0x0F78 + 6 * 0x40
    _write_u16(ram, base2, RIPPER_ID)
    _write_u16(ram, base2 + 0x02, 120)
    _write_u16(ram, base2 + 0x06, 2280)
    runner = RedIceRipper12EdgeRunner(_Env(ram))
    runner.phase = "acquire"
    action = runner.action(
        _state(room_id=0xA253, samus_x=101, samus_y=2351, pose=3)
    )
    assert action is not None
    got = list(action)
    assert got == list(buttons("UP", "X"))
    assert got != list(buttons("RIGHT", "UP", "X"))
    assert runner.policy_id == R12_POLICY
    assert runner.from_checkpoint == "lower_ripper_1"
    assert runner.to_checkpoint == "lower_ripper_2"


def test_r23_acquire_shot_has_no_walk() -> None:
    """LEFT/RIGHT on frozen r2 walks off. Offset freeze is UP+X only."""
    from retro_harness.actions import buttons

    ram = np.zeros(0x20000, dtype=np.uint8)
    base2 = 0x0F78 + 6 * 0x40
    _write_u16(ram, base2, RIPPER_ID)
    _write_u16(ram, base2 + 0x02, 125)
    _write_u16(ram, base2 + 0x06, 2280)
    _write_u16(ram, base2 + 0x26, 180)
    base3 = 0x0F78 + 7 * 0x40
    _write_u16(ram, base3, RIPPER_ID)
    _write_u16(ram, base3 + 0x02, 145)
    _write_u16(ram, base3 + 0x06, 2184)
    runner = RedIceRipper23EdgeRunner(_Env(ram))
    runner.phase = "acquire"
    action = runner.action(
        _state(room_id=0xA253, samus_x=125, samus_y=2255, pose=1)
    )
    assert action is not None
    got = list(action)
    assert got == list(buttons("UP", "X"))
    assert got != list(buttons("RIGHT", "UP", "X"))
    assert runner.policy_id == R23_POLICY
    assert runner.from_checkpoint == "lower_ripper_2"
    assert runner.to_checkpoint == "lower_ripper_3"

    _write_u16(ram, base3 + 0x02, 98)
    wait = runner.action(
        _state(room_id=0xA253, samus_x=125, samus_y=2255, pose=1)
    )
    assert list(wait) == list(buttons("UP"))


def test_r34_acquire_shot_has_no_walk_and_crouch_is_down() -> None:
    """136px gap: freeze at offset without walking, then DOWN to crouch-jump."""
    from retro_harness.actions import buttons

    ram = np.zeros(0x20000, dtype=np.uint8)
    base3 = 0x0F78 + 7 * 0x40
    _write_u16(ram, base3, RIPPER_ID)
    _write_u16(ram, base3 + 0x02, 140)
    _write_u16(ram, base3 + 0x06, 2184)
    _write_u16(ram, base3 + 0x26, 180)
    base4 = 0x0F78 + 8 * 0x40
    _write_u16(ram, base4, RIPPER_ID)
    _write_u16(ram, base4 + 0x02, 158)
    _write_u16(ram, base4 + 0x06, 2048)
    runner = RedIceRipper34EdgeRunner(_Env(ram))
    runner.phase = "acquire"
    action = runner.action(
        _state(room_id=0xA253, samus_x=140, samus_y=2159, pose=1)
    )
    assert list(action) == list(buttons("UP", "X"))
    assert runner.policy_id == R34_POLICY
    assert runner.from_checkpoint == "lower_ripper_3"
    assert runner.to_checkpoint == "lower_ripper_4"

    runner.phase = "crouch"
    crouch = runner.action(
        _state(room_id=0xA253, samus_x=140, samus_y=2159, pose=1)
    )
    assert list(crouch) == list(buttons("DOWN"))


def test_r4tun_grounded_jump_has_no_walk_then_left_in_air() -> None:
    """LEFT on frozen r4 walks off. Crouch-jump A-only until airborne, then LEFT+A."""
    from retro_harness.actions import buttons

    runner = RedIceRipper4TunnelEdgeRunner(_Env(np.zeros(0x20000, dtype=np.uint8)))
    runner.phase = "crouch"
    crouch = runner.action(
        _state(room_id=0xA253, samus_x=155, samus_y=2023, pose=1)
    )
    assert list(crouch) == list(buttons("DOWN"))
    assert runner.policy_id == R4TUN_POLICY
    assert runner.from_checkpoint == "lower_ripper_4"
    assert runner.to_checkpoint == "tunnel_floor"

    runner.phase = "jump"
    grounded = runner.action(
        _state(
            room_id=0xA253,
            samus_x=155,
            samus_y=2028,
            pose=39,
            velocity_y=0,
            vertical_direction=0,
        )
    )
    assert list(grounded) == list(buttons("A"))

    air = runner.action(
        _state(
            room_id=0xA253,
            samus_x=155,
            samus_y=1980,
            pose=77,
            velocity_y=5,
            vertical_direction=1,
        )
    )
    assert list(air) == list(buttons("LEFT", "A"))


def test_first_wall_arc_waits_until_clear_of_lower_ripper() -> None:
    """The initial rising spin must not pass through the frozen Ripper."""
    from retro_harness.actions import buttons

    ram = np.zeros(0x20000, dtype=np.uint8)
    base = 0x0F78 + 5 * 0x40
    _write_u16(ram, base, RIPPER_ID)
    _write_u16(ram, base + 0x02, 101)
    _write_u16(ram, base + 0x06, 2376)
    _write_u16(ram, base + 0x26, 180)
    runner = RedIceBottomEdgeRunner(_Env(ram))
    runner.phase = "runup"
    runner._phase_frames = 16

    too_close = runner.action(_state(samus_x=136, samus_y=2443))
    assert list(too_close) == list(buttons("RIGHT", "B"))
    assert runner.phase == "runup"

    clear = runner.action(_state(samus_x=137, samus_y=2443))
    assert list(clear) == list(buttons("RIGHT", "B", "A"))
    assert runner.phase == "spin"
