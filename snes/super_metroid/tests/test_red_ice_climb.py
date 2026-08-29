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
    UPPER_RIPPER_1,
    UPPER_RIPPER_2,
    UPPER_RIPPER_3,
    UPPER_RIPPER_4,
    HELLWAY_SILL,
    RedIceBottomEdgeRunner,
    can_attach_bottom_edge,
    can_attach_ripper1_edge,
    can_attach_ripper2_edge,
    can_attach_ripper3_edge,
    can_attach_ripper4_edge,
    can_attach_tunnel_edge,
    can_attach_mid_floor_edge,
    can_attach_thin_seat_edge,
    can_attach_upper_ripper1_edge,
    can_attach_upper_ripper2_edge,
    can_attach_upper_ripper3_edge,
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
from super_metroid.routes.kpdr.k5.red_ice_thin_to_ur1 import (
    POLICY_ID as THINUR1_POLICY,
    RedIceThinToUr1EdgeRunner,
)
from super_metroid.routes.kpdr.k5.red_ice_upper_hops import (
    POLICY_ID_UR12 as UR12_POLICY,
    POLICY_ID_UR34 as UR34_POLICY,
    RedIceUpperRipperHopRunner,
    UR12,
    UR34,
)
from super_metroid.routes.kpdr.k5.red_ice_ur3_to_hellway import (
    POLICY_ID as UR3HW_POLICY,
    RedIceUr3ToHellwayRunner,
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
    assert UPPER_RIPPER_1.matches(_state(samus_x=104, samus_y=495))
    assert not UPPER_RIPPER_1.matches(_state(samus_x=104, samus_y=587))
    assert UPPER_RIPPER_2.matches(_state(samus_x=119, samus_y=391))
    assert not UPPER_RIPPER_2.matches(_state(samus_x=119, samus_y=495))
    assert UPPER_RIPPER_3.matches(_state(samus_x=110, samus_y=295))
    assert not UPPER_RIPPER_3.matches(_state(samus_x=110, samus_y=391))
    assert UPPER_RIPPER_4.matches(_state(samus_x=110, samus_y=207))
    assert not UPPER_RIPPER_4.matches(_state(samus_x=110, samus_y=295))
    assert HELLWAY_SILL.matches(
        _state(room_id=0xA2F7, samus_x=42, samus_y=153)
    )
    assert HELLWAY_SILL.matches(
        _state(room_id=0xA2F7, samus_x=39, samus_y=139)
    )
    assert not HELLWAY_SILL.matches(_state(samus_x=192, samus_y=139))
    assert not HELLWAY_SILL.matches(
        _state(room_id=0xA2F7, samus_x=237, samus_y=139)
    )
    assert not HELLWAY_SILL.matches(
        _state(room_id=0xA2F7, samus_x=65522, samus_y=139)
    )


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
    thin = _state(
        samus_x=86,
        samus_y=587,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_thin_seat_edge(thin)
    assert not can_attach_thin_seat_edge(mid)
    ur1 = _state(
        samus_x=102,
        samus_y=495,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_upper_ripper1_edge(ur1)
    assert not can_attach_upper_ripper1_edge(thin)
    ur2 = _state(
        samus_x=119,
        samus_y=391,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_upper_ripper2_edge(ur2)
    assert not can_attach_upper_ripper2_edge(ur1)
    ur3 = _state(
        samus_x=110,
        samus_y=295,
        equipped_beams=0x1007,
        equipped_items=0x3105,
    )
    assert can_attach_upper_ripper3_edge(ur3)
    assert not can_attach_upper_ripper3_edge(ur2)


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
    assert edges["thin_seat_to_upper_ripper_1"]["status"] == "verified_dual_from_p165_thin"
    assert edges["upper_ripper_1_to_2"]["status"] == "verified_dual_from_p165_ur1"
    assert edges["upper_ripper_2_to_3"]["status"] == "verified_dual_from_p165_ur2"
    assert edges["upper_ripper_3_to_4"]["status"] == "verified_dual_from_p165_ur3"
    assert edges["upper_ripper_4_to_hellway"]["status"] == "verified_dual_from_p165_ur3"
    assert checkpoints["hellway_sill"]["roomIdHex"] == "0xA2F7"
    assert checkpoints["hellway_sill"]["x"] == [16, 80]
    assert checkpoints["hellway_sill"]["y"] == [120, 175]
    assert edges["upper_ripper_4_to_hellway"]["verification"]["isolatedLeave"] == [39, 139]
    assert edges["upper_ripper_4_to_hellway"]["verification"]["policyFrames"] == [283, 283]
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


def test_thin_ur1_face_then_freeze_has_no_walk() -> None:
    """Seat is solid so a short RIGHT turn is legal; freeze is still UP+X."""
    from retro_harness.actions import buttons

    ram = np.zeros(0x20000, dtype=np.uint8)
    base = 0x0F78 + 0 * 0x40
    _write_u16(ram, base, RIPPER_ID)
    _write_u16(ram, base + 0x02, 107)
    _write_u16(ram, base + 0x06, 520)
    runner = RedIceThinToUr1EdgeRunner(_Env(ram))
    runner.phase = "face"
    face = runner.action(
        _state(room_id=0xA253, samus_x=86, samus_y=587, pose=2, facing=4)
    )
    assert list(face) == list(buttons("RIGHT"))
    assert runner.policy_id == THINUR1_POLICY
    assert runner.from_checkpoint == "thin_seat"
    assert runner.to_checkpoint == "upper_ripper_1"

    runner.phase = "acquire"
    shot = runner.action(
        _state(room_id=0xA253, samus_x=90, samus_y=587, pose=1, facing=8)
    )
    assert list(shot) == list(buttons("UP", "X"))
    assert list(shot) != list(buttons("RIGHT", "UP", "X"))


def test_ur12_acquire_shot_has_no_walk() -> None:
    """LEFT/RIGHT on frozen ur1 walks off. Offset freeze is UP+X only."""
    from retro_harness.actions import buttons

    ram = np.zeros(0x20000, dtype=np.uint8)
    base1 = 0x0F78 + 0 * 0x40
    _write_u16(ram, base1, RIPPER_ID)
    _write_u16(ram, base1 + 0x02, 102)
    _write_u16(ram, base1 + 0x06, 520)
    _write_u16(ram, base1 + 0x26, 180)
    base2 = 0x0F78 + 3 * 0x40
    _write_u16(ram, base2, RIPPER_ID)
    _write_u16(ram, base2 + 0x02, 122)
    _write_u16(ram, base2 + 0x06, 416)
    runner = RedIceUpperRipperHopRunner(_Env(ram), UR12)
    runner.phase = "acquire"
    action = runner.action(
        _state(room_id=0xA253, samus_x=102, samus_y=495, pose=1)
    )
    assert list(action) == list(buttons("UP", "X"))
    assert list(action) != list(buttons("RIGHT", "UP", "X"))
    assert runner.policy_id == UR12_POLICY
    assert runner.from_checkpoint == "upper_ripper_1"
    assert runner.to_checkpoint == "upper_ripper_2"


def test_ur34_aims_then_freezes_in_tighter_band() -> None:
    """ur4 is higher; pose-1 UP+X at 29px misses. Aim first, 10-28px only."""
    from retro_harness.actions import buttons

    ram = np.zeros(0x20000, dtype=np.uint8)
    base3 = 0x0F78 + 2 * 0x40
    _write_u16(ram, base3, RIPPER_ID)
    _write_u16(ram, base3 + 0x02, 137)
    _write_u16(ram, base3 + 0x06, 320)
    _write_u16(ram, base3 + 0x26, 180)
    base4 = 0x0F78 + 1 * 0x40
    _write_u16(ram, base4, RIPPER_ID)
    _write_u16(ram, base4 + 0x02, 151)
    _write_u16(ram, base4 + 0x06, 232)
    runner = RedIceUpperRipperHopRunner(_Env(ram), UR34)
    runner.phase = "acquire"
    aim = runner.action(
        _state(room_id=0xA253, samus_x=134, samus_y=295, pose=1)
    )
    assert list(aim) == list(buttons("UP"))
    shot = runner.action(
        _state(room_id=0xA253, samus_x=134, samus_y=295, pose=3)
    )
    assert list(shot) == list(buttons("UP", "X"))
    _write_u16(ram, base4 + 0x02, 163)
    wait = runner.action(
        _state(room_id=0xA253, samus_x=134, samus_y=295, pose=3)
    )
    assert list(wait) == list(buttons("UP"))
    assert runner.policy_id == UR34_POLICY


def test_ur3_hellway_keeps_ur34_freeze_band_and_does_not_walk() -> None:
    """Same (10, 28) aim-then-shot as product UR34. No ice-walk. No early RIGHT."""
    from retro_harness.actions import buttons

    ram = np.zeros(0x20000, dtype=np.uint8)
    base3 = 0x0F78 + 2 * 0x40
    _write_u16(ram, base3, RIPPER_ID)
    _write_u16(ram, base3 + 0x02, 137)
    _write_u16(ram, base3 + 0x06, 320)
    _write_u16(ram, base3 + 0x26, 180)
    base4 = 0x0F78 + 1 * 0x40
    _write_u16(ram, base4, RIPPER_ID)
    _write_u16(ram, base4 + 0x02, 151)
    _write_u16(ram, base4 + 0x06, 232)
    runner = RedIceUr3ToHellwayRunner(_Env(ram))
    runner.phase = "acquire"
    aim = runner.action(
        _state(room_id=0xA253, samus_x=134, samus_y=295, pose=1)
    )
    assert list(aim) == list(buttons("UP"))
    shot = runner.action(
        _state(room_id=0xA253, samus_x=134, samus_y=295, pose=3)
    )
    assert list(shot) == list(buttons("UP", "X"))
    assert list(shot) != list(buttons("RIGHT", "UP", "X"))
    _write_u16(ram, base4 + 0x02, 163)
    wait = runner.action(
        _state(room_id=0xA253, samus_x=134, samus_y=295, pose=3)
    )
    assert list(wait) == list(buttons("UP"))
    assert runner.policy_id == UR3HW_POLICY
    assert runner.from_checkpoint == "upper_ripper_3"
    assert runner.to_checkpoint == "hellway_sill"

    runner.phase = "break"
    runner._phase_frames = 0
    burst = runner.action(
        _state(room_id=0xA253, samus_x=134, samus_y=295, pose=1)
    )
    assert list(burst) == list(buttons("UP", "X", "A"))

    runner.phase = "rise"
    keep = runner.action(
        _state(
            room_id=0xA253,
            samus_x=134,
            samus_y=184,
            pose=77,
            velocity_y=3,
            vertical_direction=1,
        )
    )
    assert list(keep) == list(buttons("A"))
    assert list(keep) != list(buttons("RIGHT", "A"))

    runner.phase = "sill"
    walk = runner.action(
        _state(
            room_id=0xA253,
            samus_x=134,
            samus_y=139,
            pose=77,
            velocity_y=0,
            vertical_direction=2,
        )
    )
    assert list(walk) == list(buttons("RIGHT"))

    door = runner.action(
        _state(
            room_id=0xA2F7,
            samus_x=237,
            samus_y=139,
            pose=11,
            game_state=11,
            door_transition=1,
        )
    )
    assert list(door) == list(buttons("RIGHT"))
    assert not runner.complete
    wrap = runner.action(
        _state(
            room_id=0xA2F7,
            samus_x=65522,
            samus_y=139,
            pose=11,
            game_state=8,
            door_transition=0,
        )
    )
    assert list(wrap) == list(buttons("RIGHT"))
    assert not runner.complete
    ice_leave = runner.action(
        _state(
            room_id=0xA2F7,
            samus_x=39,
            samus_y=139,
            pose=11,
            game_state=8,
            door_transition=0,
        )
    )
    assert ice_leave is None
    assert runner.complete


def test_bottom_edge_freezes_aligned_then_steps_off_and_jumps() -> None:
    """Same-column jump bonks ice. Freeze at abs dx<=6, walk away, A-only hop."""
    from retro_harness.actions import buttons

    ram = np.zeros(0x20000, dtype=np.uint8)
    base = 0x0F78 + 5 * 0x40
    _write_u16(ram, base, RIPPER_ID)
    _write_u16(ram, base + 0x02, 101)
    _write_u16(ram, base + 0x06, 2376)
    _write_u16(ram, base + 0x26, 0)
    runner = RedIceBottomEdgeRunner(_Env(ram))
    runner.phase = "acquire"

    shot = runner.action(_state(samus_x=101, samus_y=2443, pose=1))
    assert list(shot) == list(buttons("UP", "X"))
    assert list(shot) != list(buttons("RIGHT", "B", "A"))

    _write_u16(ram, base + 0x26, 180)
    runner.phase = "acquire"
    freeze = runner.action(_state(samus_x=101, samus_y=2443, pose=4))
    assert freeze is not None
    assert list(freeze) == list(buttons())
    assert runner.phase == "drop_aim"

    away = runner.action(_state(samus_x=94, samus_y=2443, pose=2))
    assert list(away) == list(buttons("LEFT"))
    assert runner.phase == "step_off"

    runner.phase = "step_off"
    runner._phase_frames = 0
    runner._target_x = 101
    brake = runner.action(_state(samus_x=73, samus_y=2443, pose=10))
    assert brake is not None
    assert runner.phase == "brake"
    assert list(brake) == list(buttons("RIGHT"))

    runner._phase_frames = 4
    hopped = runner.action(_state(samus_x=80, samus_y=2443, pose=9))
    assert hopped is not None
    assert runner.phase == "jump"
    assert list(hopped) == list(buttons("A"))
