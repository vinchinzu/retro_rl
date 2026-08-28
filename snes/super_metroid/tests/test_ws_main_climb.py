"""Contract tests for powered Main Shaft → Attic (rr-kw8t hop 2). No emulator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from super_metroid.combat.enemies import Enemy, list_enemies
from super_metroid.hop_glance import LeaveMiss
from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GameplayPhase, parse_state
from super_metroid.routes.kpdr.k6.ws_main_actions import (
    attic_door_action,
    climb_action,
)
from super_metroid.routes.kpdr.k6.ws_main_climb import (
    play_ws_main_to_attic,
    ws_main_attic_settled,
)
from super_metroid.routes.kpdr.k6.ws_main_geometry import (
    SHAFT_HOPS,
    WS_MAIN_ATTIC_DOOR_X,
    WS_MAIN_PHASES,
    ShaftRegion,
    at_ws_main_grate_seat,
    at_ws_main_left_platform,
    at_ws_main_mid_climb,
    at_ws_main_morph_drop,
    at_ws_main_pit,
    at_ws_main_west_super_band,
    classify_region,
    classify_ws_main_phase,
    ws_main_phase_index,
)
from super_metroid.routes.kpdr.k6.ws_main_ice import (
    ATOMIC_ID,
    ice_keepaway_action,
)
from super_metroid.routes.kpdr.k6.ws_main_shaft import (
    SAVE_COLUMN_WJ,
    at_ws_main_save_alcove,
    at_ws_main_save_column_wj,
    climb_until,
    save_alcove_jump,
    save_column_walljump,
)
from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS
from super_metroid.routes.kpdr.room_ids import ROOM_WS_ATTIC, ROOM_WS_MAIN
from super_metroid.routes.skills.basic_moves import shoot_up_action
from super_metroid.routes.skills.charge_shot import CHARGE_FULL
from super_metroid.routes.skills.geometry import PhaseStop


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_WS_MAIN,
        "samus_x": 1173,
        "samus_y": 1979,
        "pose": 1,
        "facing": FACING_RIGHT,
        "health": 299,
        "max_health": 299,
        "game_state": 8,
        "door_transition": 0,
        "selected_item": 0,
        "boss_bits": (0, 0, 0, 0x01, 0, 0, 0, 0),
    }
    values.update(overrides)
    return replace(base, **values)


class _Session:
    def __init__(self, state):
        self.state = state
        self.frame = state.frame
        self.actions = []
        self.env = None

    def step(self, action, reason):
        self.actions.append((action, reason))
        self.frame += 1
        self.state = replace(self.state, frame=self.frame)
        return self.state


def test_attic_settled_requires_gs8() -> None:
    trans = _state(room_id=ROOM_WS_ATTIC, game_state=11, door_transition=1)
    assert not ws_main_attic_settled(trans)
    gs11 = _state(room_id=ROOM_WS_ATTIC, game_state=11, door_transition=0)
    assert not ws_main_attic_settled(gs11)
    ordinary = _state(
        room_id=ROOM_WS_ATTIC, samus_x=1135, samus_y=184, pose=1, game_state=8
    )
    assert ws_main_attic_settled(ordinary)
    assert not ws_main_attic_settled(_state())
    session = _Session(ordinary)
    assert play_ws_main_to_attic(session).room_id == ROOM_WS_ATTIC
    assert session.actions == []


def test_classifier_regions_and_phases() -> None:
    assert WS_MAIN_PHASES == (
        "pit_shot",
        "grate_seat",
        "west_super",
        "mid_climb",
        "attic_seat",
        "attic_door",
    )
    assert ws_main_phase_index("grate-seat") == 1

    pin = _state()
    assert classify_region(pin) is ShaftRegion.PIT
    assert classify_ws_main_phase(pin) == "pit_shot"
    assert at_ws_main_pit(pin)
    assert not at_ws_main_grate_seat(pin)

    pocket = _state(samus_x=1177, samus_y=1883, pose=2, velocity_y=0)
    assert not at_ws_main_grate_seat(pocket)
    assert classify_region(pocket) is ShaftRegion.PIT
    assert classify_ws_main_phase(pocket) == "pit_shot"

    fire = _state(samus_x=1223, samus_y=1860, pose=3, velocity_y=0)
    assert at_ws_main_grate_seat(fire)
    assert classify_region(fire) is ShaftRegion.GRATE_SEAT
    assert classify_ws_main_phase(fire) == "grate_seat"

    take04 = _state(samus_x=1195, samus_y=1883, pose=3, velocity_y=0)
    assert at_ws_main_grate_seat(take04)
    assert classify_ws_main_phase(take04) == "grate_seat"

    stairs = _state(samus_x=1111, samus_y=1899, pose=157, velocity_y=0)
    assert classify_region(stairs) is ShaftRegion.PIT
    assert classify_region(stairs, lip_hit=True) is ShaftRegion.PIT
    assert classify_ws_main_phase(stairs) == "pit_shot"
    assert not at_ws_main_left_platform(1111, 1899, 157)
    assert at_ws_main_left_platform(1082, 1878, 2)
    shelf = _state(samus_x=1082, samus_y=1878, pose=2, velocity_y=0)
    assert classify_region(shelf) is ShaftRegion.PIT
    assert classify_region(shelf, lip_hit=True) is ShaftRegion.SHELF

    west = _state(samus_x=1152, samus_y=1675, pose=10)
    assert at_ws_main_west_super_band(west)
    assert classify_ws_main_phase(west) == "west_super"
    assert classify_region(west) is ShaftRegion.SHAFT
    assert not at_ws_main_west_super_band(_state(room_id=0xCDA8, samus_y=1675))

    mid = _state(samus_x=1152, samus_y=680, pose=1)
    assert at_ws_main_mid_climb(mid)
    assert classify_ws_main_phase(mid) == "mid_climb"

    seat = _state(samus_x=1135, samus_y=80, pose=1)
    assert classify_ws_main_phase(seat) == "attic_seat"
    assert classify_region(seat) is ShaftRegion.ATTIC_SEAT

    attic = _state(room_id=ROOM_WS_ATTIC, samus_x=1135, samus_y=184, pose=1)
    assert classify_ws_main_phase(attic) == "attic_door"
    assert classify_region(attic) is ShaftRegion.ATTIC


def test_two_hop_take02() -> None:
    pin = climb_action(1173, 1979, 1, FACING_LEFT)
    assert pin == ("LEFT",)
    assert "A" not in pin
    short = climb_action(1166, 1979, 2, FACING_LEFT)
    assert short == ("A",)
    assert "RIGHT" not in short
    land = climb_action(1166, 1979, 1, FACING_RIGHT)
    assert land == ("LEFT",)
    assert "A" not in land
    committed = climb_action(1156, 1979, 1, FACING_RIGHT)
    assert committed == ("A",)
    rise = climb_action(1156, 1920, 77, FACING_RIGHT, velocity_y=5)
    assert rise == ("A",)
    over = climb_action(1156, 1880, 81, FACING_RIGHT, velocity_y=3)
    assert over == ("RIGHT", "A")
    pocket = climb_action(1177, 1883, 2, FACING_RIGHT)
    assert pocket == ("A",)
    assert "LEFT" not in pocket
    stairs = climb_action(1111, 1899, 157, FACING_LEFT)
    assert stairs[0] == "RIGHT"
    assert "X" not in stairs
    shelf_recover = climb_action(1082, 1878, 2, FACING_RIGHT)
    assert shelf_recover[0] == "RIGHT"
    assert climb_action(1082, 1878, 2, FACING_RIGHT, lip_hit=True) == ("A",)
    for x, y, pose, facing, vy in (
        (1173, 1979, 1, FACING_LEFT, 0),
        (1166, 1979, 2, FACING_LEFT, 0),
        (1166, 1979, 1, FACING_RIGHT, 0),
        (1156, 1979, 1, FACING_RIGHT, 0),
        (1156, 1920, 77, FACING_RIGHT, 5),
        (1111, 1899, 157, FACING_LEFT, 0),
    ):
        names = climb_action(x, y, pose, facing, velocity_y=vy)
        assert "L" not in names
        assert "DOWN" not in names
        assert "X" not in names


def test_fire_slope_shoots_up_until_lip_hit() -> None:
    take02 = climb_action(1223, 1860, 3, FACING_RIGHT)
    assert take02 == shoot_up_action()
    assert "DOWN" not in take02
    jumped = climb_action(1223, 1860, 3, FACING_LEFT, lip_hit=True)
    assert jumped == ("LEFT", "A")
    assert "DOWN" not in jumped
    face = climb_action(1223, 1860, 3, FACING_RIGHT, lip_hit=True)
    assert face == ("LEFT",)
    assert "DOWN" not in face
    assert "DOWN" not in climb_action(1223, 1860, 3, FACING_RIGHT, charge=CHARGE_FULL)


def test_morph_drop_only_after_lip_hit() -> None:
    assert at_ws_main_morph_drop(1189, 1785, 2)
    assert not at_ws_main_morph_drop(1223, 1860, 3)
    assert "DOWN" not in climb_action(1189, 1785, 2, FACING_LEFT)
    assert climb_action(1189, 1785, 2, FACING_LEFT, lip_hit=True) == ("DOWN",)
    assert "DOWN" not in climb_action(1223, 1860, 3, FACING_RIGHT, lip_hit=True)


def test_shaft_hops_are_dpad_sides() -> None:
    assert SHAFT_HOPS
    for hop in SHAFT_HOPS:
        assert hop.side in ("LEFT", "RIGHT")
        assert hop.takeoff.side in ("LEFT", "RIGHT")


def test_save_alcove_jumps_left() -> None:
    planted = _state(samus_x=1235, samus_y=1851, pose=10, facing=FACING_LEFT)
    assert at_ws_main_save_alcove(planted)
    assert classify_region(planted) is ShaftRegion.SAVE_ALCOVE
    session = _Session(planted)
    save_alcove_jump(session, "test")
    assert session.actions[0][1] == "test_alcove_jump"
    face = _Session(_state(samus_x=1235, samus_y=1851, pose=9, facing=FACING_RIGHT))
    save_alcove_jump(face, "test")
    assert face.actions[0][1] == "test_alcove_face"


def test_save_column_wj_band_excludes_lip_pocket_save_door() -> None:
    leftover = _state(samus_x=1220, samus_y=1843, pose=77, velocity_y=0)
    assert at_ws_main_save_column_wj(leftover)
    human = _state(samus_x=1216, samus_y=1852, pose=19, velocity_y=2)
    assert at_ws_main_save_column_wj(human)
    assert not at_ws_main_save_column_wj(
        _state(samus_x=1220, samus_y=1843, pose=77, velocity_y=2, facing=FACING_LEFT)
    )
    assert not at_ws_main_save_column_wj(
        _state(samus_x=1177, samus_y=1883, pose=2, velocity_y=0)
    )
    assert not at_ws_main_save_column_wj(
        _state(samus_x=1232, samus_y=1843, pose=77, velocity_y=0)
    )
    assert SAVE_COLUMN_WJ.into == "RIGHT"
    assert SAVE_COLUMN_WJ.flip == "LEFT"
    session = _Session(human)
    save_column_walljump(session, "test", lambda st: False)
    reasons = [r for _, r in session.actions]
    assert any(r.endswith("_save_wj_into") for r in reasons)
    assert any(r.endswith("_save_wj_flip") for r in reasons)


def test_climb_until_overlay_save_and_lip() -> None:
    def _stop_after_first(sess: _Session):
        return lambda st: sess.frame > 0

    alcove = _Session(
        _state(samus_x=1235, samus_y=1851, pose=10, facing=FACING_LEFT)
    )
    climb_until(alcove, "test", _stop_after_first(alcove))
    assert any("alcove" in r for _, r in alcove.actions)

    column = _Session(
        _state(samus_x=1216, samus_y=1852, pose=19, velocity_y=2)
    )
    climb_until(column, "test", _stop_after_first(column))
    assert any(r.endswith("_save_wj_into") for _, r in column.actions)

    shelf = _Session(
        _state(samus_x=1082, samus_y=1878, pose=2, facing=FACING_RIGHT)
    )
    climb_until(shelf, "test", _stop_after_first(shelf))
    assert not any("shelf_hole" in r for _, r in shelf.actions)
    assert any("_pit" in r for _, r in shelf.actions)

    stairs = _Session(
        _state(samus_x=1111, samus_y=1899, pose=157, facing=FACING_LEFT)
    )
    climb_until(stairs, "test", _stop_after_first(stairs))
    assert not any("shelf_hole" in r for _, r in stairs.actions)
    assert any("_pit" in r for _, r in stairs.actions)

    lip = _Session(
        _state(samus_x=1223, samus_y=1860, pose=3, facing=FACING_RIGHT)
    )
    climb_until(lip, "test", _stop_after_first(lip))
    assert any(r.endswith("_lip_up") for _, r in lip.actions)
    assert not any("DOWN" in str(a) for a, _ in lip.actions)


def test_play_shots_climbs_jumps(monkeypatch: pytest.MonkeyPatch) -> None:
    from super_metroid.routes.kpdr.k6 import ws_main_climb as mod

    session = _Session(_state())
    seen: dict[str, object] = {}

    def _require(sess, room, label):
        seen["require"] = (room, label)

    def _shot(sess, label):
        seen["shot"] = label

    def _climb(sess, label, done):
        del done
        seen.setdefault("climbs", []).append(label)
        sess.state = replace(sess.state, samus_x=1135, samus_y=80, pose=1)

    def _jump(sess, label):
        seen["jump"] = label
        sess.state = replace(sess.state, room_id=ROOM_WS_ATTIC, game_state=11)

    def _settle(sess, dest_room, *, label, settle_frames=200, land_frames=90):
        del settle_frames, land_frames
        seen["settle"] = (dest_room, label)
        sess.state = replace(sess.state, room_id=dest_room, game_state=8, door_transition=0)
        return sess.state

    monkeypatch.setattr(mod, "require_room", _require)
    monkeypatch.setattr(mod, "three_shot_tunnel", _shot)
    monkeypatch.setattr(mod, "climb_until", _climb)
    monkeypatch.setattr(mod, "_jump_up_attic", _jump)
    monkeypatch.setattr(mod, "settle_ceiling_dest", _settle)
    out = mod.play_ws_main_to_attic(session)
    assert seen["require"][0] == ROOM_WS_MAIN
    assert seen["shot"] == "ws_main_to_attic_pit_shot"
    assert seen["climbs"] == [
        "ws_main_to_attic_grate_seat",
        "ws_main_to_attic_west_super",
        "ws_main_to_attic_mid_climb",
        "ws_main_to_attic_attic_seat",
    ]
    assert seen["jump"] == "ws_main_to_attic"
    assert seen["settle"] == (ROOM_WS_ATTIC, "ws_main_to_attic")
    assert ws_main_attic_settled(out)


def test_phased_play_stop_at_pit_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    from super_metroid.routes.kpdr.k6 import ws_main_climb as mod

    session = _Session(_state())
    seen: list[str] = []

    def _require(sess, room, label):
        del sess, room, label

    def _shot(sess, label):
        seen.append(label)
        sess.state = replace(sess.state, samus_x=1104, samus_y=1981, pose=48)

    monkeypatch.setattr(mod, "require_room", _require)
    monkeypatch.setattr(mod, "three_shot_tunnel", _shot)
    monkeypatch.setattr(mod, "climb_until", lambda *a, **k: None)
    with pytest.raises(PhaseStop) as caught:
        mod.play_ws_main_to_attic(session, start="pit_shot", stop="pit_shot")
    assert caught.value.phase == "pit_shot"
    assert seen == ["ws_main_to_attic_pit_shot"]


def test_wrong_room_and_missing_boss_bit() -> None:
    session = _Session(_state(room_id=0xCC6F, samus_x=657, samus_y=91, pose=2))
    with pytest.raises(LeaveMiss, match="ws_main_to_attic") as caught:
        play_ws_main_to_attic(session)
    err = caught.value
    assert err.leftover["xy"] == [657, 91]
    assert "expected Attic 0xCA52, got 0xCC6F" in str(err)
    boss = _Session(_state(boss_bits=(0, 0, 0, 0, 0, 0, 0, 0)))
    with pytest.raises(LeaveMiss, match="not defeated") as caught_boss:
        play_ws_main_to_attic(boss)
    assert caught_boss.value.leftover["xy"] == [1173, 1979]


def test_never_uses_l_as_hop_side() -> None:
    cases = (
        (1173, 1979, 1, FACING_LEFT, 0, False),
        (1166, 1979, 2, FACING_LEFT, 0, False),
        (1156, 1979, 1, FACING_RIGHT, 0, False),
        (1156, 1920, 77, FACING_RIGHT, 5, False),
        (1177, 1883, 2, FACING_RIGHT, 0, False),
        (1111, 1899, 157, FACING_LEFT, 0, False),
        (1223, 1860, 3, FACING_RIGHT, 0, False),
        (1223, 1860, 3, FACING_LEFT, 0, True),
        (1082, 1878, 2, FACING_RIGHT, 0, True),
        (1152, 1675, 2, FACING_LEFT, 0, False),
        (1152, 680, 1, FACING_RIGHT, 0, False),
        (1135, 80, 1, FACING_LEFT, 0, False),
    )
    for x, y, pose, facing, vy, lip in cases:
        names = climb_action(x, y, pose, facing, velocity_y=vy, lip_hit=lip)
        assert "L" not in names
        assert "SUPER" not in names
    for hop in SHAFT_HOPS:
        assert hop.side != "L"
        assert hop.takeoff.side != "L"
    for frame in range(80):
        names = attic_door_action(WS_MAIN_ATTIC_DOOR_X, 80, 1, frame)
        assert "L" not in names
        assert "SUPER" not in names


def test_registered() -> None:
    from super_metroid.routes.kpdr.k6 import play_ws_main_to_attic as play

    assert KPDR_SEGMENTS["ws_main_to_attic"] is play


def test_probe_uses_repo_headed() -> None:
    src = Path(__file__).resolve().parents[1] / "scripts" / "probe" / "ws_main_climb.py"
    text = src.read_text(encoding="utf-8")
    assert "from retro_harness.headed import" in text
    assert "add_headed_flag" in text
    assert "attach_headed" in text
    assert "--stop-at" in text
    assert "play_ws_main_to_attic" in text


def test_ice_overlay() -> None:
    from super_metroid.routes.kpdr.k6.ws_main_ice import (
        COVERN_ID,
        shelf_covern_ice_action,
    )

    blob = Enemy(0, ATOMIC_ID, 1150, 1160, 250, 0)
    assert ice_keepaway_action(1173, 1979, FACING_LEFT, (blob,)) is None
    shot = ice_keepaway_action(1152, 1163, FACING_LEFT, (blob,))
    assert shot is not None and ("X" in shot or "A" in shot)
    assert ice_keepaway_action(1152, 1163, FACING_LEFT, ()) is None
    covern = Enemy(0, COVERN_ID, 1129, 1818, 80, 0)
    assert ice_keepaway_action(1075, 1845, FACING_LEFT, (covern,)) is not None
    frozen = Enemy(0, ATOMIC_ID, 1152, 1163, 250, 80)
    assert ice_keepaway_action(1152, 1163, FACING_LEFT, (frozen,)) == ()
    shelf = Enemy(0, COVERN_ID, 1129, 1818, 80, 0)
    stairs = Enemy(1, COVERN_ID, 1048, 1928, 80, 0)
    live = shelf_covern_ice_action(1082, 1878, FACING_RIGHT, (shelf, stairs))
    assert live is not None and "LEFT" not in live
    charged = shelf_covern_ice_action(
        1082, 1878, FACING_RIGHT, (shelf, stairs), charge=CHARGE_FULL
    )
    assert charged and "A" in charged
    assert shelf_covern_ice_action(1082, 1878, FACING_RIGHT, (stairs,)) is None
    frozen_c = Enemy(0, COVERN_ID, 1129, 1818, 80, 80)
    assert shelf_covern_ice_action(1082, 1878, FACING_RIGHT, (frozen_c, stairs)) is None

    ram = np.zeros(0x2000, dtype=np.uint8)
    base = 0x0F78
    ram[base] = ATOMIC_ID & 0xFF
    ram[base + 1] = ATOMIC_ID >> 8
    ram[base + 0x02] = 1150 & 0xFF
    ram[base + 0x03] = 1150 >> 8
    ram[base + 0x06] = 1163 & 0xFF
    ram[base + 0x07] = 1163 >> 8
    ram[base + 0x14] = 250
    ram[base + 0x26] = 40
    session = _Session(_state())
    session.env = type("E", (), {"get_ram": lambda self: ram})()
    found = list_enemies(session)
    assert len(found) == 1
    assert found[0].enemy_id == ATOMIC_ID
    assert found[0].freeze_timer == 40
