"""Forward replay must prove the route goal as well as tape synchronization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from alttp import ram
from alttp.scripts import audit_power_on as audit


@pytest.mark.parametrize(
    "through, screen, x, y, source_ok, capacity, accepted",
    [
        ("sanctuary_exit", 0x13, 1880, 1083, True, 32, True),
        ("kakariko", 0x18, 511, 1584, True, 32, True),
        ("kakariko", 0x18, 600, 1600, True, 32, True),
        ("kakariko", 0x19, 600, 1600, True, 32, False),
        ("kakariko", 0x18, 1600, 1600, True, 32, False),
        ("kakariko", 0x13, 1880, 1083, True, 32, False),
        ("kakariko", 0x18, 511, 1584, False, 32, False),
        ("eastern_grounds", 0x1E, 3328, 2047, True, 32, True),
        ("eastern_grounds", 0x1E, 3936, 2460, True, 32, False),
        ("eastern_grounds", 0x1E, 3328, 2047, False, 32, False),
        ("eastern_grounds", 0x1E, 3328, 2047, True, 24, False),
        ("eastern_grounds", 0x1D, 2664, 1544, True, 32, False),
    ],
)
def test_overworld_replay_goal_checks_the_standing_quadrant(
    through: str, screen: int, x: int, y: int, source_ok: bool,
    capacity: int, accepted: bool,
) -> None:
    final = {
        "has_control": True, "indoors": False, "dark_world": False,
        "screen_id": screen, "link_x": x, "link_y": y,
        "has_fighter_sword": True, "has_lamp": True, "hp": 32,
    }
    story = {
        "follower": 0, "progress": 2, "progress_flags": 0x15,
        "max_hp": capacity,
    }
    assert audit.replay_goal_accepted(
        {"through": through, "source_ok": source_ok}, final, story
    ) is accepted


@pytest.mark.parametrize("marker, at_exit, source_ok, accepted", [
    (3, True, True, True),
    (1, True, True, False),
    (None, True, True, False),
    (3, False, True, False),
    (3, True, False, False),
])
def test_hut_replay_requires_elder_marker_and_mapped_exit(
    marker: int | None, at_exit: bool, source_ok: bool, accepted: bool,
) -> None:
    from alttp.room_map import load_room_map

    door = load_room_map("room_05").door("south_to_0x1e")
    assert door is not None and door.landing_xy is not None
    x, y = door.landing_xy
    final = {
        "has_control": True, "indoors": False, "dark_world": False,
        "screen_id": 0x1E, "link_x": x, "link_y": y if at_exit else y + 32,
        "has_fighter_sword": True, "has_lamp": True, "hp": 32,
    }
    story = {
        "follower": 0, "progress": 2, "progress_flags": 0x15,
        "max_hp": 32, "map_indicator": marker,
    }
    assert audit.replay_goal_accepted(
        {"through": "sahasrahla_hut", "source_ok": source_ok}, final, story
    ) is accepted


@pytest.mark.parametrize("offset, screen, marker, source_ok, accepted", [
    (0, 0x1E, 3, True, True),
    (4, 0x1E, 3, True, True),
    (5, 0x1E, 3, True, False),
    (0, 0x1F, 3, True, False),  # Standing quadrant is not the loaded screen.
    (0, 0x1E, 1, True, False),
    (0, 0x1E, 3, False, False),
])
def test_eastern_approach_replay_requires_mapped_point_and_loaded_screen(
    offset: int, screen: int, marker: int, source_ok: bool, accepted: bool,
) -> None:
    from alttp.room_map import load_room_map

    point = load_room_map("screen_1e_eastern").point("eastern_door")
    assert point is not None
    final = {
        "has_control": True, "indoors": False, "dark_world": False,
        "screen_id": screen, "link_x": point.x + offset, "link_y": point.y,
        "has_fighter_sword": True, "has_lamp": True, "hp": 28,
    }
    story = {
        "follower": 0, "progress": 2, "progress_flags": 0x15,
        "max_hp": 32, "map_indicator": marker,
    }
    assert audit.replay_goal_accepted(
        {"through": "eastern_approach", "source_ok": source_ok}, final, story
    ) is accepted


@pytest.mark.parametrize(
    "source_ok, progress, flags, follower, submodule, accepted",
    [
        (True, 1, 0, 1, 0, False),  # Sanctuary arrival before Zelda/priest dialogue.
        (True, 2, 4, 1, 0, False),  # Zelda has not been handed off.
        (True, 2, 0, 0, 0, False),  # Story completion event is absent.
        (True, 2, 4, 0, 1, False),  # Transition has not settled.
        (False, 2, 4, 0, 0, False),  # A synchronized failed controller is still red.
        (True, 2, 4, 0, 0, True),
    ],
)
@pytest.mark.parametrize("final_version", ("current", "older", "changed"))
def test_synchronized_sanctuary_replay_requires_completed_story(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    source_ok: bool, progress: int, flags: int, follower: int,
    submodule: int, accepted: bool, final_version: str,
) -> None:
    data = np.zeros(0x24000, dtype=np.uint8)
    for offset, value in {
        ram.MODULE: 7, ram.SUBMODULE: submodule, ram.INDOORS: 1,
        ram.ROOM_ID: ram.SANCTUARY_ROOM, ram.EQUIP_SWORD: 1,
        ram.LINK_ITEM_LAMP: 1, ram.LINK_HP: 24, ram.LINK_MAX_HP: 24,
        ram.SRAM_PROGRESS: progress, ram.SRAM_PROGRESS_FLAGS: flags,
        ram.FOLLOWER: follower,
    }.items():
        data[ram.wram_index(offset)] = value
    env = SimpleNamespace(
        closed=False, get_ram=lambda: data,
        reset=lambda: (None, {}),
        step=lambda _action: (None, 0.0, False, False, {}),
        em=SimpleNamespace(), data=SimpleNamespace(),
    )
    env.close = lambda: setattr(env, "closed", True)
    final = ram.snapshot_to_diag(ram.read_snapshot(data))
    story = audit.story_glance(env)
    if final_version == "older":
        final.pop("max_hp")  # Added after the published C1 tape was recorded.
    elif final_version == "changed":
        final["link_x"] += 1  # Recorded fields still have to match.
    tape = tmp_path / "sanctuary.tape.npz"
    np.savez_compressed(
        tape, buttons=np.zeros((1, 12), dtype=np.uint8),
        frame_crc=np.asarray([audit.ram_crc(data)], dtype=np.uint32),
        meta=json.dumps({
            "start": "NONE", "through": "sanctuary", "survival": False,
            "source_ok": source_ok, "source_final": final, "source_story": story,
        }),
    )
    monkeypatch.setattr(audit, "build_boot_env", lambda **_kwargs: env)
    monkeypatch.setattr(audit, "inject_sram_into_env", lambda *_args: None)
    monkeypatch.setattr(audit, "save_end", lambda *_args: None)
    report = audit.replay(SimpleNamespace(tape=tape, out=tmp_path / "replay.json"))

    clean = accepted and final_version != "changed"
    assert env.closed
    assert report["synchronization_ok"] is (final_version != "changed")
    assert report["desynced"] == 0
    assert report["audit"]["set_state"] == report["audit"]["writes"] == 0
    assert report["route_goal_ok"] is accepted
    assert report["ok"] is clean
    assert report["clean_chain"] is clean
    assert report["development_only"] is not clean


@pytest.mark.parametrize("case", (
    "valid", "older_glance", "assisted", "same_stop", "unknown_stop", "wrong_rom", "desync", "wrong_leave",
))
def test_verified_prefix_rejects_bad_inputs_before_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    import hashlib

    data = np.zeros(0x24000, dtype=np.uint8)
    for offset, value in {
        ram.MODULE: 7, ram.INDOORS: 1, ram.ROOM_ID: ram.SANCTUARY_ROOM,
        ram.EQUIP_SWORD: 1, ram.LINK_ITEM_LAMP: 1, ram.LINK_HP: 10,
        ram.LINK_MAX_HP: 24, ram.SRAM_PROGRESS: 2, ram.SRAM_PROGRESS_FLAGS: 0x15,
    }.items():
        data[ram.wram_index(offset)] = value
    calls = []
    env = SimpleNamespace(
        get_ram=lambda: data, reset=lambda: calls.append("reset"),
        step=lambda _action: calls.append("step"),
    )
    rom = tmp_path / "rom.sfc"
    rom.write_bytes(b"test ROM identity")
    monkeypatch.setattr(audit, "INTEGRATION_DIR", tmp_path)
    meta = {
        "start": "NONE", "through": "sanctuary", "survival": False,
        "source_ok": True, "source_final": ram.snapshot_to_diag(ram.read_snapshot(data)),
        "source_story": audit.story_glance(env),
        "rom_sha256": hashlib.sha256(rom.read_bytes()).hexdigest(),
    }
    through = "kakariko"
    crc = audit.ram_crc(data)
    if case == "older_glance":
        meta["source_final"].pop("max_hp")
    elif case == "assisted":
        meta["survival"] = True
    elif case == "same_stop":
        through = "sanctuary"
    elif case == "unknown_stop":
        meta["through"] = "unknown"
    elif case == "wrong_rom":
        meta["rom_sha256"] = "other ROM"
    elif case == "desync":
        crc ^= 1
    elif case == "wrong_leave":
        meta["source_final"]["link_x"] += 1
    tape = tmp_path / "prefix.npz"
    np.savez_compressed(
        tape, buttons=np.zeros((1, 12), dtype=np.uint8),
        frame_crc=np.asarray([crc], dtype=np.uint32), meta=json.dumps(meta),
    )
    if case in ("valid", "older_glance"):
        assert audit.play_verified_prefix(env, tape, through) == (
            "sanctuary_heart", "sanctuary_exit", "sanctuary_to_kakariko",
        )
    else:
        with pytest.raises((ValueError, RuntimeError)):
            audit.play_verified_prefix(env, tape, through)
        if case not in ("desync", "wrong_leave"):
            assert calls == []  # No emulator work for an incompatible tape.
