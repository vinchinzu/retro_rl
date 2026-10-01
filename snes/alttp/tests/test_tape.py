"""Button tape pack → load round trip. No ROM."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from alttp.paths import RECORDINGS_DIR
from alttp.spine.tape import (
    TAPE_SUFFIX,
    ButtonTape,
    LiveRun,
    load_tape,
    pack_buttons,
    ram_crc,
    sha1_file,
    tape_path,
)
from retro_harness.actions import SNES_ACTION_SIZE, snes_action


class _FakeEnv:
    def __init__(self) -> None:
        self.stepped: list[list[int]] = []

    def step(self, action):
        self.stepped.append(list(action))
        return None, 0.0, False, False, {}


class _AuditMemory:
    def __init__(self) -> None:
        self.writes: list[tuple[int, str, int]] = []

    def assign(self, address, dtype, value):
        self.writes.append((address, dtype, value))


class _AuditData:
    def __init__(self) -> None:
        self.memory = _AuditMemory()

    def set_value(self, name, value):
        self.memory.assign(0x7EF36D, "|u1", value)


class _AuditEmulator:
    def __init__(self) -> None:
        self.loads: list[bytes] = []

    def set_state(self, state):
        self.loads.append(state)


class _AuditEnv(_FakeEnv):
    def __init__(self) -> None:
        super().__init__()
        self.data = _AuditData()
        self.em = _AuditEmulator()

    @property
    def unwrapped(self):
        return self


@pytest.mark.parametrize("unwrapped", [False, True])
@pytest.mark.parametrize("method", ["assign", "set_value"])
def test_live_run_counts_real_write_paths(unwrapped, method) -> None:
    raw = _AuditEnv()
    live = LiveRun(raw)
    env = live.env.unwrapped if unwrapped else live.env
    if method == "assign":
        env.data.memory.assign(0x7E0022, "|u1", 42)
    else:
        env.data.set_value("health", 24)

    assert len(raw.data.memory.writes) == 1
    assert live.report().writes == 1


@pytest.mark.parametrize("unwrapped", [False, True])
def test_live_run_counts_real_emulator_load_paths(unwrapped) -> None:
    raw = _AuditEnv()
    live = LiveRun(raw)
    env = live.env.unwrapped if unwrapped else live.env
    env.em.set_state(b"restore")

    assert raw.em.loads == [b"restore"]
    report = live.report()
    assert report.set_state == 1
    assert report.mid_run_state_load
    assert len(report.sites) == 1
    assert "test_live_run_counts_real_emulator_load_paths" in report.sites[0]


@pytest.mark.rom
def test_live_run_counts_snes_memory_and_load_on_rom() -> None:
    from alttp.ram import LINK_X, LINK_Y, read_snapshot, wram_index
    from alttp.startup import build_boot_env

    raw = build_boot_env(state_name="NONE", render_mode=None)
    try:
        raw.reset()
        live = LiveRun(raw)
        env = live.env
        before = read_snapshot(env.get_ram())
        checkpoint = env.em.get_state()
        # The real helpers unwrap, then assign each low/high WRAM byte.
        for offset, value in ((LINK_X, 1), (LINK_Y, 2), (0xE2, 0x1234), (0xE8, 0x5678)):
            env.unwrapped.data.memory.assign(offset, "|u1", value & 0xFF)
            env.unwrapped.data.memory.assign(offset + 1, "|u1", value >> 8)
        env.data.update_ram()
        after = read_snapshot(env.get_ram())
        assert (after.link_x, after.link_y) == (1, 2)
        ram = env.get_ram()
        assert [int(ram[wram_index(a)]) for a in (0xE2, 0xE3, 0xE8, 0xE9)] == [
            0x34, 0x12, 0x78, 0x56
        ]
        assert live.report().writes == 8

        env.unwrapped.em.set_state(checkpoint)
        env.data.update_ram()
        restored = read_snapshot(env.get_ram())
        assert (restored.link_x, restored.link_y) == (before.link_x, before.link_y)
        assert live.report().set_state == 1
        assert len(live.report().sites) == 1
    finally:
        raw.close()


def test_pin_provenance_records_kind_and_frame(tmp_path, monkeypatch) -> None:
    from alttp.spine import tape as tape_mod

    monkeypatch.setattr(tape_mod, "_git_identity", lambda: ("abc123", True))
    state = tmp_path / "Full_castle_to_sword.state"
    state.write_bytes(b"pin")
    sidecar = tape_mod.write_pin_provenance(
        state, source_run="spine:room_50", frame=40, kind="natural"
    )
    assert sidecar == tmp_path / "Full_castle_to_sword.provenance.json"
    loaded = tape_mod.read_pin_provenance(state)
    assert loaded is not None
    assert loaded["kind"] == "natural"
    assert loaded["frame"] == 40
    assert loaded["git_sha"] == "abc123"
    assert loaded["dirty"] is True
    assert loaded["source_run"] == "spine:room_50"
    assert tape_mod.read_pin_provenance(tmp_path / "missing.state") is None


def test_tape_path_is_recordings_tag() -> None:
    assert tape_path("verified_tip") == RECORDINGS_DIR / f"verified_tip{TAPE_SUFFIX}"
    assert tape_path(f"verified_tip{TAPE_SUFFIX}").name == f"verified_tip{TAPE_SUFFIX}"


def test_pack_load_same_vectors(tmp_path) -> None:
    frames: list[list[int]] = [
        list(snes_action()),
        list(snes_action("UP", "A", "L")),
        [1] * SNES_ACTION_SIZE,
    ]
    for index in range(SNES_ACTION_SIZE):
        vec = [0] * SNES_ACTION_SIZE
        vec[index] = 1
        frames.append(vec)
    packed = [pack_buttons(frame) for frame in frames]

    rom = tmp_path / "rom.bin"
    rom.write_bytes(b"not-a-rom")
    digest = sha1_file(rom)
    assert digest == hashlib.sha1(b"not-a-rom").hexdigest()

    env = _FakeEnv()
    tape = ButtonTape(boot_state="YazeSlot000", rom_sha1=digest)
    tape.attach(env)
    for frame in frames:
        env.step(frame)
    ram = np.arange(64, dtype=np.uint8)
    path = tape.save("opening", ram=ram, directory=tmp_path)

    assert path == tmp_path / f"opening{TAPE_SUFFIX}"
    assert [list(frame) for frame in env.stepped] == frames
    loaded = load_tape(path)
    assert loaded.buttons.tolist() == [vec.tolist() for vec in packed]
    assert loaded.boot_state == "YazeSlot000"
    assert loaded.rom_sha1 == digest
    assert loaded.final_ram_crc == ram_crc(ram)
    assert loaded.meta["frames"] == len(frames)
    assert loaded.meta["button_order"][0] == "B"
    assert len(loaded.meta["button_order"]) == SNES_ACTION_SIZE

    other = ram.copy()
    other[0] = np.uint8(int(other[0]) ^ 0xFF)
    assert ram_crc(other) != ram_crc(ram)
