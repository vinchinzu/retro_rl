"""ButtonTape: what the live run played is what the replay plays."""

from __future__ import annotations

import numpy as np

from zelda_i.rollout import press
from zelda_i.runner import (
    SYNC_RAM,
    TAPE_SUFFIX,
    ButtonTape,
    load_tape,
    pack_buttons,
    ram_crc,
    unpack_buttons,
)


class _FakeEnv:
    """Each step writes the button mask into RAM, so the CRC follows input."""

    def __init__(self) -> None:
        self.ram = np.zeros(0x2800, dtype=np.uint8)
        self.stepped: list[list[int]] = []

    def step(self, action):
        self.stepped.append(list(action))
        self.ram[0x100] = pack_buttons(action) & 0xFF
        self.ram[0x0E] += 1  # a zero-page temporary: outside the sync key
        return self.ram, 0.0, False, False, {}

    def get_ram(self):
        return self.ram


def test_pack_round_trips_every_nes_button() -> None:
    for names in [(), ("A",), ("UP", "A"), ("LEFT", "B"), ("START",), ("SELECT", "DOWN")]:
        frame = list(press(*names))
        assert unpack_buttons(pack_buttons(frame)) == frame


def test_tape_records_env_step_and_round_trips(tmp_path) -> None:
    env = _FakeEnv()
    tape = ButtonTape()
    tape.attach(env)
    played = [press("UP"), press("UP", "A"), press(), press("RIGHT")]
    for frame in played:
        env.step(list(frame))
    path = tape.save(tmp_path / f"t{TAPE_SUFFIX}", ram=env.get_ram(), tag="t", ok=True)

    loaded = load_tape(path)
    assert [unpack_buttons(b) for b in loaded.buttons] == [list(f) for f in played]
    assert loaded.meta == {"frames": 4, "tag": "t", "ok": True}
    assert len(loaded.crcs) == 4
    assert np.array_equal(loaded.ram, env.get_ram())

    # A replay into a fresh env reproduces every frame's key.
    replay = _FakeEnv()
    for bits, crc in zip(loaded.buttons, loaded.crcs):
        replay.step(unpack_buttons(int(bits)))
        assert ram_crc(replay.get_ram()) == int(crc)


def test_sync_key_ignores_temporaries_and_cart_wram() -> None:
    ram = np.zeros(0x2800, dtype=np.uint8)
    base = ram_crc(ram)
    ram[0x0E] = 5
    ram[0x800] = 0x70  # $6000
    assert ram_crc(ram) == base
    ram[SYNC_RAM.start] = 1
    assert ram_crc(ram) != base
