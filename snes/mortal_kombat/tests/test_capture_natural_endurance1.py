"""ROM-free tests for Endurance 1 capture helpers."""

from __future__ import annotations

import numpy as np

from mortal_kombat.ram import ADDR_MATCH_COUNTER, make_test_ram
from mortal_kombat.roster import KIND_RAM_V3, StageSlot
from mortal_kombat.scripts.capture_natural_endurance1 import (
    RelabelMatchPolicy,
    apply_oracle,
    format_rle,
    mask_from_buttons,
    rle_encode,
)


def test_rle_encode_and_format() -> None:
    encoded = rle_encode([0, 0, 8, 8, 8, 144])
    assert encoded == [(0, 2), (8, 3), (144, 1)]
    text = format_rle(encoded)
    assert text.startswith("NATURAL_ENDURANCE1_RLE:")
    assert "(0, 2)" in text and "(8, 3)" in text


def test_mask_from_buttons_twelve_snes_bits() -> None:
    buttons = np.zeros(12, dtype=np.int8)
    buttons[0] = 1
    buttons[7] = 1
    buttons[11] = 1
    assert mask_from_buttons(buttons) == (1 << 0) | (1 << 7) | (1 << 11)


def test_relabel_match_pokes_copy_only() -> None:
    class Inner:
        kind = KIND_RAM_V3
        seen = -1

        def reset(self) -> None:
            return None

        def act(self, ram, rgb, *, deterministic: bool = False):
            del rgb, deterministic
            self.seen = int(ram[ADDR_MATCH_COUNTER])
            return np.zeros(12, dtype=np.int8)

    inner = Inner()
    policy = RelabelMatchPolicy(inner, 4)
    ram = make_test_ram(match_counter=7)
    policy.act(ram, None, deterministic=True)
    assert inner.seen == 4
    assert int(ram[ADDR_MATCH_COUNTER]) == 7


def test_apply_oracle_forces_every_slot_and_drops_backups() -> None:
    class Runner:
        slots = [
            StageSlot("Endurance1", "E1", 7, "old.zip", KIND_RAM_V3, backups=["pix.zip"]),
            StageSlot("Endurance1B", "E1B", 8, "old.zip", KIND_RAM_V3, backups=["pix.zip"]),
        ]

    runner = Runner()
    apply_oracle(runner, ladder_model="mk1_v3_Match5_ppo_final.zip", pixel_model=None)
    assert {slot.model for slot in runner.slots} == {"mk1_v3_Match5_ppo_final.zip"}
    assert all(slot.kind == KIND_RAM_V3 and slot.backups == [] for slot in runner.slots)
    assert {slot.match_id for slot in runner.slots} == {7, 8}
