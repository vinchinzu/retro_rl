"""No-ROM tests for ALTTP SRAM / script helpers."""

from __future__ import annotations

from alttp.startup import (
    BACKUP_SLOT_OFFSET,
    PRIMARY_SLOT_OFFSET,
    PROGRESS_INDICATOR_OFFSET,
    SLOT_SIZE,
    action_for,
    build_blank_sram,
    inject_sram_into_state,
    parse_s9x_blocks,
    repair_sram_checksums,
    run_button_script,
    slot_checksum,
)


def test_repair_sram_checksums_updates_primary_slot() -> None:
    sram = bytearray(0x2000)
    sram[PRIMARY_SLOT_OFFSET + 0x10] = 0x12
    repaired = repair_sram_checksums(bytes(sram))
    checksum, inverse = slot_checksum(repaired[:SLOT_SIZE])
    stored_inverse = repaired[0x4FC] | (repaired[0x4FD] << 8)
    stored_checksum = repaired[0x4FE] | (repaired[0x4FF] << 8)
    assert stored_checksum == checksum
    assert stored_inverse == inverse


def test_blank_sram_is_valid_zero_progress() -> None:
    patched = build_blank_sram()
    assert len(patched) == 0x2000
    assert patched[PRIMARY_SLOT_OFFSET + PROGRESS_INDICATOR_OFFSET] == 0
    assert patched[BACKUP_SLOT_OFFSET + PROGRESS_INDICATOR_OFFSET] == 0


def test_parse_and_inject_sram_block() -> None:
    state = b"SRA:002048:" + (b"\x00" * 2048)
    sram = bytes([0xAA]) * 0x2000
    injected = inject_sram_into_state(state, sram)
    blocks = parse_s9x_blocks(injected)
    sra = blocks["SRA"]
    assert injected[sra["data_offset"]] == 0xAA


def test_action_for_sets_named_buttons() -> None:
    action = action_for("UP", "A")
    assert action[4] == 1  # UP
    assert action[8] == 1  # A


class _FakeEnv:
    def __init__(self) -> None:
        self.actions: list[object] = []

    def step(self, action: object) -> None:
        self.actions.append(action)


def test_run_button_script_holds_frames() -> None:
    env = _FakeEnv()
    run_button_script(env, (("A", 2), ("NONE", 1)))
    assert len(env.actions) == 3
    assert env.actions[0][8] == 1
    assert env.actions[2].sum() == 0
