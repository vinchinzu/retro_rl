"""Title → fresh file → Link's House exit → Hyrule Castle grounds."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import stable_retro as retro

from alttp.overworld import (
    next_direction_to_hyrule_castle,
    on_hyrule_castle_screen,
)
from alttp.paths import (
    BOOT_STATE,
    FIRST_ACTION_STATE,
    GAME_SPEC,
    HYRULE_CASTLE_GROUNDS_STATE,
    INTEGRATION_DIR,
)
from alttp.ram import AlttpSnapshot, read_snapshot
from retro_harness.env import write_state_bytes
from retro_harness.snes import idle_action, snes_action

PRIMARY_SLOT_OFFSET = 0x000
BACKUP_SLOT_OFFSET = 0xF00
SLOT_SIZE = 0x500
PROGRESS_INDICATOR_OFFSET = 0x3C5
STARTING_POINT_OFFSET = 0x3C8
PROGRESS_INDICATOR_3_OFFSET = 0x3C9
SAVEGAME_DARK_WORLD_OFFSET = 0x3CA

FRESH_PROFILE_NAME_ENTRY_SCRIPT: tuple[tuple[str, int], ...] = (
    ("A", 8),
    ("DOWN", 8),
    ("NONE", 8),
    ("RIGHT", 8),
    ("NONE", 8),
    ("RIGHT", 8),
    ("NONE", 8),
    ("RIGHT", 8),
    ("NONE", 8),
    ("START", 8),
    ("NONE", 120),
)
OPEN_FILE_SELECT_SCRIPT: tuple[tuple[str, int], ...] = (
    ("START", 5),
    ("NONE", 120),
)
FRESH_PROFILE_LOAD_SLOT_ONE_SCRIPT: tuple[tuple[str, int], ...] = (
    ("A", 8),
    ("NONE", 180),
)
FRESH_PROFILE_WAKE_SCRIPT: tuple[tuple[str, int], ...] = (
    ("NONE", 130),
    ("B", 8),
    ("NONE", 24),
    ("A", 8),
    ("NONE", 120),
    ("DOWN", 120),
    ("NONE", 180),
    ("DOWN", 120),
    ("NONE", 60),
)
FRESH_PROFILE_EXIT_HOUSE_SCRIPT: tuple[tuple[str, int], ...] = (
    ("RIGHT", 20),
    ("DOWN", 120),
    ("NONE", 180),
)


@dataclass(frozen=True)
class StartupResult:
    phase: str
    snapshot: AlttpSnapshot
    frames: int


def action_for(*buttons: str) -> np.ndarray:
    return snes_action(*buttons, dtype=np.int8)


def no_action() -> np.ndarray:
    return idle_action(dtype=np.int8)


def step_frames(env: object, action: np.ndarray, frames: int) -> None:
    for _ in range(max(0, frames)):
        env.step(action)  # type: ignore[attr-defined]


def snapshot_env(env: object) -> AlttpSnapshot:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return read_snapshot(ram)


def run_button_script(
    env: object,
    script: Iterable[tuple[str, int]],
) -> None:
    for button, frames in script:
        action = no_action() if button == "NONE" else action_for(button)
        step_frames(env, action, frames)


def slot_checksum(slot_data: bytes) -> tuple[int, int]:
    slot = bytearray(slot_data)
    slot[0x4FC:0x500] = b"\x00\x00\x00\x00"
    checksum = sum(slot[:0x4FE]) & 0xFFFF
    inverse = (0xFFFF - checksum) & 0xFFFF
    return checksum, inverse


def _repair_slot(slot_data: bytes) -> bytes:
    slot = bytearray(slot_data)
    checksum, inverse = slot_checksum(slot)
    slot[0x4FC] = inverse & 0xFF
    slot[0x4FD] = (inverse >> 8) & 0xFF
    slot[0x4FE] = checksum & 0xFF
    slot[0x4FF] = (checksum >> 8) & 0xFF
    return bytes(slot)


def repair_sram_checksums(sram_data: bytes) -> bytes:
    if len(sram_data) < BACKUP_SLOT_OFFSET + SLOT_SIZE:
        raise ValueError("Expected an 8KB ALTTP SRAM image")
    sram = bytearray(sram_data)
    primary = _repair_slot(sram[PRIMARY_SLOT_OFFSET : PRIMARY_SLOT_OFFSET + SLOT_SIZE])
    backup = _repair_slot(sram[BACKUP_SLOT_OFFSET : BACKUP_SLOT_OFFSET + SLOT_SIZE])
    sram[PRIMARY_SLOT_OFFSET : PRIMARY_SLOT_OFFSET + SLOT_SIZE] = primary
    sram[BACKUP_SLOT_OFFSET : BACKUP_SLOT_OFFSET + SLOT_SIZE] = backup
    return bytes(sram)


def build_blank_sram() -> bytes:
    return repair_sram_checksums(bytes(0x2000))


def parse_s9x_blocks(data: bytes) -> dict[str, dict[str, int]]:
    blocks: dict[str, dict[str, int]] = {}
    idx = data.find(b"\n") + 1 if data[:2] == b"#!" else 0
    while idx + 10 < len(data):
        if data[idx + 3 : idx + 4] == b":" and data[idx + 10 : idx + 11] == b":":
            try:
                tag = data[idx : idx + 3].decode("ascii")
                size = int(data[idx + 4 : idx + 10].decode("ascii"))
                data_offset = idx + 11
                if 0 <= size and data_offset + size <= len(data):
                    blocks[tag] = {"data_offset": data_offset, "size": size}
                    idx = data_offset + size
                    continue
            except Exception:
                pass
        idx += 1
    return blocks


def inject_sram_into_state(state_bytes: bytes, sram_data: bytes) -> bytes:
    blocks = parse_s9x_blocks(state_bytes)
    if "SRA" not in blocks:
        raise ValueError("State does not contain an SRA block")
    sram = repair_sram_checksums(sram_data)
    sra = blocks["SRA"]
    modified = bytearray(state_bytes)
    end = sra["data_offset"] + len(sram)
    modified[sra["data_offset"] : end] = sram
    if sra["size"] >= BACKUP_SLOT_OFFSET + SLOT_SIZE:
        backup_start = sra["data_offset"] + BACKUP_SLOT_OFFSET
        modified[backup_start : backup_start + SLOT_SIZE] = sram[:SLOT_SIZE]
    return bytes(modified)


def inject_sram_into_env(env: object, sram_data: bytes) -> None:
    state = env.em.get_state()  # type: ignore[attr-defined]
    patched = inject_sram_into_state(state, sram_data)
    env.em.set_state(patched)  # type: ignore[attr-defined]


def wait_for_title_screen(
    env: object,
    *,
    max_frames: int = 3600,
    poll_frames: int = 60,
    skip_intro_after_frames: int = 720,
) -> StartupResult:
    frames = 0
    while frames < max_frames:
        snapshot = snapshot_env(env)
        if snapshot.is_title_screen:
            return StartupResult("title", snapshot, frames)
        if frames >= skip_intro_after_frames:
            step_frames(env, action_for("START"), 3)
            frames += 3
        step_frames(env, no_action(), poll_frames)
        frames += poll_frames
    return StartupResult("timeout", snapshot_env(env), frames)


def open_file_select(env: object) -> StartupResult:
    run_button_script(env, OPEN_FILE_SELECT_SCRIPT)
    return StartupResult("file_select", snapshot_env(env), 125)


def wait_for_control(
    env: object,
    *,
    max_cycles: int = 400,
    settle_frames: int = 10,
    confirm_buttons: Iterable[str] = ("A", "B", "START", "Y"),
) -> StartupResult:
    frames = 0
    sequence = tuple(confirm_buttons)
    for cycle in range(max_cycles):
        snapshot = snapshot_env(env)
        if snapshot.has_control:
            return StartupResult("control", snapshot, frames)
        if snapshot.is_text_mode and sequence:
            step_frames(env, action_for(sequence[cycle % len(sequence)]), 2)
            frames += 2
        step_frames(env, no_action(), settle_frames)
        frames += settle_frames
    return StartupResult("timeout", snapshot_env(env), frames)


def create_fresh_profile(env: object) -> None:
    run_button_script(env, FRESH_PROFILE_NAME_ENTRY_SCRIPT)


def load_fresh_profile_slot_one(env: object) -> None:
    run_button_script(env, FRESH_PROFILE_LOAD_SLOT_ONE_SCRIPT)
    wait_for_control(env)


def advance_fresh_profile_to_links_house_overworld(env: object) -> None:
    run_button_script(env, FRESH_PROFILE_WAKE_SCRIPT)
    run_button_script(env, FRESH_PROFILE_EXIT_HOUSE_SCRIPT)


def advance_to_hyrule_castle_grounds(
    env: object,
    *,
    max_steps: int = 7200,
    settle_frames: int = 120,
) -> StartupResult:
    for step in range(max_steps):
        snapshot = snapshot_env(env)
        if on_hyrule_castle_screen(snapshot) and snapshot.has_control:
            step_frames(env, no_action(), settle_frames)
            return StartupResult("castle_grounds", snapshot_env(env), step)
        direction = next_direction_to_hyrule_castle(snapshot)
        action = no_action() if direction is None else action_for(direction)
        env.step(action)  # type: ignore[attr-defined]
    raise RuntimeError(
        "Failed to route from Link's House overworld to Hyrule Castle grounds"
    )


def build_boot_env(state_name: str = BOOT_STATE, render_mode: str | None = "rgb_array"):
    return GAME_SPEC.make_env(
        state_name,
        render_mode=render_mode,
        use_restricted_actions=retro.Actions.ALL,
    )


def boot_past_title_to_castle(
    env: object | None = None,
    *,
    close: bool = True,
) -> StartupResult:
    """Create a fresh file from title and walk to castle grounds."""
    owns_env = env is None
    if env is None:
        env = build_boot_env()
    try:
        env.reset()  # type: ignore[attr-defined]
        title = wait_for_title_screen(env)
        if title.phase != "title":
            raise RuntimeError(
                f"never reached title (mode={title.snapshot.game_mode:#04x})"
            )
        inject_sram_into_env(env, build_blank_sram())
        step_frames(env, no_action(), 2)
        open_file_select(env)
        create_fresh_profile(env)
        load_fresh_profile_slot_one(env)
        advance_fresh_profile_to_links_house_overworld(env)
        return advance_to_hyrule_castle_grounds(env)
    finally:
        if owns_env and close:
            env.close()  # type: ignore[attr-defined]


def create_castle_grounds_state(
    *,
    also_first_action: bool = True,
) -> StartupResult:
    """Run title→castle and persist development save states."""
    env = build_boot_env()
    try:
        result = boot_past_title_to_castle(env, close=False)
        if not result.snapshot.on_castle_grounds:
            raise RuntimeError(
                "startup finished off castle grounds: "
                f"mode={result.snapshot.game_mode:#04x} "
                f"screen={result.snapshot.screen_id:#04x} "
                f"indoors={result.snapshot.indoors}"
            )
        state = env.em.get_state()
        write_state_bytes(
            INTEGRATION_DIR / f"{HYRULE_CASTLE_GROUNDS_STATE}.state",
            state,
        )
        if also_first_action:
            write_state_bytes(
                INTEGRATION_DIR / f"{FIRST_ACTION_STATE}.state",
                state,
            )
        return result
    finally:
        env.close()
