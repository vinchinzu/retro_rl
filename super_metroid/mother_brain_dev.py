"""Development helpers for Mother Brain → escape → credits work.

This module is intentionally *development-only*. It door-warps, grants a late
game loadout, and writes resource/invincibility assists so the endgame segment
can be iterated without a full midgame route. Continuous acceptance still
forbids progression writes and state loads.

Door pointers (bank $83):
- ``0xAAC8`` — Rinka Shaft → Mother Brain right door
- ``0xAA8C`` — Mother Brain → Tourian Escape Room 1

Shared primitives live in :mod:`super_metroid.dev_common` and are re-exported
here for backward-compatible imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retro_harness.actions import idle_action
from super_metroid.dev_common import (
    apply_dev_survivability,
    boot_from_state,
    door_warp,
    make_dev_env,
    place_samus,
    save_dev_state,
)
from super_metroid.paths import INTEGRATION_DIR
from super_metroid.ram import (
    ADDR_ESCAPE_TIMER_FRAMES,
    ADDR_ESCAPE_TIMER_MINUTES,
    ADDR_ESCAPE_TIMER_SECONDS,
    ADDR_TIMER_TYPE,
    EVENT_MOTHER_BRAIN_DEFEATED,
    parse_env_state,
    set_event_flag,
    write_wram_u8,
    write_wram_u16,
)

# Re-export shared helpers so existing ``from mother_brain_dev import …`` keeps working.
__all__ = [
    "apply_dev_survivability",
    "boot_from_state",
    "capture_escape_room1",
    "capture_mother_brain_entry",
    "door_warp",
    "grant_late_loadout",
    "mark_mother_brain_defeated",
    "place_samus",
    "start_escape_timer",
    "DOOR_RINKA_TO_MOTHER_BRAIN",
    "DOOR_MOTHER_BRAIN_TO_ESCAPE1",
    "ROOM_MOTHER_BRAIN",
    "ROOM_ESCAPE_1",
    "ROOM_ESCAPE_2",
    "ROOM_ESCAPE_3",
    "ROOM_ESCAPE_4",
    "ROOM_LANDING_SITE",
    "SOURCE_STATE",
    "MB_ENTRY_STATE",
    "MB_TANK_STATE",
    "ESCAPE1_STATE",
]

# Almost-full late-game loadout for endgame development.
# Items: morph/bombs/hijump/speed/space/screw/gravity/varia/spring/grapple/xray-ish.
LATE_ITEMS = 0xF32F
# Charge + wave + ice + plasma (no spazer conflict).
LATE_BEAMS = 0x100B

DOOR_RINKA_TO_MOTHER_BRAIN = 0xAAC8
DOOR_MOTHER_BRAIN_TO_ESCAPE1 = 0xAA8C

ROOM_MOTHER_BRAIN = 0xDD58
ROOM_ESCAPE_1 = 0xDE4D
ROOM_ESCAPE_2 = 0xDE7A
ROOM_ESCAPE_3 = 0xDEA7
ROOM_ESCAPE_4 = 0xDEDE
ROOM_LANDING_SITE = 0x91F8

SOURCE_STATE = INTEGRATION_DIR / "natural_post_spore_spawn.state"
MB_ENTRY_STATE = INTEGRATION_DIR / "dev_mother_brain_entry.state"
MB_TANK_STATE = INTEGRATION_DIR / "dev_mother_brain_at_tank.state"
ESCAPE1_STATE = INTEGRATION_DIR / "dev_escape_room1.state"


def grant_late_loadout(env: Any) -> None:
    """Write a development late-game inventory and full resources."""
    write_wram_u16(env, 0x09A2, LATE_ITEMS)
    write_wram_u16(env, 0x09A4, LATE_ITEMS)
    write_wram_u16(env, 0x09A6, LATE_BEAMS)
    write_wram_u16(env, 0x09A8, LATE_BEAMS)
    write_wram_u16(env, 0x09C4, 1499)
    write_wram_u16(env, 0x09C2, 1499)
    write_wram_u16(env, 0x09C8, 230)
    write_wram_u16(env, 0x09C6, 230)
    write_wram_u16(env, 0x09CC, 50)
    write_wram_u16(env, 0x09CA, 50)
    write_wram_u16(env, 0x09D0, 50)
    write_wram_u16(env, 0x09CE, 50)
    write_wram_u16(env, 0x09D4, 400)
    write_wram_u16(env, 0x09D6, 400)
    write_wram_u16(env, 0x09D2, 1)  # missiles selected
    write_wram_u16(env, 0x079F, 5)  # Tourian area index


def start_escape_timer(env: Any, *, minutes: int = 3) -> None:
    """Arm a Tourian-style escape timer (development)."""
    write_wram_u8(env, ADDR_TIMER_TYPE, 2)
    write_wram_u8(env, ADDR_ESCAPE_TIMER_MINUTES, minutes)
    write_wram_u8(env, ADDR_ESCAPE_TIMER_SECONDS, 0)
    write_wram_u8(env, ADDR_ESCAPE_TIMER_FRAMES, 0)


def mark_mother_brain_defeated(env: Any) -> None:
    """Set the MB-defeated event and Tourian boss bit (development)."""
    set_event_flag(env, EVENT_MOTHER_BRAIN_DEFEATED)
    # Zebetite progress events 3/4/5.
    for event_id in (3, 4, 5):
        set_event_flag(env, event_id)
    # boss_bits_for_area[Tourian]
    write_wram_u8(env, 0xD828 + 5, 0x02)


def capture_mother_brain_entry(
    *,
    source: Path = SOURCE_STATE,
    output: Path = MB_ENTRY_STATE,
    interior_x: int = 880,
    interior_y: int = 180,
) -> dict[str, object]:
    """Door-warp into Mother Brain and save a settled interior state."""
    env = make_dev_env()
    try:
        boot_from_state(env, source)
        grant_late_loadout(env)
        state = door_warp(
            env, DOOR_RINKA_TO_MOTHER_BRAIN, expected_room=ROOM_MOTHER_BRAIN
        )
        if state.room_id != ROOM_MOTHER_BRAIN:
            raise RuntimeError(
                f"expected Mother Brain 0x{ROOM_MOTHER_BRAIN:04X}, "
                f"got 0x{state.room_id:04X}"
            )
        grant_late_loadout(env)
        place_samus(env, interior_x, interior_y)
        for _ in range(20):
            apply_dev_survivability(env)
            env.step(idle_action())
        state = parse_env_state(env)
        save_dev_state(env, output)
        return {
            "statePath": str(output.resolve()),
            "roomIdHex": f"0x{state.room_id:04X}",
            "samusX": state.samus_x,
            "samusY": state.samus_y,
            "enemy0Hp": state.enemy0_hp,
            "developmentOnly": True,
        }
    finally:
        env.close()


def capture_escape_room1(
    *,
    source: Path = SOURCE_STATE,
    output: Path = ESCAPE1_STATE,
    # y≈100 is free pipe corridor air; y=139 door-height embeds in solids.
    interior_x: int = 400,
    interior_y: int = 100,
) -> dict[str, object]:
    """Skip to post-MB escape: event flags + door warp into Escape Room 1."""
    env = make_dev_env()
    try:
        boot_from_state(env, source)
        grant_late_loadout(env)
        mark_mother_brain_defeated(env)
        start_escape_timer(env)
        state = door_warp(
            env, DOOR_MOTHER_BRAIN_TO_ESCAPE1, expected_room=ROOM_ESCAPE_1
        )
        if state.room_id != ROOM_ESCAPE_1:
            raise RuntimeError(
                f"expected Escape 1 0x{ROOM_ESCAPE_1:04X}, "
                f"got 0x{state.room_id:04X}"
            )
        grant_late_loadout(env)
        mark_mother_brain_defeated(env)
        start_escape_timer(env)
        place_samus(env, interior_x, interior_y)
        for _ in range(20):
            apply_dev_survivability(env)
            env.step(idle_action())
        state = parse_env_state(env)
        save_dev_state(env, output)
        return {
            "statePath": str(output.resolve()),
            "roomIdHex": f"0x{state.room_id:04X}",
            "samusX": state.samus_x,
            "samusY": state.samus_y,
            "timerType": state.timer_type,
            "escapeMinutes": state.escape_timer_minutes,
            "eventFlags": list(state.event_flags),
            "developmentOnly": True,
        }
    finally:
        env.close()
