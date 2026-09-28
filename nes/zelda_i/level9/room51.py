"""Level 9 room 0x51 → 0x41 north thread.

The live half of the old dump/probe module: the statue-diamond waypoint walk
that ``level9.natural_path`` drives, plus the ROM door predicates the tests
pin. The probe/dump CLI half was removed 2026-09-07 (``rr-yxy6`` parked; the
power-on spine reaches 0x41 through this step function, not a fixture).

No door poke on 0x41. No object / room / door / inventory / progression /
capacity writes.
"""

from __future__ import annotations

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import lattice_door_step
from zelda_i.level9.ganon import LEVEL9
from zelda_i.level9.path import NORTH_DOOR_X
from zelda_i.level9.stairs import (
    ROOM41,
    ROOM41_ROM_SOUTH,
    ROOM51,
    ROOM51_ROM_NORTH,
    stair_loader_for,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

ROOM51_SOUTH_Y = 189
ROOM51_WEST_X = 48
ROOM51_EAST_X = 208
ROOM51_MID_Y = 141
ROOM51_THREAD_Y = 133
ROOM51_THREAD_X = 144
ROOM51_NORTH_BAND_Y = 93
ROOM51_DOOR_X_TOL = 1

_DOOR_STAND = {
    "UP": (NORTH_DOOR_X, 77),
    "DOWN": (NORTH_DOOR_X, ROOM51_SOUTH_Y),
    "LEFT": (ROOM51_WEST_X, ROOM51_MID_Y),
    "RIGHT": (ROOM51_EAST_X, ROOM51_MID_Y),
}


def in_room_51(snap: ZeldaSnapshot) -> bool:
    return snap.mode == PLAY_MODE and snap.level == LEVEL9 and snap.screen == ROOM51


def room51_rom_north_is_open() -> bool:
    return ROOM51_ROM_NORTH == 0 and ROOM41_ROM_SOUTH == 7


def room51_is_rom_predecessor_of_41() -> bool:
    """ROM only: 0x51 north open pairs 0x41 south shutter. Live dest separate."""
    return room51_rom_north_is_open()


def room51_loader_avoids_41() -> bool:
    """True when the 0x51 neighbor-scroll does not stage 0x41 doors."""
    return stair_loader_for(ROOM51).from_room != ROOM41


def room51_to_41_step(snap: ZeldaSnapshot, frame_i: int = 0) -> FrameAction:
    """Thread the statue diamond UP through 0x51 north open door -> uncleared 0x41.

    No door poke on 0x41.
    Waypoint path: (120, 205) -> y<=189 -> x<=96 -> y<=141 -> x>=128 -> y<=93 -> x<=120 -> UP.
    """
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.transitioning or snap.mode in (4, 6, 7):
        return FrameAction(nes_action("UP"), "room41_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM41:
        return FrameAction(nes_idle_action(), "room41_arrived")
    if snap.screen != ROOM51:
        return FrameAction(
            nes_idle_action(),
            f"unexpected_room_0x{snap.screen:02x}",
        )
    # ROM lattice first. The hand thread below presses UP from any pose it
    # was not tuned on: after rollout-guard detours it pinned Link under the
    # statue diamond at (112,109) for 20000 frames (C11 offset 6). Keep the
    # thread's A cadence: 0x51's $17 Like Likes engulf Link, and only a
    # slash frees him (a bare lattice walk stood swallowed at (48,181)).
    step = lattice_door_step(None, snap, "UP")
    if step is not None:
        act = nes_action(step, "A") if frame_i % 4 == 0 else nes_action(step)
        return FrameAction(act, "room51_lattice")
    x = int(snap.link_x)
    y = int(snap.link_y)

    if y > 189:
        d = "UP"
        reason = "room51_approach_south_aisle"
    elif y >= 180 and x > 96:
        d = "LEFT"
        reason = "room51_nav_west_aisle"
    elif x <= 96 and y > 141:
        d = "UP"
        reason = "room51_climb_west_aisle"
    elif y in range(137, 146) and x < 128:
        d = "RIGHT"
        reason = "room51_cross_center_aisle"
    elif x in range(124, 133) and y > 93:
        d = "UP"
        reason = "room51_climb_east_aisle"
    elif y <= 95 and x > 120:
        d = "LEFT"
        reason = "room51_align_north_door"
    else:
        d = "UP"
        reason = "room51_push_north"

    btn = "A" if frame_i % 4 == 0 else ""
    act = nes_action(d, btn) if btn else nes_action(d)
    return FrameAction(act, reason)


