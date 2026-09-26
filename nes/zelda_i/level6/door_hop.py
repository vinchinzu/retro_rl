"""L6 occupancy dest-hop rows.

Nine leftover geometries as ``DoorHopSpec`` rows over the shared engine in
``zelda_i.dungeon.door_hop``.  Observe + replan + stand; do not fail the hop
on occupancy miss.
"""

from __future__ import annotations

from typing import Any

from zelda_i.dungeon.door_hop import DOOR_HOP_MAX_FRAMES, DoorHopSpec
from zelda_i.dungeon.hop_controller import WAIT_SCROLL_B
from zelda_i.level6.occupancy import l6_play_dest_success, record_l6_walk
from zelda_i.level6.overworld import (
    LEVEL6,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_GLEEOK_ROOM,
    LEVEL6_GOHMA_ROOM,
    LEVEL6_GOHMA_WING_1D_ROOM,
    LEVEL6_GOHMA_WING_2C_ROOM,
    LEVEL6_GOHMA_WING_2D_ROOM,
    LEVEL6_MAP_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
)

SOUTH_DOOR_X, SOUTH_DOOR_Y, SOUTH_BAND_Y, SOUTH_DOOR_TOL = 120, 189, 181, 4
EAST_DOOR_X, EAST_DOOR_Y, EAST_DOOR_TOL = 208, 141, 4
WEST_DOOR_X, WEST_DOOR_Y, WEST_SPAWN_XMIN = 32, 141, 16
NORTH_DOOR_X, NORTH_DOOR_Y, EAST_SPAWN_XMAX = 120, 93, 232
NORTH_HALT_Y, CLIP_Y = 109, 141
SOUTH09_MAX_FRAMES = SOUTH19_MAX_FRAMES = DOOR_HOP_MAX_FRAMES
EAST29_MAX_FRAMES = EAST39_MAX_FRAMES = DOOR_HOP_MAX_FRAMES
WEST19_MAX_FRAMES = SOUTH18_MAX_FRAMES = SOUTH1D_MAX_FRAMES = WEST2D_MAX_FRAMES = NORTH2C_MAX_FRAMES = DOOR_HOP_MAX_FRAMES

__all__ = [
    "CLIP_Y", "DOOR_HOP_MAX_FRAMES", "EAST29_MAX_FRAMES", "EAST39_MAX_FRAMES",
    "EAST29_SPEC", "EAST39_SPEC", "EAST_DOOR_TOL", "EAST_DOOR_X", "EAST_DOOR_Y",
    "EAST_SPAWN_XMAX", "NORTH2C_MAX_FRAMES", "NORTH2C_SPEC",
    "NORTH_DOOR_X", "NORTH_DOOR_Y", "NORTH_HALT_Y", "SOUTH09_MAX_FRAMES",
    "SOUTH09_SPEC",
    "SOUTH18_MAX_FRAMES", "SOUTH18_SPEC", "SOUTH19_MAX_FRAMES", "SOUTH19_SPEC",
    "SOUTH1D_MAX_FRAMES", "SOUTH1D_SPEC",
    "SOUTH_BAND_Y", "SOUTH_DOOR_TOL", "SOUTH_DOOR_X", "SOUTH_DOOR_Y",
    "WEST19_MAX_FRAMES", "WEST19_SPEC", "WEST2D_MAX_FRAMES", "WEST2D_SPEC",
    "WEST_DOOR_X", "WEST_DOOR_Y", "WEST_SPAWN_XMIN",
    "L6_DOOR_HOPS",
]


def _spec(*args: Any, **kw: Any) -> DoorHopSpec:
    """One L6 row: bind the level and the L6 occupancy success/recorder."""
    kw.setdefault("level", LEVEL6)
    kw.setdefault("success_fn", l6_play_dest_success)
    kw.setdefault("record_fn", record_l6_walk)
    return DoorHopSpec(*args, **kw)


SOUTH09_SPEC = _spec(
    "level6_south_0x09", LEVEL6_ROD_WIZZ_ROOM, (SOUTH_DOOR_X, SOUTH_DOOR_Y),
    "DOWN", "occupancy to (120,189) then DOWN; halt y<=109; dest is RAM",
    north_halt_y=NORTH_HALT_Y, north_halt_reason="south_north_halt",
    south_band=True,
)
SOUTH19_SPEC = _spec(
    "level6_south_0x19", LEVEL6_MAP_ROOM, (SOUTH_DOOR_X, SOUTH_DOOR_Y),
    "DOWN", "occupancy to (120,189) then DOWN; never UP; dest is RAM",
    south_band=True, forbid_up=True,
)
EAST29_SPEC = _spec(
    "level6_east_0x29", LEVEL6_DARK_29_ROOM, (EAST_DOOR_X, EAST_DOOR_Y),
    "RIGHT", "RIGHT+DOWN clip off (55,133), occupancy y=141 RIGHT; dest is RAM",
    clip_y=CLIP_Y, clip_buttons=("RIGHT", "DOWN"), clip_side="below",
    clip_reason="east_clip", push_at_goal=True, align="y",
)
# Power-on compose leaves the clear39 leftover north of the waist at
# (95,109), not the (136,173) the mid-dungeon pin used. Occupancy grades
# every cardinal as a miss during spawn latency and boxes in place, so use
# a cardinal clip: hold DOWN to the y=141 waist, then cardinal RIGHT into
# the kill-door. No key spend.
EAST39_SPEC = _spec(
    "level6_east_0x39", LEVEL6_DARK_39_ROOM, (EAST_DOOR_X, EAST_DOOR_Y),
    "RIGHT", "cardinal DOWN to y=141 waist, then RIGHT to east door; dest is RAM",
    clip_y=CLIP_Y, clip_buttons=("DOWN",), clip_side="below",
    clip_reason="east_descend", push_at_goal=True, cardinal_hold=True,
)
WEST19_SPEC = _spec(
    "level6_west_0x19", LEVEL6_MAP_ROOM, (WEST_DOOR_X, WEST_DOOR_Y), "LEFT",
    "y=141 first, occupancy to (32,141), LEFT; halt y<=109; no KEY-UP 0x09; skip Map",
    dest_room=LEVEL6_GLEEOK_ROOM, wait_modes=WAIT_SCROLL_B,
    grid_xmin=WEST_SPAWN_XMIN, grid_ymin=NORTH_HALT_Y, push_at_goal=True,
    align="y", north_halt_y=NORTH_HALT_Y, north_halt_reason="north_key_halt",
    forbid_up=True, forbid_up_y=WEST_DOOR_Y, forbid_up_reason="north_key_halt",
    stand_reason="occupancy_stand", fail_key_up=LEVEL6_ROD_WIZZ_ROOM,
    fail_backtrack=LEVEL6_DARK_29_ROOM, track_keys=True, fail_ow=True,
    key_from="19",
)
SOUTH18_SPEC = _spec(
    "level6_south_0x18", LEVEL6_GLEEOK_ROOM, (SOUTH_DOOR_X, SOUTH_DOOR_Y),
    "DOWN",
    "x-align occupancy (120,y) then DOWN (120,189); halt y<=109; "
    "no KEY-UP 0x09; no CheckWarp north hole; south clip LIVE-TBD",
    dest_room=LEVEL6_WIZZROBE_28_ROOM, wait_modes=WAIT_SCROLL_B,
    grid_ymin=NORTH_HALT_Y, south_band=True, align="x",
    north_halt_y=NORTH_HALT_Y, north_halt_reason="north_hole_halt",
    forbid_up=True, forbid_up_reason="north_hole_halt",
    stand_reason="occupancy_stand", fail_key_up=LEVEL6_ROD_WIZZ_ROOM,
    fail_backtrack=LEVEL6_MAP_ROOM, track_keys=True, fail_ow=True, key_from="18",
)
SOUTH1D_SPEC = _spec(
    "level6_south_0x1d", LEVEL6_GOHMA_WING_1D_ROOM, (SOUTH_DOOR_X, SOUTH_DOOR_Y),
    "DOWN",
    "occupancy to (120,189) then DOWN from leftover (96,157); dest play 0x2d "
    "(120,77); keys stay 4; N/W/E wall; do not batch 0x2C/Gohma",
    dest_room=LEVEL6_GOHMA_WING_2D_ROOM, wait_modes=WAIT_SCROLL_B,
    grid_ymin=NORTH_HALT_Y, south_band=True,
    north_halt_y=NORTH_HALT_Y, north_halt_reason="south_north_halt",
    forbid_up=True, forbid_up_reason="south_north_halt",
    stand_reason="occupancy_stand", track_keys=True, fail_ow=True, key_from="1d",
)
# Power-on leftover is north-mouth (120,77). Occupancy y-align LEFT
# false-misses the waist (2px DOWN, then wizzrobe knockback) and stands
# at (80,141) until it drifts to the SW pocket (32,189). Cardinal y-align
# then LEFT, same class as EAST39_SPEC. Keys stay 3 (no top-up).
WEST2D_SPEC = _spec(
    "level6_west_0x2d", LEVEL6_GOHMA_WING_2D_ROOM, (WEST_DOOR_X, WEST_DOOR_Y),
    "LEFT",
    "cardinal UP/DOWN to y=141 from leftover (120,77), then LEFT to west "
    "door; dest play 0x2c; keys stay 3; west is open; fail 0x1D/Gohma 0x1C",
    dest_room=LEVEL6_GOHMA_WING_2C_ROOM, wait_modes=WAIT_SCROLL_B,
    grid_xmin=WEST_SPAWN_XMIN, push_at_goal=True, align="y",
    cardinal_hold=True,
    forbid_up=True, forbid_up_y=NORTH_HALT_Y, forbid_up_reason="north_back_halt",
    stand_reason="occupancy_stand", fail_backtrack=LEVEL6_GOHMA_WING_1D_ROOM,
    track_keys=True, fail_ow=True, key_from="2d",
)
# Power-on leftover is east-mouth (224,141). Occupancy LEFT false-misses
# the waist, then BFS wants DOWN and south_open_halt stands while
# wizzrobes shuffle x. Cardinal x-align then UP. Keys 3→2 (no top-up).
NORTH2C_SPEC = _spec(
    "level6_north_0x2c", LEVEL6_GOHMA_WING_2C_ROOM, (NORTH_DOOR_X, NORTH_DOOR_Y),
    "UP",
    "cardinal LEFT/RIGHT to x=120 from leftover (224,141), then KEY-UP; dest "
    "play 0x1c; keys 3->2; fail 0x2D / south 0x3C; do not fight Gohma",
    dest_room=LEVEL6_GOHMA_ROOM, wait_modes=WAIT_SCROLL_B,
    grid_xmax=EAST_SPAWN_XMAX, push_at_goal=True, align="x",
    cardinal_hold=True,
    forbid_down=True, stand_reason="occupancy_stand",
    fail_backtrack=LEVEL6_GOHMA_WING_2D_ROOM, track_keys=True, fail_ow=True,
    key_from="2c",
)
L6_DOOR_HOPS: tuple[DoorHopSpec, ...] = (
    SOUTH09_SPEC, SOUTH19_SPEC, EAST29_SPEC, EAST39_SPEC,
    WEST19_SPEC, SOUTH18_SPEC, SOUTH1D_SPEC, WEST2D_SPEC, NORTH2C_SPEC,
)
# NOTE: room 0x29's south-mouth -> north-door hop is NOT expressed as a
# generic DoorHopSpec (see rr-mzxn). The LEFT+UP clip needed to dodge the
# tile-244 hazard at (120,157) drags Link toward the west wall (x=32),
# outside this module's default OccupancyGrid xmin=40, stranding the BFS
# walker with no legal neighbor cell out of the pocket -- byte-identical
# power-on stall at (32,145), never reaching (120,93). It also needs a
# live-enemy reclear pass and a door-band clip near the goal that this
# generic spec has no hook for. See ``zelda_i.level6.inland29`` for the
# dedicated controller (restored; a wide xmin=16 grid + reclear + door-band
# clip + a 12000-frame budget instead of the shared 4000).
