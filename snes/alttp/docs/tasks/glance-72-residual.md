# Glance leftover — rr-ccxt.15

Planner owns `docs/STATUS.md`. Leave proof is RAM glance, not MP4.
Isolated stair leftover only. Not natural_entry, not continuous, not STATUS.

## Published leftover

`alttp.screen_glance.ROOM_01` (`hop="room_01"`, `room=0x01`) and
`ROOM_72` (`hop="room_72"`, `room=0x72`). `ROOM_50` unchanged.

Follower unset (Zelda not rescued). Sword min fighter. Indoors. Module
`0x07` submodule `0`. Keys unset.

### ROOM_01 — F1 well

Bands from `maps/room_01.json` `down_to_0x72` path
`stair_west_corridor` (760,120) pad **4** union `down_stair_approach`
(760,99) pad **2**. Not a looser well±4 / default-only band.

| Field | Value |
|-------|--------|
| hop | `room_01` |
| room | `0x01` |
| x | `[756, 764]` — 760 ±4 corridor / ±2 well |
| y | `[97, 124]` — well 99±2 union corridor 120±4 |
| module | `0x07` (indoors control-ready) |
| submodule | `0` (stair anim 14 is a miss) |
| sword_min | `1` (fighter) |
| follower | `None` (Zelda not yet rescued) |
| keys | unset |
| indoors | `True` |

CastleRoom01 spawn (960,120) and 0x50-east land ~(560,120) miss the well.

### ROOM_72 — B1 landing

Bands from `maps/room_72.json` `north_to_0x01` reverse landing
`castle_b1_guard` (1273,3665) and `f1_stair_approach` (1272,3656), pad
`pathTolerances.default` **12**.

| Field | Value |
|-------|--------|
| hop | `room_72` |
| room | `0x72` |
| x | `[1260, 1285]` — 1272/1273 ±12 |
| y | `[3644, 3677]` — 3656/3665 ±12 |
| module | `0x07` |
| submodule | `0` |
| sword_min | `1` |
| follower | `None` |
| keys | unset |
| indoors | `True` |

CastleB1Key (1320,3656) misses the stair column.

`leftover_from_snapshot` / `leftover_from_mapping` still return leftover
when misses is non-empty.

Unit: `uv run pytest alttp/tests/test_screen_glance.py -q`. No ROM.

Did not STATUS-promote. Did not import `zelda_i`. Did not write `$F3CC` /
sword / keys. Did not copy approach xy into Python (bands from map pads).
