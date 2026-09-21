# Glance leftover — rr-ccxt.5

Planner owns `docs/STATUS.md`. Leave proof is RAM glance, not MP4.

## Published leftover

`alttp.screen_glance.ROOM_50` (`hop="room_50"`, `room=0x50`).

Bands from `maps/room_50.json` spawn / east door, pad `pathTolerances.default` 12:

| Field | Value |
|-------|--------|
| hop | `room_50` |
| room | `0x50` |
| x | `[436, 492]` — spawn 448 / east door 480 ±12 |
| y | `[2668, 2692]` — y=2680 ±12 |
| module | `0x07` (indoors control-ready) |
| submodule | `0` |
| sword_min | `1` (fighter) |
| follower | `None` (Zelda not yet rescued) |
| keys | unset |
| indoors | `True` |

`leftover_from_snapshot` boots: `room`, `module`, `submodule`, `x`, `y`, `sword`, `follower`, `keys`, `indoors`, `screen`. Leftover is returned even when misses is non-empty.

Unit: `uv run pytest alttp/tests/test_screen_glance.py -q`. No ROM.

Did not STATUS-promote. Did not import `zelda_i`. Did not write `$F3CC` / sword / keys.
