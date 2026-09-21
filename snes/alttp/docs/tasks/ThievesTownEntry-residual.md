# ThievesTownEntry leftover — rr-yk6q

Planner owns `docs/STATUS.md`. Isolated / state-load only. Not natural_entry,
not continuous, not power-on, not STATUS.

## Sitting

Claimed `rr-yk6q`. Pin method: load `HyruleCastleGrounds`, poke Thieves' Town
entrance kit + `$F3C5=3`, then a **real** indoor room-load via
`which_entrance=$010E=0x34` + `Module_PreDungeon` (`$10=0x06`). Overlay is the
Thieves' Town huge-room lobby (not a freeze / Link's House / Hera).

Confirmed entry room **0xDB** (`$A0`). Map: `maps/room_db.json`.

## Pin

| Field | Value |
|-------|--------|
| save | `ThievesTownEntry.state` |
| source | `HyruleCastleGrounds` + pokes + PreDungeon |
| verification | isolated / state-load |
| which_entrance | `$010E=0x34` (kEntranceNames: `[Room #219] Thieves's Town`; `0x33` is Hera) |
| `$040C` | `0x16` (22) Thieves' Town dungeon index |
| `$0FFF` | `0` (no DW overworld walk; dungeon still loaded) |

## Pokes

`env.unwrapped.data.memory.assign` at bus `$7E:F3xx` (offset `0xF3xx` raises
`IndexError: No known mapping`; banked assign sticks). Read-after-write via
`get_ram()[wram_index(offset)]`.

| WRAM | Before | After | Name |
|------|--------|-------|------|
| `$F342` | 0 | 1 | hookshot |
| `$F343` | 0 | 10 | bombs |
| `$F345` | 0 | 1 | fire rod |
| `$F34B` | 0 | 1 | hammer |
| `$F353` | 0 | 1 | mirror |
| `$F354` | 0 | **1** | power gloves (not titans) |
| `$F355` | 0 | 1 | boots |
| `$F357` | 0 | 1 | moon pearl |
| `$F359` | 0 | 2 | master sword |
| `$F35A` | 0 | 1 | fighter shield |
| `$F36C` | 24 | 56 | max HP (7 containers) |
| `$F36D` | 24 | 56 | HP |
| `$F36E` | 0 | 128 | magic full |
| `$F379` | 248 | `0x12` | ability lift+dash (not lift2) |
| `$F3C5` | 0 | 3 | Agahnim 1 / DW open |

Did not poke `$F3CC`, `$F366`/`$F367` (prize / big-key bits), `$F354=2`.

## Glance (lobby pin)

`recordings/probe_thieves_entry/lobby.png`. Module `0x07` submodule `0`,
controllable, indoors.

| Field | Value |
|-------|--------|
| room | `0xDB` |
| xy | `(5880, 7128)` |
| module / sub | `0x07` / `0` |
| sword | `2` (master) |
| `$F3CC` | `5` (from `HyruleCastleGrounds`; **not poked**) |
| keys | `0` (`$F36F=0`; HUD keys, not the `0xFF` blank) |
| `$F354` | `1` |
| `$F357` | `1` |
| `$F345` | `1` |
| `$F342` | `1` |
| `$F34B` | `1` |
| `$F355` | `1` |
| `$F3C5` | `3` |
| `$F366`/`$F367` | `0` / `0` (no prize) |
| `$040C` | `22` (`0x16`) |
| `$010E` | `52` (`0x34`) |
| `$0FFF` | `0` |
| `$00EE` | `0` |

## First hop

**South (isolated):** `room_engine.py run room_db --edge south_to_outcasts --state ThievesTownEntry --no-clear`

`ok=True` `phase=via_south_to_outcasts` frames=191. Final
`(504,1979)` module `0x09` screen `0x58` indoors=False (Village of Outcasts).
Backtrack, not the dungeon-forward hop.

**East (not isolated):** `east_to_0xDC` (huge-room compass quadrant). Conveyor
approach `(5940, 6865)`. `room_engine` fail-closed:
`ok=False` blocker `door push failed (east_to_0xDC) xy=(5952,6877)` still
`0xDB`. RIGHT on the belt does not cross into `0xDC` this sitting (pots /
conveyor / hostiles).

## Files

- `custom_integrations/Zelda3-Snes/ThievesTownEntry.state`
- `docs/tasks/ThievesTownEntry-residual.md` (this file)
- `scratch/probe_thieves_entry.py`
- `recordings/probe_thieves_entry/`
- `maps/room_db.json`

## Next action

Isolate `east_to_0xDC`: walk north of the conveyor (not on the belt) or ride
it after clearing pots / Zazaks, then `room_engine.py run room_db --edge east_to_0xDC`.
Do not STATUS.

## Non-claims

Did not STATUS-promote. Did not treat the pin as power-on. Did not poke
`$F3CC`. Did not copy approach xy into Python hops. Did not overwrite
`verified_tip_run.json`. Did not claim `rr-ccxt.20` / `.21`.
