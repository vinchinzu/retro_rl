# Skull Woods entry residual (`rr-lvnu`)

Planner owns `docs/STATUS.md`. Isolated / state-load only. Not natural_entry,
not continuous, not power-on. Did not STATUS. Did not poke `$F3CC`. Did not
copy approach xy into Python hops. Did not touch Sanctuary maps / graph /
`verified_tip_run.json`.

## Bead / save

| Field | Value |
|-------|--------|
| Bead | `rr-lvnu` (still `in_progress`) |
| Save | **`SkullWoodsEntry.state` not written** — no indoor pin with dungeon graphics |
| Source | `HyruleCastleGrounds` |
| Label | isolated / state-load |

A module-`0x07` poke of `$A0=0x58` produced RAM room `0x58` but the overlay was
still Hyrule Castle grounds in the rain. Rejected. Indoor pin only after a
real room-load.

## Pokes

`env.unwrapped.data.memory.assign(offset, "|u1", value)`. `$F3xx` is **not** a
`get_ram()` index. `assign(0xF359)` raises `No known mapping`. Working address
is **`$7EF3xx`**. Read-after-write via `get_ram()[wram_index(0xF3xx)]`.

| WRAM | Assign | Before | After | Item |
|------|--------|--------|-------|------|
| `$F342` | `$7EF342` | 0 | 1 | hookshot |
| `$F343` | `$7EF343` | 0 | 10 | bombs |
| `$F345` | `$7EF345` | 0 | 0 | fire rod **absent** (prize) |
| `$F34B` | `$7EF34B` | 0 | 1 | hammer |
| `$F354` | `$7EF354` | 0 | 1 | power gloves |
| `$F355` | `$7EF355` | 0 | 1 | boots |
| `$F357` | `$7EF357` | 0 | 1 | moon pearl |
| `$F359` | `$7EF359` | 0 | 2 | master sword |
| `$F35A` | `$7EF35A` | 0 | 1 | fighter shield |
| `$F36C` | `$7EF36C` | 24 | 56 | max HP (7 hearts) |
| `$F36D` | `$7EF36D` | 24 | 56 | HP full |
| `$F36E` | `$7EF36E` | 0 | 128 | magic full |
| `$F3C5` | `$7EF3C5` | 0 | 3 | Agahnim 1 / DW open |
| `$F3C8` | `$7EF3C8` | 0 | 3 | spawn post-Agahnim (module 0x08 still used house) |
| `$F3CA` | `$7EF3CA` | 0 | `0x40` | SRAM dark world |

Did **not** poke `$F3CC`. Source `HyruleCastleGrounds` already has `$F3CC==5`.

## Confirmed (table / ROM, not yet live indoor)

Archipelago entrance `0x29` = Skull Woods First Section Door (southeast skull,
not boss hut `0x2A` / room `0x59`):

| Field | Value |
|-------|--------|
| Entrance | `0x29` |
| Indoor room | **`0x58`** (Skull 1 Lobby / map / big-chest supertile 88) |
| OW screen | `0x40` |
| OW door xy | ~(744, 584) (entrance table `link_x=0x02E8`, `link_y=0x0248`) |
| Pinball drop | room **`0x68`** (east hole) — not this pin |
| Boss hut | entrance `0x2A`, room `0x59` — not this pin |

## Real OW path this sitting (overlays look like the world)

1. Poke kit on `HyruleCastleGrounds` (screen `0x1B`, rain, sword=2).
2. Module `$10=0x08` rebuilds OW → **Link's House** screen `0x2C` (2232, 2811).
   `$F3C8=3` did not send us to the pyramid; `$8A=0x40` was ignored.
3. South off the porch to ~(2394, 2940), LEFT: next screen is **DW `0x6B`**
   (`$8A` jumped `0x2C→0x6B` because `$F3CA=0x40` / `$0FFF==1`).
4. `0x6B` south pocket is a dead end. RIGHT → bomb-shop screen **`0x6C`**.
5. Slope at x≈2394, hold UP → **pyramid `0x5B`** ~(2408, 2528). Moon pearl
   holds Link form. Overlay is the pyramid, not a freeze.

Checkpoints (dev only, under `recordings/probe_skull_entry/`):
`dw_6c_north.state` (pyramid south), `dw_pyr_west.state` (south fence west).

## Glance leftover (pyramid south — not the dungeon)

From `dw_6c_north.state` after settle. Control-ready overworld.

| Field | Value |
|-------|--------|
| Room (`$A0` leftover) | `0x04` (last indoor; Link's House) |
| Module / sub | `$10==0x09` `$11==0` ctrl=1 |
| Screen | **`0x5B`** pyramid |
| xy | **(2408, 2528)** |
| Indoors | `False` |
| Dark world | `$0FFF==1` `$F3CA==0x40` |
| Sword | `$F359==2` |
| Keys | `$F36F==0xFF` |
| Follower | `$F3CC==5` (source; not poked) |
| Fire rod | `$F345==0` |
| Hookshot / hammer / pearl / boots / gloves | 1 / 1 / 1 / 1 / 1 |

West along the south fence reaches ~(1632, 2446) then a cliff. East stairs at
~(2444, 2488) did not climb. Hookshot north did not clear the moat. No
`0x40` screen yet.

## First hop

**Not isolated.** Need a real indoor load of `0x58` first. Map
`maps/room_58.json` not written. `room_engine.py run` not attempted (and
`in_room()` currently requires `not dark_world`).

## Files

- `docs/tasks/SkullWoodsEntry-residual.md` (this file)
- `scratch/probe_skull_entry.py`
- `recordings/probe_skull_entry/` (overlays + `poke.json` / `walk.json` / `dash.json`)

## Non-claims

Did not STATUS-promote. Did not treat a pin as power-on. Did not write
`$F3CC` / sword / keys as route facts. Did not copy approach xy into Python.
Did not overwrite `verified_tip_run.json`. Did not keep the fake `$A0=0x58`
state. Did not claim `rr-ccxt.20` / `.21`.

## Next action

From `dw_6c_north.state` (pyramid `0x5B` (2408, 2528)), reach OW **`0x40`**
(west via village `0x58`, or north around the pyramid), walk into the
southeast skull door at ~(744, 584), pin `SkullWoodsEntry.state` only when
overlay is Skull Woods lobby `0x58` (`$10==0x07` `$11==0`). Then map
`room_58.json` and isolate the first indoor door (WS/ES).
