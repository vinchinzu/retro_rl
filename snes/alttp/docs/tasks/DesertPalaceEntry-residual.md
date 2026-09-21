# Desert Palace entry pin — `rr-alwe`

Planner owns `docs/STATUS.md`. Dev pin, not continuous, not power-on.
Isolated / state-load only. Hypothesis room **0x84 confirmed** (`$A0`).

## Result: **ok** — `DesertPalaceEntry.state` + isolated `north_to_0x74`

Main/book entrance (ROM `$02C813` index **0x09** → room `0x84`), not west
`0x83` / east `0x85` / north-ledge `0x63`. Overlay after the north hop is the
Desert Palace pot corridor. South DOWN lands overworld desert `(296, 3240)`
screen `0x30` (Archipelago south-exit coords).

## Pin method

Source: `FighterSword` (room `0x55`, sword already `1`, lamp/book/bow/boots `0`).
Pokes via `env.unwrapped.data.memory.assign`. High WRAM is banked
`0x7EF3xx` (bare `$F3xx` raises `No known mapping`); low WRAM `$010E` as-is.
Read-after-write through `get_ram()` / `wram_index`. Then real dungeon load:
`$010E=0x09`, module `$10=0x06` (PreDungeon), settle to `$10=0x07` `$11=0`.

Overworld walk was not used this sitting. Indoor pin is after a real room-load
(controllable, dungeon `$040C==6`). Isolated, not power-on.

## Pokes (every write)

| WRAM | assign | before | after | why |
|------|--------|--------|-------|-----|
| `$F340` | `0x7EF340=2` | 0 | 2 | bow+arrows |
| `$F34A` | `0x7EF34A=1` | 0 | 1 | lamp |
| `$F34E` | `0x7EF34E=1` | 0 | 1 | book |
| `$F355` | `0x7EF355=1` | 0 | 1 | boots |
| `$F359` | `0x7EF359=1` | 1 | 1 | fighter sword (already held) |
| `$F377` | `0x7EF377=30` | 0 | 30 | arrows so bow is usable |
| `$F354` | `0x7EF354=0` | 0 | 0 | gloves **not** granted (prize) |
| `$010E` | `0x010E=0x09` | 125 (`0x7D` hole) | 9 | dungeon entrance = DP main |
| `$10` | `0x06` | 7 | 6 then settles 7 | PreDungeon load |
| `$11` | `0x00` | 0 | 0 | submodule |

Did not poke `$F3CC`. Entrance 0x08 is Eastern Palace `0xC9` on this ROM
(table index 8); do not reuse the overworld door_index 0x08 as `$010E`.

## Leave proof — pin glance (`DesertPalaceEntry`)

| Field | Value |
|-------|--------|
| Room | **`0x84`** (2296, 4568) |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 |
| Indoors | true |
| Sword | `$F359==1` |
| Lamp | `$F34A==1` |
| Bow | `$F340==2` |
| Book | `$F34E==1` |
| Boots | `$F355==1` |
| Gloves | **`$F354==0`** |
| Arrows | `$F377==30` |
| Keys | `$F36F==0` |
| Follower | `$F3CC==0` |
| HP | `$F36D==24` / `$F36C==24` |
| Dungeon | `$040C==6` (Desert) |
| Entrance | `$010E==9` |
| Layer | `$00EE==0` |

## First indoor hop — `north_to_0x74` isolated ok

Straight UP from spawn wedges on the alcove north wall `y≈4376`. West around
then back to hall center, UP through the north door.

`room_engine.py run room_84 --edge north_to_0x74 --state DesertPalaceEntry`

| Field | Value |
|-------|--------|
| ok | **True** (`source=state_load_dev`, 565f) |
| Dest room | **`0x74`** (2272, 4052) |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 |
| Sword / keys / follower | 1 / 0 / 0 |
| Door-push | from (2283, 4103) UP → 0x74 |
| Settled landing | **(2272, 4052)** |

South DOWN from spawn is overworld `(296, 3240)` (backtrack, not the indoor hop).
East/west room-changes (`0x85` / `0x83`) were not isolated this sitting.

## Files

- `custom_integrations/Zelda3-Snes/DesertPalaceEntry.state`
- `docs/tasks/DesertPalaceEntry-residual.md` (this file)
- `scratch/probe_dp_entry.py`
- `maps/room_84.json`
- `recordings/probe_dp_entry/` (leftover, pin overlays, hop, isolated edge)

## Non-claims

Did not STATUS-promote. Did not treat the pin as power-on. Did not poke `$F3CC`.
Did not copy approach xy into Python. Did not overwrite `verified_tip_run.json`.
Did not claim `rr-ccxt.20` / `.21`. Did not add an `escape_graph` hop.

## Next

Isolated leftover is `0x74` (2272, 4052) keys=0. Next sitting: map `room_74`
and/or isolate 0x84 east→`0x85` / west→`0x83` (torch-key dash still in 0x84).
Not continuous. Not STATUS.
