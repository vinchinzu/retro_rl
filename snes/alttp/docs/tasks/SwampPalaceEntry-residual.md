# Swamp Palace entry residual (`rr-q41r`)

Planner owns `docs/STATUS.md`. Dev pin / state-load only. Not power-on, not
continuous, not STATUS. Did not poke `$F3CC`. Did not copy approach xy into
Python. Did not overwrite `verified_tip_run.json`.

## Pin

`SwampPalaceEntry.state` from `FighterSword` + WRAM pokes, then a **real**
dungeon load: `$10=0x06` with entrance `$010E=0x25`. Confirmed `$A0==0x28`.

Label: **isolated / state-load**. Indoor pin after room-load (controllable,
overlay is the swamp lobby, not a freeze).

## Glance (lobby leftover)

| Field | Value |
|-------|--------|
| Room | **0x28** (40) Swamp Palace lobby |
| Module | `$10==0x07` `$11==0` ctrl=1 |
| xy | **(4344, 1496)** south-east landing |
| Sword | `$F359==2` master |
| Keys | `$F36F==0` |
| Follower | **`$F3CC==0`** |
| Hookshot | `$F342==0` (prize not granted) |
| Indoors | true; `$0FFF==0`; `$00EE==0` |
| Entrance | `$010E==0x25` |

## Pokes

`env.unwrapped.data.memory.assign` at bank `$7E` (`0x7EF3xx`). Bare `0xF3xx`
raises `No known mapping`. Read-after-write via `get_ram()[wram_index(offset)]`.

| WRAM | Assign | Value | Note |
|------|--------|-------|------|
| `$F340` | `0x7EF340` | 0 | bow none |
| `$F342` | `0x7EF342` | 0 | hookshot none (prize) |
| `$F343` | `0x7EF343` | 10 | bombs |
| `$F34B` | `0x7EF34B` | 1 | hammer |
| `$F353` | `0x7EF353` | 1 | mirror |
| `$F354` | `0x7EF354` | 1 | power gloves |
| `$F355` | `0x7EF355` | 1 | boots |
| `$F357` | `0x7EF357` | 1 | moon pearl |
| `$F359` | `0x7EF359` | 2 | master sword |
| `$F35A` | `0x7EF35A` | 2 | fire shield |
| `$F36C`/`$F36D` | `0x7EF36C`/`D` | `0x18` | 3 hearts full |
| `$F36E` | `0x7EF36E` | `0x80` | magic |
| `$F37B` | `0x7EF37B` | 0 | magic consumption |
| `$F3C5` | `0x7EF3C5` | 3 | Agahnim 1 / DW open |
| `$F2BB` | `0x7EF2BB` | `0x20` | LW dam overlay (screen 0x3B bit5) |
| `$F2FB` | `0x7EF2FB` | `0x20` | DW swamp overlay (screen 0x7B bit5) |
| `$F051` | `0x7EF051` | 1 | room 0x28 `save_dung_info` bit8 |
| `$F216` | `0x7EF216` | `0x80` | dam interior room 267 switch-pulled |

Did not poke `$F3CC`. Did not poke `$F356` flippers (0). Did not grant hookshot.

## Water state

Overlay bits and room flag are set. South canal **grate is walkable** (Link
does not swim). North floor (tektites `0x81`/`0x9A`, key chest, north stairs)
is **visible** past a wall at **y=1352**. West lip (4160, 1357) UP does not
climb the cropped ladder. Flippers=0. Indoor water is **not** a proven
flood-cross this sitting.

## First hop

**`south_to_overworld` isolated ok** (`room_engine` `ok=True` 188f).

Hold DOWN from spawn → module `0x0F` spotlight → **overworld screen 0x7B**
**(1912, 3819)** `$10==0x09` `$11==0` ctrl=1 sword=2 `$F3CC==0` keys=`0xFF`
(OW blank). DW swamp facade. `maps/room_28.json` `landingXy` [1912, 3819].

**`north_to_0x38` not isolated.** First indoor door (small-key stairs, z3
Swamp Pot Row). Canal lip is a wall. Needs flood-cross + entrance key.

## Files

- `custom_integrations/Zelda3-Snes/SwampPalaceEntry.state`
- `docs/tasks/SwampPalaceEntry-residual.md` (this file)
- `scratch/probe_swamp_entry.py`
- `recordings/probe_swamp_entry/` (`leftover.json`, overlays, edge json)
- `maps/room_28.json`

## Non-claims

Did not STATUS. Did not treat the pin as power-on. Did not poke `$F3CC`.
Did not copy approach xy into Python. Did not overwrite
`verified_tip_run.json`. Did not touch Sanctuary maps/graph/`full_tip`.
Did not claim `rr-ccxt.20` / `.21`.

## Next

Flood-cross the 0x28 canal (cropped west ladder / flippers trial) → entrance
key on the north floor → isolate `north_to_0x38`.
