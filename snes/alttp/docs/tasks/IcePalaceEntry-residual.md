# Ice Palace entry residual (`rr-3yxm`)

Planner owns `docs/STATUS.md`. Isolated / state-load only. Not natural_entry,
not continuous, not power-on, not STATUS.

## Result: **ok** — pin + first indoor door (intra-room west)

`IcePalaceEntry.state` is a real room-load of Ice Palace lobby **`$A0==0x0E`**
(overlay ice tiles + Freezor, not a freeze). First hop is Ice Lobby WS: melt
Freezor with fire rod, LEFT at (7469, 364) into the west (jelly/Bari) quadrant.
Still **`$A0==0x0E`**. `room_engine` dest-check wants a room-id change, so the
hop is probe-isolated, not `run_room_edge` green.

## Pin method

Load `HyruleCastleGrounds` (OW 0x1B). Poke kit + `$F3C5=3`. Trigger a real
underworld load:

- `$10E=0x2D` (`which_entrance`, kEntranceNames Ice Palace / room 14)
- `$10C=0x06` `saved_module_for_menu` = PreDungeon
- `$10=0x06` `Module_PreDungeon` (not 0x2C — that is Lost Woods 0xE1)

Assign is `env.unwrapped.data.memory.assign(bus, "|u1", v)` with
`bus = 0x7E0000+off` for WRAM `>= $2000` (bare `0xF3xx` is "No known mapping").
Read-after-write via `get_ram()[wram_index(off)]`.

## Pokes (pin)

| WRAM | Want | After | Name |
|------|------|-------|------|
| `$F345` | 1 | 1 | fire rod |
| `$F354` | 2 | 2 | titans |
| `$F355` | 1 | 1 | boots |
| `$F356` | 1 | 1 | flippers |
| `$F357` | 1 | 1 | moon pearl |
| `$F359` | 2 | 2 | master sword |
| `$F35B` | 0 | 0 | green mail (prize not granted) |
| `$F36C` | `0x40` | 64 | max HP 8 hearts |
| `$F36D` | `0x40` | 64 | HP |
| `$F36E` | `0x80` | 128 | magic |
| `$F379` | `0x1E` | 30 | ability lift2 + swim + dash (boot was `0xF8`) |
| `$F3C5` | 3 | 3 | Aga 1 / DW open |

Did not poke `$F3CC`. Boot leftover on `HyruleCastleGrounds` is **5**
(`CanEnterWithTagalong` treats 5 as free). Reload of the pin still has 5.

Hop-only (not stored in the pin): `$0303=5` `$0307=1` (fire rod Y). Doorway
`$6C==1` at spawn blocks `LinkItem_Rod` until Link walks north off the stairs.

## Pin glance (`IcePalaceEntry`)

| Field | Value |
|-------|--------|
| Room | **`0x0E`** (7544, 472) |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 |
| Indoors | 1; `$6C` doorway=1 |
| Sword | `$F359==2` |
| Armor | `$F35B==0` |
| Gloves / pearl / rod / flippers / boots | 2 / 1 / 1 / 1 / 1 |
| Keys | `$F36F==0` |
| Follower | `$F3CC==5` (unpoked boot leftover) |
| Dungeon | `$040C==0x12` Ice Palace |
| Entrance | `$010E==0x2D` |
| HP / magic | 64 / 128 |
| Dark world | `$0FFF==0` `$F3CA==0` (warped from LW) |

Reload of `IcePalaceEntry.state` matches this glance.

## First hop leftover — west jelly quadrant (isolated)

Kill Freezor `0xA1` (7520, 296) with one fire-rod shot from (7520, 328). West
shutter opens. LEFT at **(7469, 364)** y-band ~360, 520f, intra-room scroll to
**0x0E (7232, 367)**. Overlay is the ice Bari room (chest + north key door).

| Field | Value |
|-------|--------|
| Room | **`0x0E`** (7232, 367) |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 |
| Sword | `$F359==2` |
| Armor | `$F35B==0` |
| Keys | `$F36F==0` |
| Follower | `$F3CC==5` |
| HP / magic | 56 / 112 (Bari nibble; one rod shot) |
| Facing | `$2F==4` LEFT |
| `$040C` | `0x12` |

Cardinals from spawn (Freezor still up): UP wedges (7544, 304); DOWN exits OW
(3256, 3531) `$10==0x09` screen 0x36 (LW because pin is not DW); LEFT/RIGHT stay
on the south stairs lip.

## Files

- `custom_integrations/Zelda3-Snes/IcePalaceEntry.state`
- `docs/tasks/IcePalaceEntry-residual.md` (this file)
- `scratch/probe_ice_entry.py`
- `recordings/probe_ice_entry/` (`leftover.json`, `pin.png`, `hop_west.png`, cardinals)
- `maps/room_0e.json` (confirmed `$A0==0x0E`)

## Non-claims

Did not STATUS-promote. Did not treat the pin as power-on. Did not poke
`$F3CC`. Did not copy approach xy into Python hops. Did not overwrite
`verified_tip_run.json`. Did not claim `rr-ccxt.20` / `.21`. Did not add an
`escape_graph` edge. `room_engine run` was not claimed: dest is the same room.

## Next

West-quad leftover is the jelly/Bari room in **0x0E** (7232, 367). Kill Bari
for the small key, then the north locked stairs (Ice Jelly Key Down). That is
the first *inter-room* hop. Fire rod must stay in kit; sword will not melt
Freezor if you replay from the pin.
