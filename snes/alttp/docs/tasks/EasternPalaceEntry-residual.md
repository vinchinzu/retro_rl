# Eastern Palace entry pin (`rr-y8xg`)

Planner owns `docs/STATUS.md`. Dev pin, not continuous, not power-on.
Session: `.grok/skills/alttp-session/SKILL.md`.
Handoff: `docs/tasks/HANDOFF-dungeon-starts.md`.

## Result: **ok** — `EasternPalaceEntry.state` is room **0xC9**

Hypothesis `$A0==0xC9` confirmed on the live pin. Isolated / state-load
only. Overworld walk from `HyruleCastleGrounds` did not finish (opening
rain soldiers on screen `0x2C`). Indoor pin is a real PreDungeon room-load
(module `$10=0x06`, entrance `$010E=0x08`), controllable, overlay is the
EP lobby (teal tiles, wall map), not a freeze.

## Pin glance (leave proof)

| Field | Value |
|-------|--------|
| Save | `EasternPalaceEntry.state` |
| Room | **0xC9** (`$A0`) |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 |
| xy | **(4856, 6616)** south alcove |
| Sword | `$F359==1` (fighter) |
| Shield | `$F35A==1` |
| Lamp | `$F34A==1` |
| Bow | **`$F340==0`** (prize not granted) |
| Keys | `$F36F==0` (dungeon HUD) |
| Follower | `$F3CC==5` (grounds leftover; **not poked**; no Zelda sprite) |
| Progress | `$F3C5==1` (poked; uncle/sword, not Zelda-rescued) |
| HP | `$F36D==24` / `$F36C==24` |
| Layer | `$00EE==0` |
| Dungeon | `$040C==4` |
| Entrance | `$010E==0x08` |

## Pokes

Source `HyruleCastleGrounds` (2386, 2528) screen `0x1B` `$10==0x09`.
`memory.assign` at WRAM `$F3xx` raised `IndexError: No known mapping`;
writes landed at bank `$7E` (`0x7EF359` etc.). Read-after-write via
`get_ram()[wram_index(offset)]`.

| WRAM | Item | Before | After | Via |
|------|------|--------|-------|-----|
| `$F359` | fighter sword | 0 | 1 | `memory.assign 0x7ef359` |
| `$F35A` | fighter shield | 0 | 1 | `memory.assign 0x7ef35a` |
| `$F34A` | lamp | 0 | 1 | `memory.assign 0x7ef34a` |
| `$F3C5` | progress (uncle/sword) | 0 | 1 | `memory.assign 0x7ef3c5` |

Did not poke `$F3CC`. Did not poke `$F340` (bow stayed 0). `$F3C5=1` was
an attempt to lift the opening-rain soldier lock; rain overlay still fell
on screen `0x2C` and the east soldier still talked. Walk abandoned;
PreDungeon 0x08 still used that progress byte.

Entrance load pokes (not inventory): `$010E=0x08` (u16), `$10=0x06`,
`$11=0`. Settle 122f → 0xC9 (4856, 6616) `$10==0x07`.

## First hop: `south_to_overworld` **isolated ok**

`room_engine.py run room_c9 --edge south_to_overworld --state EasternPalaceEntry`

| Field | Value |
|-------|--------|
| ok | `True` phase `via_south_to_overworld` 192f |
| From | 0xC9 (4856, 6616) |
| To | **outdoors screen 0x1E (3920, 1579)** `$10==0x09` `$11==0` ctrl=1 |
| Sword / lamp / bow | 1 / 1 / 0 |

Spawn is already on the south door. DOWN. Overlay is the EP building
stairs / owl statues.

North hall not isolated: UP through a 16px railing gap reaches
**(4864, 6448)** against the alcove north wall. Visible north door does
not room-change. Three popos (`0x4F`) sit at y≈6224 behind that wall.

## Files

- `custom_integrations/Zelda3-Snes/EasternPalaceEntry.state`
- `docs/tasks/EasternPalaceEntry-residual.md` (this file)
- `scratch/probe_ep_entry.py`
- `maps/room_c9.json`
- `recordings/probe_ep_entry/` (`enter.json`, `lobby.json`,
  `room_c9_south_to_overworld.json`, overlays)

## Non-claims

Did not STATUS-promote. Did not treat the pin as power-on. Did not poke
`$F3CC`. Did not copy approach xy into Python. Did not overwrite
`verified_tip_run.json`. Did not claim `rr-ccxt.20` / `.21`. Did not
touch Sanctuary maps or `escape_graph.py`. Did not grant bow.

## Next action

Isolate the north hall of 0xC9 (popos at y≈6224) and the first deeper
door (likely 0xB9 = 0xC9−0x10). Alcove nook (4864, 6448) is not that
door. Graph hop only if added, and only `verification=isolated`.
