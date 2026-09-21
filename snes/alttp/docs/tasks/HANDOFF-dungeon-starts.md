# Living residuals — dungeon-start pins

Planner owns `docs/STATUS.md`. Each claimed bead overwrites **its** residual.
Session: `.grok/skills/alttp-session/SKILL.md`.

Epic `rr-ldhk`. These are **development pins**, not continuous, not power-on.
Sanctuary spine `rr-ccxt` stays separate; do not claim `rr-ccxt.20` / `.21`.
Do not overwrite `recordings/verified_tip_run.json`. Do not STATUS.

| Bead | Dungeon | Save | Residual | Owns (only these) |
|------|---------|------|----------|-------------------|
| `rr-y8xg` | Eastern Palace | `EasternPalaceEntry` | `EasternPalaceEntry-residual.md` | `maps/room_c9.json` (confirm hex), `scratch/probe_ep_entry.py`, `recordings/probe_ep_entry/` |
| `rr-alwe` | Desert Palace | `DesertPalaceEntry` | `DesertPalaceEntry-residual.md` | `maps/room_84.json` (confirm), `scratch/probe_dp_entry.py`, `recordings/probe_dp_entry/` |
| `rr-17wv` | Tower of Hera | `HeraEntry` | `HeraEntry-residual.md` | `maps/room_77.json` (confirm), `scratch/probe_hera_entry.py`, `recordings/probe_hera_entry/` |
| `rr-9zkv` | Palace of Darkness | `PalaceOfDarknessEntry` | `PalaceOfDarknessEntry-residual.md` | `maps/room_4a.json` (confirm), `scratch/probe_pod_entry.py`, `recordings/probe_pod_entry/` |
| `rr-q41r` | Swamp Palace | `SwampPalaceEntry` | `SwampPalaceEntry-residual.md` | `maps/room_28.json` (confirm), `scratch/probe_swamp_entry.py`, `recordings/probe_swamp_entry/` |
| `rr-lvnu` | Skull Woods | `SkullWoodsEntry` | `SkullWoodsEntry-residual.md` | `maps/room_58.json` (confirm), `scratch/probe_skull_entry.py`, `recordings/probe_skull_entry/` |
| `rr-yk6q` | Thieves' Town | `ThievesTownEntry` | `ThievesTownEntry-residual.md` | `maps/room_db.json` (confirm), `scratch/probe_thieves_entry.py`, `recordings/probe_thieves_entry/` |
| `rr-3yxm` | Ice Palace | `IcePalaceEntry` | `IcePalaceEntry-residual.md` | `maps/room_0e.json` (confirm), `scratch/probe_ice_entry.py`, `recordings/probe_ice_entry/` |

Room hexes are **hypotheses**. Confirm `$A0` on the live pin before naming the map file. If the real entry room differs, use that hex and say so in the residual.

## Sitting

1. `bd update <your-id> --status in_progress` — claim **exactly that bead**.
2. Overwrite your residual under `docs/tasks/`.
3. Build `{Save}.state` in `custom_integrations/Zelda3-Snes/`.
4. Leave proof: RAM glance (room, module/sub, xy, sword, keys, loadout bytes). No MP4.

## Pin method (dev)

Load `FighterSword` or `HyruleCastleGrounds`. Poke **only** the inventory / progress bytes this dungeon needs. Walk into the overworld door when that is cheap (Eastern Palace is three screens east of castle `0x1B`). Dark-world dungeons may need moon pearl + mirror + `$F3C5` progress. If the overworld walk does not finish this sitting, an indoor pin is allowed **only** after a real room-load (controllable, overlay looks like the dungeon, not a freeze). Label the pin `isolated` / state-load.

Do not grant the dungeon prize. Entrance equipment is the RTA-typical kit **before** that dungeon.

Write pokes via `env.unwrapped.data.memory.assign(offset, "|u1", value)` (Zelda I `mem_write` shape). WRAM `$F3xx` uses offset `0xF3xx`, not the `get_ram()` index. Confirm with a read-after-write. Document every poke in the residual.

| WRAM | Item |
|------|------|
| `$F340` | bow (`0` none, `2` bow+arrows) |
| `$F342` | hookshot (`1`) |
| `$F343` | bombs (count) |
| `$F345` | fire rod (`1`) |
| `$F34A` | lamp (`1`) |
| `$F34B` | hammer (`1`) |
| `$F34E` | book (`1`) |
| `$F353` | mirror (`1`) |
| `$F354` | gloves (`1` power, `2` titans) |
| `$F355` | boots (`1`) |
| `$F356` | flippers (`1`) |
| `$F357` | moon pearl (`1`) |
| `$F359` | sword (`1` fighter, `2` master) |
| `$F35A` | shield (`1` fighter, `2` fire) |
| `$F36C`/`$F36D` | max/current HP (`0x18`/`0x18` = 3 containers full is a minimum; more is fine) |
| `$F3C5` | progress (`2` Zelda rescued, `3` Agahnim 1 / DW open) |

## Loadouts (prize not granted)

| Dungeon | Sword | Must have | Must NOT have |
|---------|-------|-----------|---------------|
| Eastern Palace | fighter | lamp, shield | bow |
| Desert Palace | fighter | lamp, bow, book, boots | gloves |
| Tower of Hera | fighter | lamp, bow, book, boots, power gloves | moon pearl |
| Palace of Darkness | master | moon pearl, gloves, boots, bow, lamp; DW + `$F3C5>=3` | hammer |
| Swamp Palace | master | moon pearl, hammer, mirror, boots, gloves | hookshot |
| Skull Woods | master | moon pearl, hammer, hookshot, boots, gloves | fire rod |
| Thieves' Town | master | moon pearl, hammer, hookshot, fire rod, boots, gloves | titans |
| Ice Palace | master | moon pearl, titans, fire rod, flippers, boots | blue mail (`$F35B==0`) |

## After the pin

Copy `maps/room_61.json` schema. Measure spawn xy, one door approach, overlay PNG. Isolated `room_engine.py run` for the first door. Graph edge only if you add one, and only `verification=isolated`.

## Forbidden

`docs/STATUS.md`, `opening_route/escape_graph.py`, `opening_route/full_tip.py`, `recordings/verified_tip_run.json`, Sanctuary maps `room_{50,01,51,52,55,60,61,62,70,71,72,80,81,82}.json`, `ram.py`, `AGENTS.md`, another worker's residual/map/scratch/state.

## Non-claims (every residual)

Did not STATUS-promote. Did not treat the pin as power-on. Did not poke `$F3CC`. Did not copy approach xy into Python. Did not overwrite `verified_tip_run.json`.
