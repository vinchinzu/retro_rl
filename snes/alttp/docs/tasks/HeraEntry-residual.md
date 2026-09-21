# Tower of Hera entry pin — `rr-17wv`

Planner owns `docs/STATUS.md`. Dev pin, **isolated / state-load**, not continuous,
not power-on. Session: `.grok/skills/alttp-session/SKILL.md`.

**Bead:** `rr-17wv` (in_progress). Epic `rr-ldhk`.
**Save:** `HeraEntry.state`
**Confirmed indoor room:** **`$A0==0x77`** (hypothesis matched). Map:
`maps/room_77.json`.

## Pin method

1. Load `FighterSword` (secret entrance `0x55` (2803, 2680) `$10==0x07`/`$11==0`,
   sword already 1, `$F3C5==1`, `$F3CC==0`).
2. Poke Hera kit (read-after-write). Raw `assign(0xF3xx)` did **not** stick;
   **`assign(0x7EF3xx, "|u1", v)`** (`bank7e`) did.
3. Real dungeon load: `$010E=0x33`, `$10=0x06`, `$11=0` → settle in **0x77**
   (3832, 4032). Isolated indoor pin (layer `$00EE==1`).
4. **Real overworld door:** DOWN → Death Mountain screen **`0x03` (2288, 123)**
   `$10==0x09`, then UP through the Hera door → same 0x77 spawn, 2F green
   platform + blue pegs. Re-saved `HeraEntry.state` from that leftover.

Overworld climb from castle grounds was not finished this sitting.

## Pokes (every one)

| WRAM | before | want | after | method |
|------|--------|------|-------|--------|
| `$F340` bow | 0 | 2 | 2 | bank7e |
| `$F377` arrows | 0 | 30 | 30 | bank7e |
| `$F34A` lamp | 0 | 1 | 1 | bank7e |
| `$F34E` book | 0 | 1 | 1 | bank7e |
| `$F354` gloves | 0 | 1 | 1 | bank7e |
| `$F355` boots | 0 | 1 | 1 | bank7e |
| `$F357` moon pearl | 0 | 0 | 0 | already |
| `$F359` sword | 1 | 1 | 1 | already |
| `$F36C` max HP | 24 | 24 | 24 | already |
| `$F36D` HP | 24 | 24 | 24 | already |

Did **not** poke `$F3CC`, sword (already 1), keys, or `$F3C5`. `$F35A` shield=1
came with `FighterSword`. Prize / moon pearl absent.

## HeraEntry glance (leave proof)

| Field | Value |
|-------|--------|
| room `$A0` | **0x77** |
| module/sub `$10`/`$11` | **0x07 / 0** ctrl=1 |
| xy | **(3832, 4032)** |
| sword `$F359` | 1 |
| follower `$F3CC` | 0 |
| keys `$F36F` | 0 |
| layer `$00EE` | 1 |
| loadout | `$F340=2` `$F34A=1` `$F34E=1` `$F354=1` `$F355=1` `$F357=0` `$F377=30` `$F3C5=1` |

2F entrance: green south platform, blue crystal pegs, mini-moldorms `0x18`,
crystal `0x1E` at (3840, 3968). DOWN from spawn is Death Mountain (backtrack).

## First hop: `west_down_to_0x87` — isolated ok (probe)

Blue pegs cage spawn. Slash/spin the `0x1E` crystal on the green platform, then
west alcove → stair lip (3696, 3928) → approach **(3728, 3904)**. Trigger is
**UP** onto the west down-stairs (x=3704 UP wedges). Dest settle → **0x87 1F**.

`SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_hera_entry.py --hop`

| Field | 0x87 leftover |
|-------|----------------|
| room | **0x87** |
| module/sub | **0x07 / 0** ctrl=1 |
| xy | **(3721, 4424)** |
| sword / `$F3CC` / keys | 1 / 0 / 0 |
| layer `$00EE` | 0 |
| loadout | same as pin (no pearl) |
| HUD | `1F` |

`room_engine.py run room_77 --edge west_down_to_0x87 --state HeraEntry --no-clear`
**wedges** at west_alcove (3792, 3982): no crystal-slash primitive. Probe hop is
the isolated leftover. Overlay: `recordings/probe_hera_entry/hop_0x87.png`.

## Files

- `custom_integrations/Zelda3-Snes/HeraEntry.state`
- `maps/room_77.json`
- `scratch/probe_hera_entry.py`
- `recordings/probe_hera_entry/` (`pin.json`, `reenter.json`,
  `hop_west_down_to_0x87.json`, `HeraEntry.png`, `hop_0x87.png`)

## Non-claims

Did not STATUS-promote. Did not treat the pin as power-on. Did not poke
`$F3CC`. Did not copy approach xy into Python hops / graph. Did not overwrite
`verified_tip_run.json`. Did not claim `rr-ccxt.20` / `.21`. Did not add an
`escape_graph` edge.

## Next

Crystal-switch step so `room_engine` can go green on `west_down_to_0x87` from
pegs-up `HeraEntry`, or map `0x87` (1F small-key room) from the hop leftover.
