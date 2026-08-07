# Level 8 — The Lion (route notes)

Status: **assisted-ow-bush** (candle-blocked for entry)

Planning sources:

- [Zelda Dungeon — Level 8: The Lion](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-8-the-lion/)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)
- Assist: [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md) (Survival infinite-life only)

Walkthrough claims that are emulator-verified are marked; source-only claims
stay labeled. **No Clean STATUS claims.**

## Overworld

### Door / bush screen (live)

| Claim | Source | Live |
|-------|--------|------|
| Bush pocket screen | walkthrough path decode | **`0x6D`** (assisted) |
| Mouth under lone bush | walkthrough | **not opened** — needs candle |
| Entry room id | — | **unknown** (no enter) |
| Triforce bit | walkthrough | `0x80` (source) |

Dead-end geometry (live, `OW_6D` / `Level8BushOW`):

- Enter **0x6D** only from **0x5D south @ x≈48**.
- Only open exit found: **UP @ x≈48 → 0x5D**.
- Raster + UP pushes without candle: **no** mode-16 mouth (expected).

Evidence: `recordings/l8_bush_recon.json`, `l8_bush_6d.png`,
`custom_integrations/.../Level8BushOW.state`, `OW_6D.state`.

### Verified walk (assisted) — start → bush 0x6D

Source “right×4 …” collides with **0x79 rocky pocket**. Live detour reuses the
L1 north lane and L2 door corridor + **0x5C maze**:

```text
0x77 E@y≈140 → 0x78 N@x≈48 → 0x68 N@x≈48 → 0x58
  E@y≈155 → 0x59 E → 0x5A E → 0x5B
  (climb to y≈88) E@y80–95 → 0x5C
  [maze: east @y≈88 → channel → east @y≈128] → 0x5D
  S@x≈48 → 0x6D  (Lion bush pocket)
```

Hop table + controller: `level8_overworld.LEVEL8_BUSH_HOPS`,
`OverworldToLevel8Controller` (maze waypoints =
`overworld.LEVEL2_5C_MAZE_WAYPOINTS`).

Probe:

```bash
PYTHONPATH=nes uv run python nes/zelda_i/scripts/probe_level8_entry.py \
  --infinite-life --save-state --tag l8_recon
```

Mid-path fixtures used during recon: `OW_5B`, `OW_5C`, `OW_5D`, `OW_6A`,
`OW_6B`, `OW_6C` (0x6C is a **side pocket** UP-only to 0x5C — **not** on the
bush route).

### Required item — Blue Candle

| Field | Value |
|-------|--------|
| RAM | `ADDR_CANDLE = 0x065B` (non-zero) |
| Source price | **60 rupees** (Blue Candle, merchant caves) |
| Also works | Red Candle from L7 (source) |
| Assist | **inventory poke forbidden** (`ASSIST_CONTRACT`) |

#### Candle blocker (this recon)

- No existing save state had candle (`PostSwordStart`, L1/L2 fixtures: all 0).
- Early shop path from IGN (“N of start then W”) → live **0x67** has **no
  west corridor** in band sweeps (dead-end north of start; same trap family as
  AGENTS.md).
- Full rupee farm + shop purchase not completed this session.
- **Blocker:** cannot open 0x6D bush without natural candle acquisition.
- Delivered: verified OW path to bush + `Level8BushOW.state` (no
  `Level8Entrance.state`).

When candle is available: select candle as B-item, stand near lone bush, B to
ignite, UP into mouth → expect `level==8`, mode 5; then
`save_state(..., "Level8Entrance")` and `--probe-rooms`.

## Interior (source → live)

| Room / feature | Enemies / notes | Live |
|----------------|-----------------|------|
| Entry | unknown | **not entered** |
| Manhandla early | source | no |
| Book of Magic (staircase) | `ADDR_BOOK=0x0661` | no |
| Darknut / keys / Compass / Map | source | no |
| Gohma side (arrow) | source | no |
| Magical Key (staircase) | `ADDR_MAGIC_KEY=0x0664` | no |
| Boss Gleeok 4-head | Heart → TF | no |

Items optional for credits (source). TF bit **`0x80`**.

## Boss / Triforce

- Boss: **Gleeok (4 heads)** — source only.
- `ADDR_TRIFORCE & 0x80` after shard 8 — source only.

## Checkpoints

| State | Provenance |
|-------|------------|
| `Level8BushOW` / `OW_6D` | Assisted settle on bush screen; **no candle** |
| `OW_5D` | Live south of maze; path parent of 0x6D |
| `OW_5C` | North corridor entry from 0x5B |
| `Level8Entrance` | **not created** (candle blocker) |

## Scaffold modules

| Path | Role |
|------|------|
| `level8_overworld.py` | Hops, maze, `OverworldToLevel8Controller` |
| `scripts/probe_level8_entry.py` | Assisted OW walk + optional burn/enter/room probe |
| `docs/LEVEL8_ROUTE.md` | This file |

## Evidence

- `recordings/l8_bush_recon.json` — 0x6D settle, exits, path string
- `recordings/l8_5d_DOWN_48_sc6d.png` — live 0x5D→0x6D
- `recordings/l8_bush_6d.png` — bush pocket
- `recordings/l8_free_explore.json` — early connectivity
- `recordings/l8_*_exits.json` — pocket maps (0x6B/0x6C/0x5C)

## Next

1. Natural or assisted **rupee farm → Blue Candle shop** (map exact OW screen +
   cave enter; no inventory poke).
2. Burn bush on 0x6D → `Level8Entrance` + entry room id + cardinal room probe.
3. Isolated pure room segments after graph exists.
4. Do not promote Clean until candle + path are natural-entry backed.
