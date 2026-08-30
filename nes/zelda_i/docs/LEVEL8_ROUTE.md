# Level 8 — The Lion (route notes)

Status: **PARTIAL** — assisted OW bush path green; shop OW path **green**
(rr-ccx). The cumulative Red-Candle route still needs the measured post-L7
leave, exact burn tile, and live entry. The 60R shop path is fallback-only.

Wave A now has a fail-closed cumulative seam in `level8/{entry,dungeon,hops,spine}.py`.
This is implementation structure, not new route evidence. The public chapter
targets are `level8-entry`, `level8-magic-key`, and `level8`; all three remain
red until their natural predecessor and exact live endpoints are supplied.

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
| Mouth under lone bush | walkthrough | **not opened** this bead |
| Entry room id | — | **unknown** (no enter) |
| Triforce bit | walkthrough | `0x80` (source) |

Dead-end geometry (live, `OW_6D` / `Level8BushOW`):

- Enter **0x6D** only from **0x5D south @ x≈48**.
- Walkable (assisted recon): left corridor **x≈32–56** + mid sand channel
  **y≈88–96** east to **x≈144** (see `recordings/l8_walkable.png`).
- Only open screen exit found without candle: **UP @ x≈48 → 0x5D**.
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

Hop table + controller: `level8.overworld.LEVEL8_BUSH_HOPS`,
`OverworldToLevel8Controller` (maze waypoints =
`overworld.graph.LEVEL2_5C_MAZE_WAYPOINTS`).

Isolated `probe_level8_entry.py` pruned. The durable runner does not attach
the new L8 seam to the shared continuous spine yet:

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1
```

Mid-path fixtures used during recon: `OW_5B`, `OW_5C`, `OW_5D`, `OW_6A`,
`OW_6B`, `OW_6C` (0x6C is a **side pocket** UP-only to 0x5C — **not** on the
bush route).

### Fallback item — Blue Candle

The cumulative mainline does **not** require this shop. It inherits the
naturally earned Red Candle (`ADDR_CANDLE == 2`) from Level 7. The Blue Candle
shop/farm is retained only as a disabled, route-ineligible fallback for an
unexpected Candle-0 handoff; normally that mismatch fails the L7 contract.

| Field | Value |
|-------|--------|
| RAM | `ADDR_CANDLE = 0x065B` (1=blue, 2=red) |
| B-item cursor | `ADDR_SELECTED_ITEM = 0x0656` — live candle pos **`4`** |
| Once-per-screen | `ADDR_CANDLE_USED = 0x0513` (0 ready / 1 used; leave screen to reset blue) |
| Source price | **60 rupees** (Blue Candle, merchant caves) |
| Also works | Red Candle from L7 (source; multi-use per screen) |
| Assist | **inventory poke forbidden** for Clean / published assisted STATUS |

#### Shop (live, rr-ccx) — first-quest **O-6** / screen **`0x5E`**

| Field | Live |
|-------|------|
| Map id | GameFAQs **O-6** regular shop (Shield 160 / Key 100 / Candle 60) |
| OW path | **verified assisted** `CANDLE_SHOP_HOPS` (see below) |
| Cave mouth | **UP @ x≈112** on 0x5E (mode 16→11); approach y≈77 |
| Cave fixture | **`CandleShop5E`** — mode **11**, screen **`0x5E`**, spawn xy≈(112,213) |
| Merchant | type **`0x78`** @ (120, 128) |
| Left item | type `0x40` @ x≈**72** — Magical Shield **160R** |
| Mid item | type `0x40` zone x≈**120** — Key **100R** (touch y≈149 drains R) |
| Right / candle | touch ≈(**152, 149**) — Blue Candle **60R** → `ADDR_CANDLE=1`, R−60 |
| Rupees in fixture | **0** (buy needs farm; poke-R recon only) |
| False lead | IGN “N of start then W” → live **0x67** no west corridor |

##### Verified walk (assisted) — start → shop cave 0x5E

Reuses L8 bush corridor through **0x5C maze** + **0x5D**, then **east**
(not south into bush):

```text
0x77 E@y≈140 → 0x78 N@x≈48 → 0x68 N@x≈48 → 0x58
  E@y≈155 → 0x59 E → 0x5A E → 0x5B
  (climb y≈88) E@y80–95 → 0x5C [maze] → 0x5D
  E@y≈130–150 → **0x5E**  (enter y≈141)
  cave: UP @ x≈112 → mode 11
```

Hop table + controller: `level8.overworld.CANDLE_SHOP_HOPS`,
`OverworldToCandleShopController` (door_x=`CANDLE_SHOP_CAVE_X`).

Isolated `probe_level8_entry.py` pruned. Shop hops live on
`OverworldToCandleShopController`. The durable runner does not attach the new
L8 seam to the shared continuous spine yet:

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1
```

##### Buy interaction (live geometry; needs ≥60R)

1. Settle cave bottom (dialog timer →0; ~120f idle OK).
2. **UP** the stairs until `link_y ≤ 150`.
3. **RIGHT** along y≈149 until `link_x ≥ 152` (do **not** linger on mid
   x≈120 if you only have 60–99R — Key costs 100 and will drain all R).
4. Touch right zone → `ADDR_CANDLE=1`; rupees drain async (−60).
5. Exit cave **DOWN** to OW 0x5E (post-buy residual for runner).

Constants: `CANDLE_BUY_X/Y`, `CANDLE_SHOP_PRICE`, pedestals
`CANDLE_SHOP_ITEM_*_X`.

##### Rupee farm sketch (residual — not automated)

| Idea | Notes |
|------|--------|
| Path Octoroks | Screens **0x59–0x5E** (type `0x03`); `RUPEE_FARM_SCREENS_SKETCH` |
| Pre-shop farm | Clear 0x5A/0x5B/0x5E before cave enter until `rupees ≥ 60` |
| Fixture | `CandleShop5E` has **0R** — farm **before** or reload after farm state |
| Clean rule | No RAM poke for rupees/candle on published tracks |

#### Candle recon (`rr-q8a` + shop path `rr-ccx`)

- Inventory poke (`poke_candle_for_recon`: candle + selected=4 + clear used)
  shows candle on HUD B-slot (`recordings/l8_pos4.png`).
- Pressing B with candle selected sets **`0x0513=1`** (engine accepts use).
- **Level8Entrance not created**: fire→stairs on 0x6D not confirmed after
  dense walkable burns (corridor + left column + enemy clear). Need further
  bush-tile targeting or natural candle path.
- Default burn aim in controller: **(136, 93)** east channel (source “lone bush
  blocking pathway”).
- Shop OW + cave path **assisted green** (`recordings/l8_shop_path.json`);
  natural 60R + buy still residual.

**Historical recon blocker:** natural candle (farm→buy) + verified mouth open
on 0x6D → `Level8Entrance.state` + entry room id.

For the cumulative route, replace “natural candle” above with the measured
post-L7 Red Candle handoff. The old start-based controller remains recon-only.
Its historical burn-budget-on-`0x6D` result is not entry success; the canonical
`BurnLevel8BushController` fails whenever its budget expires without live L8
play, even if Link is still controllable on `0x6D`.

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
| `OW_5D` | Live south of maze; path parent of 0x6D / west of shop |
| `OW_5C` | North corridor entry from 0x5B |
| `BFS_5E` / `OW_5E` | Live OW on shop screen (west edge entry y≈141) |
| `CandleShop5E` | Cave on **0x5E** mode 11 (0R; buy residual) |
| `CandleOwned` | **not created** (natural buy residual) |
| `Level8Entrance` | **not created** (candle→burn residual) |

## Scaffold modules

| Path | Role |
|------|------|
| `level8/overworld.py` | Bush + **shop** hops, burn controller, `OverworldToCandleShopController` |
| `level8/entry.py` | Canonical measured post-L7 approach, natural pause selection, fail-closed Red Candle burn |
| `level8/dungeon.py` | Route-ineligible chapter/topology specs and exact stop predicates; no invented room IDs |
| `level8/hops.py` | Fresh chapter/controller factories and three `SpineHop` rows |
| `level8/spine.py` | `L8_THROUGH`, `L8_STOPS`, `continue_level8_spine` |
| Isolated `probe_level8_entry.py` | pruned; Composer `scripts/run_survival_spine.py` |
| `docs/LEVEL8_ROUTE.md` | This file |

### Wave A handoff contract

The integrator must provide all of the following before the default seam can
move: settled post-L7 OW screen/x/y, exact keys/bombs/rupees/hearts/B-slot,
Whistle/Food/Rod/Bow/arrows, Candle 2, full health, TF exactly `0x7F`, and a
live-derived hop table ending at `0x6D`. The bush burn additionally needs an
observed Link tile, facing, push direction, and natural `ADDR_CANDLE_USED`
transition. The entry room, Magical Key room, boss room, Triforce room, and
post-fanfare leave remain `None`; walkthrough grid positions are hypotheses.

Fixture work may fill controller behavior while keeping
`route_eligible=false`. Only cumulative recomposition from the natural L7
predecessor may promote the handoff/topology/endpoint contracts.

## Evidence

- `recordings/l8_bush_recon.json` — 0x6D settle, exits, path string
- `recordings/l8_5d_DOWN_48_sc6d.png` — live 0x5D→0x6D
- `recordings/l8_bush_6d.png` — bush pocket
- `recordings/l8_walkable.png` — pink walkable samples on 0x6D
- `recordings/l8_pos4.png` — HUD candle after recon poke (selected=4)
- `recordings/l8_candle_burn_enter.json` — burn recon summary (entrance false)
- `recordings/l8_candle_shop66.json` — earlier 0x66 cave probes (not shop)
- `recordings/l8_shop_path.json` — assisted start→0x5E hop success
- `recordings/l8_5e_cave_x112.png` / `l8_5e_cave_enter.png` — cave mouth
- `recordings/l8_buy_ok_repro.png` — recon buy @ (152,149) with poked 80R
- `recordings/l8_*_exits.json` — pocket maps (0x6B/0x6C/0x5C)

## Next

1. Receive the measured natural post-L7 leave/inventory and derive its hop
   table through live `0x5C` geometry to **0x6D**.
2. From Red-Candle state: burn the exact live-confirmed bush tile on **0x6D** →
   `Level8Entrance` + entry room id.
3. Decode the interior graph offline, then live-confirm entry → Magical Key →
   boss/shard rooms without promoting source room IDs.
4. Keep the 60R Blue Candle farm/shop as fallback-only, outside `L8_THROUGH`.
5. Do not promote Clean; Wave A fixture evidence remains route-ineligible.
