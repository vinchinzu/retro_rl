# Level 7 — The Demon (route notes)

**Status:** Phase 1 (2026-09-02). The **L6 leave is measured and verified**:
`--through level6-exit` 2/2 → OW `0x22` `(112,125)` TF `0x3F`, keys 2 bombs 8
rupees 42, `selected_item=2` (arrows), Whistle 1, Food 0, Candle 0, 8 HC full.
`MEASURED_POST_L6_EXIT.verified=True`. On a continuous power-on
`--through level7-entry` the post-L6 controller now **walks the bait prefix
green** (`0x22→0x32→0x33→0x23→0x24→0x25`, 1577f, `writes=0`) and fails closed
at `level7_bait_purchase` → `bait_shop_geometry_unobserved`. The Survival
rupee top-up (42→60R, `SPINE_L7_RUPEE_RETOPUP`) fires before that stage.
Pond `0x42`, drain, entry room, the bait shop `0x34` geometry, and all
dungeon rooms remain **hypothesis**. Cumulative spine chapters stay
fail-closed (`route_eligible=false`). There is no pond checkpoint, L7 entry
room, or Clean segment. **Whistle** from Level 5 gates pond drain;
**Bait/Food** is a natural 60R shop buy (never `ADDR_FOOD` write; the rupee
**count** top-up to the shop cost is the disclosed Survival shortcut).

**Beads:** `rr-7vc` (closed planning), `rr-dnp` (live pond approach), `rr-8t4.1`
(Wave A recon). Do not STATUS-promote.

Planning sources:

- [Zelda Dungeon — Level 7: The Demon](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-7-the-demon/)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)
- RAM: `ADDR_WHISTLE` (`0x065C`), `ADDR_FOOD` (`0x065D`), `ADDR_CANDLE`,
  `ADDR_TRIFORCE`

All screen/room ids are **source-hypothesized** unless marked **(live)**.

---

## Gates / required capabilities

| Cap | RAM | Source role |
|-----|-----|-------------|
| **Whistle / Recorder** | `ADDR_WHISTLE` (`0x065C`) ≠ 0 | Drain pond → entrance; Digdogger shrink |
| **Bait / Food** | `ADDR_FOOD` (`0x065D`) ≠ 0 | Hungry Goriya gate (mid-dungeon) |
| Bombs | `ADDR_BOMBS` | Many secret walls; bomb-skip locked doors |
| Sword | `ADDR_SWORD` | Combat (Magical Sword ideal later) |
| Keys | `ADDR_KEYS` | Only **4 keys** for **5 locks** in dungeon (source) — bomb-skip or pre-carry |
| **Red Candle** (dungeon item) | `ADDR_CANDLE` value 2 (source) | Multi-use flame per screen |
| Triforce shard 7 | `ADDR_TRIFORCE & 0x40` | Clear stop |

**Predecessors:** L5 Whistle; buy Bait on OW before or after pond approach.
Planning-only may document dev pokes of whistle/food — **never** Clean STATUS.

---

## Overworld

### Bait shop (source)

From start: **up, left×3, up×3** → Armos field; tap **top-row middle** Armos
for staircase → special shop → **Bait 60R**. `U L×3 U×3` from start `0x77`
walks `0x67 → 0x66 → 0x65 → 0x64 → 0x54 → 0x44 → 0x34` — i.e. **the shop
screen `0x34` is entered walking NORTH from `0x44`**, and the staircase is
revealed by the top-row-middle Armos *on `0x34` itself*.

External overworld grid (nesmaps Q1 / GameFAQs; cols A–P, rows 1–8):
L6 = **C3 = `0x22`** ✓live; Power Bracelet Armos = **E3 = `0x24`**
(tap top-right statue); cheap Bait shop = **E4 = `0x34`**, directly south of
`0x24`. Pond / L7 entrance from shop: `D×2 L×2 U` = `0x34→0x44→0x54→0x53→
0x52→0x42`.

| Landmark | Source hops from start | Hypothesized id | Live? |
|----------|------------------------|-----------------|-------|
| Bait Armos / special shop | U L×3 U×3 (arrive `0x34` from `0x44`, north) | **`0x34`** | no |

**2026-09-02 Phase 1 recon — the fixture prefix over-shot into a dead pocket.**
Screens `0x33` / `0x23` / `0x24` / `0x25` are a mountain-bounded desert pocket:
every south edge tested is solid mountain and there is **no walkable
`{0x24,0x33} → row-4` transition**. The shop `0x34` is only reachable from the
south (`0x44 ↑ 0x34`). Concretely established this sitting
(`scratch/sweep_25_armos.py`, `scratch/probe_24_to_shop.py`,
`scratch/probe_33_south_to_shop.py`, `--from-state Level6ExitOverworld`):

- `0x25` south edge = solid mountain (x=128 walled). Its **only** non-backtrack
  exit is a *hidden* north passage at **x≈208 → `0x15`** (mountain path, wrong
  way), revealed after bumping the rightmost `0x25` Armos. The prior handoff's
  "`0x25 → DOWN → 0x35 → 0x34`" plan is **impossible** — retire it.
- `0x24` south edge = solid mountain at every x in {40,72,104,128,152,184,216};
  the 10-Armos sweep revealed **no** staircase toward `0x34`. (`0x24`'s Armos
  staircase, top-right statue, is the Power *Bracelet*, not the shop.)
- `0x33` south edge = solid mountain at x∈{120,160,208}; `0x33→0x34` RIGHT
  still walled. The `0x33` Armos block y=141 horizontal travel — detour above
  or below the statue rows.
- **Recommended next route (untested):** from L6 `0x22`, `0x22 ↓ 0x32 ↓ 0x42`
  (pond screen) `→ 0x43 → 0x44 ↑ 0x34`. This reaches the shop from the source
  direction *and* passes through the pond `0x42` for free. The `0x32↔0x42`
  south link and `0x42→0x43→0x44` chain are unverified. The whole
  `0x23/0x24/0x25` detour is then unnecessary — do not keep extending it.

**Measured post-L6 leave (2026-09-02):** `--through level6-exit` **1/1**
(`l6_exit_ow.json`) — the shard fanfare auto-warps Link to OW **`0x22`
`(112,125)`** mode 5, TF `0x3F`, keys 2 bombs 8 rupees 42 Rod 1 Bow 1
arrows 1, 8 HC full. Screen `0x22` **confirms** the bait/pond route
assumption. The old poke fixture `Level6ExitOverworld` (play `0x22`
`(120,221)` TF `0x3F` Food=0 80R, White Sword, blue candle, keys 3) was a
*hand-built recon poke*, not a fanfare — its `(120,221)` position and
White-Sword/candle/rupee/keys loadout are superseded by the real
`(112,125)` return with the spine's actual inventory. Packet:
`MEASURED_POST_L6_EXIT` in `level7/entry.py` (`verified=False` until the L7
owner attaches). Controller: `OverworldToBaitShopController`.

Fixture-live prefix (`l7_bait_25`, Survival, `route_eligible=false`):

```text
0x22 → 0x32 → 0x33 → 0x23 → 0x24 → 0x25
```

- **Real fanfare return is `0x22` `(112,125)`** (the mouth tile, upper-centre).
  First move is DOWN toward `0x32` — away from the mouth, so re-entry is not a
  risk. `(112,125)` is only a "mode 16 → L6" trap on a fresh UP into it, not on
  emerging. The old fixture's `(120,221)` south-edge start is superseded but the
  hop chain below is unchanged (straight DOWN the x≈112 corridor to `0x32`).
- `0x32` north `(120,61)`: LEFT to x=112 then DOWN. **Dead:** `off_north`
  DOWN at x=120 (`l7_bait_from_l6`, tile 216).
- `0x32→0x33` RIGHT at y=141 (live L6 reverse).
- `0x33` `(208,141)` UP at x=208 to `0x23`. **Dead:** RIGHT at y=141 into
  `0x34` (east mountain, `l7_bait_32ax`).
- `0x23→0x24` RIGHT at y=141 (live L6 reverse).
- **0x24→0x25 RIGHT @ y=141 (fixture-live, `l7_bait_25` 1/1 1,438f).** L6
  reverse of `0x24→0x23` LEFT @ y=141. SE leftover `(208,189)` UPs to the
  band then RIGHT. South wall is mountain.
- **Dead on 0x24:** south edge is solid mountain at every x tried
  ({16,40,72,104,128,152,160,184,208,216}). The `0x24` 10-Armos sweep reveals
  no shop staircase (top-right statue = Power Bracelet). Occupancy xmin=14 west
  pocket `(0,141)`; occupancy SW box `(25,181)`.
- **`0x25` is a dead pocket** — see the Phase 1 recon block above. Its south is
  walled; its only hidden exit (north x≈208 → `0x15`) goes the wrong way. The
  `0x22→…→0x25` prefix is fixture-live geometry but a dead end for Bait.
- **Next sitting:** abandon the `0x23/0x24/0x25` detour. Recon
  `0x22 ↓ 0x32 ↓ 0x42 → 0x43 → 0x44 ↑ 0x34` instead (shop from the south, per
  source; passes the pond `0x42`).

Armos tap (top-row middle on `0x34`), 60R Food, and pond `0x42` are not this
sitting. Food stayed 0; rupees 80→81 on the walk. Zero deaths;
`progression_writes=capacity_writes=0`.

### Whistle pond (source)

From bait shop screen: **down×2, left×2, up** → pond (looks like fairy pond
but is not). Equip Whistle on B, use once → water drains → stairs into L7.

| Landmark | Source hops from shop `0x34` | Hypothesized id | Live? |
|----------|------------------------------|-----------------|-------|
| L7 pond / entrance | D×2 L×2 U | **`0x42`** | no |

The executable pond controller skips the unverified shop detour. Geometry-only
(no Whistle required). Its live prefix is:

```text
0x77→0x78→0x68→0x58→0x57→0x56→0x55→0x65→0x64→0x54→0x53→0x52
```

Live geometry (Survival, `PostSwordStart`, `--no-video`):

- `0x65→0x64` arrives on the east ledge around `(232,109)`; go DOWN to the
  open band, LEFT to the north gap around `x≈48`, then UP to `0x54`.
- `0x54→0x53` is LEFT around `y≈141`.
- **Dead belief:** on `0x53` at `(224,173)`, `hop10_ay` DOWN toward `y≈189` is
  blocked (`l7_dnp_pond_assisted_v9`).
- **0x53 micro (fixture-live):** LEFT inland from the east edge (`x>192`)
  before descending, then LEFT at/below `y≈189` into `0x52`. Occupancy miss →
  block cell → replan; no path → stand.
- **Next leftover:** play `0x52` `(112,181)`, hop `0x42` UP. `hop11_ax` then
  `unstick_wait` — north gap on `0x52` is not `x=112`. Pond `0x42` is still
  unobserved.

Evidence: `recordings/l7_dnp_pond_53.json` and `_final.png` (this sitting);
prior miss `recordings/l7_dnp_pond_assisted_v9.json`. Zero deaths;
`progression_writes=capacity_writes=0`; `success=false`; `route_eligible=false`.

**Controllers:** the spine uses `level7.entry.PostLevel6OverworldController`
(shared `OverworldHandoff` gate + fixture-live `POST_L6_TO_BAIT_HOPS`), which
refuses every frame until the handoff verifies. `OverworldToBaitShopController`
(ungated post-L6 fixture) and `OverworldToLevel7PondController` (start-based
pond recon) in `level7.overworld` stay recon-only. Isolated
`probe_level7_entry.py` pruned. Whistle is a pond-**drain** gate, not a
geometry-walk gate.

### Live recon goals

1. Reach pond screen without Whistle (map pond geometry only).
2. Save `OW_L7Pond` if pond screen confirmed.
3. With real Whistle: drain, enter, confirm `level == 7`, entry room.
4. Save `Level7Entrance` + `recordings/l7_*_recon.json`.

**Do not** poke Whistle / Food for Clean claims.

---

## Interior (source speed route)

RAM room IDs **unknown**. Offline graph: `level7/graph.py` (source ids
`0x7xx`, every `ram_id=None`, `evidence=hypothesis`). Prefer bomb walls over
the fifth lock; Hungry Goriya is a Food gate; Red Candle is `ADDR_CANDLE`
1→2 naturally. Key/bomb ledger is in `LEVEL7_KEY_BOMB_LEDGER`.

Key themes: bomb walls, key shortage, Digdogger re-spawns, hungry Goriya,
“tip of the nose” staircase, Red Candle, forced Digdogger before boss,
Aquamentus.

| Step | Action (source) | Notes |
|------|-----------------|-------|
| Entry | RIGHT | into dungeon body |
| N path | Moldorms | bombs reward optional |
| R | Goriya clear → Old Man | “THERE’S A SECRET IN THE TIP OF THE NOSE” |
| R | Digdogger | Whistle → multi-mini; optional skip |
| R | Stalfos **key** | then backtrack left ×4 |
| Bomb walls | left / up secrets | Goriya bomb drops |
| S / keys | more keys | Dodongo room skippable |
| Bomb capacity #2 | 16 bombs | source 100R room (source) |
| Compass | Stalfos drop | side path |
| Hungry Goriya | equip **Bait**, drop | **hard gate** without Food |
| Map room | center Map | bomb N “missing” map room instead of locked E |
| Bomb chain | rupees / Goriya | keys as needed |
| Tip-of-nose room | Wallmasters + push mid-right block | stairs after Map context |
| Stairs path | bomb R → boss | |
| Force Digdogger | whistle + bomb/sword | door N opens only after kill (source) |
| Boss | **Aquamentus** | same as L1; Magical Sword trivial |
| E of boss | center | **Triforce shard 7** |

**Key item:** Red Candle (`ADDR_CANDLE`; blue→red upgrade).
**Boss:** Aquamentus (object type may match L1 live type — verify).
**Triforce bit:** `0x40`.

### Policy notes (planning)

- Whistle on B for pond + every Digdogger.
- Bait on B once at hungry Goriya (consumes Food).
- Key economy: prefer bomb walls over fifth lock.
- Wallmasters: grab → entrance warp; clear before block push.
- Boss: A-spam head; no special item beyond sword.

---

## Cumulative chapter seam

**Wired into the main spine (2026-09-02).** `zelda_i.spine.survival` now
imports `continue_level7_spine`; `L7_THROUGH` is appended to `SPINE_THROUGH`.
For an L7 `--through` target the L6 suffix is driven to `level6-exit` (the
measured OW `0x22` `(112,125)` return) and L7 continues from screen `0x22`
with `MEASURED_POST_L6_EXIT` as the handoff.

`MEASURED_POST_L6_EXIT` is the **shared `zelda_i.overworld.stitch.OverworldHandoff`**
packet (no L7-local `PostLevel6Handoff` type any more), **`verified=True`**
(2026-09-02: `--through level6-exit` measured 2/2 with `selected_item=2`
captured; `route_eligible` still `False`). The spine controller
(`PostLevel6OverworldController`) carries the fixture-live bait prefix
`POST_L6_TO_BAIT_HOPS` (`0x22→0x32→0x33→0x23→0x24→0x25`) as its default hops.
The measured leave stands on the `0x22` mouth tile, so the controller's
re-entry refusal only arms after Link steps off it (`_left_mouth` latch).
On `--through level7-entry` the controller now **walks `0x22→0x25` green**
(`level7_post_l6_overworld`, 1577f) and then fails closed at
`level7_bait_purchase` → `bait_shop_geometry_unobserved` (no observed shop
`0x34` geometry; the pond `0x42` route past `0x25` is also unobserved). The
`0x77`-start pond walk and the ungated `OverworldToBaitShopController` stay
recon-only.

`level7/spine.py` exposes only the three plan-level targets:

| `--through` | Internal chapter stages | Current evidence |
|-------------|-------------------------|------------------|
| `level7-entry` | post-L6 overworld; natural Bait purchase; pond drain/entry | wired; `level7_post_l6_overworld` **green**; fails closed at `level7_bait_purchase` (`bait_shop_geometry_unobserved`) |
| `level7-red-candle` | entry to Hungry Goriya; tip-of-nose stairs; Red Candle pickup | wired; fails closed |
| `level7` | forced Digdogger; Aquamentus/heart; shard/settled leave | wired; fails closed |

Factories in `level7/hops.py` always return fresh controllers. **The L6 leave
is measured and `verified=True`** — `MEASURED_POST_L6_EXIT`, OW `0x22`
`(112,125)` TF `0x3F`, `selected_item=2` (`--through level6-exit` 2/2). The
post-L6 controller walks `0x22→0x25` green; **Bait purchase** still refuses
until shop geometry exists (`shop_cave_xy=None` → `bait_shop_geometry_unobserved`).
The 42R→60R Bait gap is closed by `SPINE_L7_RUPEE_RETOPUP` (documented
Survival rupee-count top-up, mirroring the bomb/key top-ups; a natural OW
farm is a separate bead). Hungry Goriya fails closed without Food. Red
Candle fails closed until
the item room is observed and `ADDR_CANDLE` becomes 2 naturally. The
start-based pond walk is recon-only. Source room ids cannot become executable
stop predicates. The live L6 residual play `0x09` `(56,109)` TF `0x1F`
Rod=0 is **not** an L7 start.

Promote a blocker only with a chapter handoff containing the exact natural
predecessor, live room/screen transitions, item and key/bomb deltas, and the
one-frame policy that replaces it. Keep room specs and exact endpoint
predicates in `level7/dungeon.py`; keep movement in `level7/path.py` or a
cohesive purpose-named path module.

The stop predicates additionally require:

- `level7-entry`: exact observed entry room, TF `0x3F`, Whistle and Food owned;
- `level7-red-candle`: exact observed room, TF `0x3F`, Candle exactly `2`,
  Whistle retained, Food consumed;
- `level7`: exact observed settled leave, TF `0x7F`, Candle `2`, Whistle
  retained, one natural heart-container increase, and full hearts.

The stop specs intentionally carry no room ids yet, use
`evidence=hypothesis`, and set `route_eligible=false`, so all three fail
closed. A room id alone is insufficient: public success accepts only an exact
stop promoted to `natural-segment` or `spine-green` evidence.

## Boss / Triforce evidence still needed

```text
level7_boss_cleared  — TBD: Aquamentus dead + HC
level7_complete      — ADDR_TRIFORCE & 0x40
```

The public seam uses the strict stop contracts in `level7/dungeon.py`; there
is no loose bit-only or unknown-room dungeon predicate in `overworld.py`.

---

## Checkpoints (planned names)

| State | When |
|-------|------|
| `Level6ExitOverworld` | Save-state name still used by `scratch/run_bait_from_l6_exit.py` recon. The old `(120,221)` / 80R poke loadout and `HYPOTHESIZED_POST_L6_EXIT` packet were **deleted** — superseded by the measured `--through level6-exit` return `0x22` `(112,125)` (`MEASURED_POST_L6_EXIT`, an `OverworldHandoff`). |
| `OW_L7Pond` | Pond screen mapped (Whistle optional) |
| `OW_L7BaitShop` | Armos shop screen |
| `Level7Entrance` | `level==7`, play, entry room |
| `Level7RedCandle` | after Red Candle |
| `Level7Complete` | `triforce & 0x40` |

---

## Scaffold / probe

Isolated `probe_level7_entry.py` pruned. The L7 seam is now attached in
`zelda_i.spine.survival` (`continue_level7_spine` after the L6 suffix):

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level7-entry --no-video --trials 1
```

runs the continuous power-on tape through the measured L6 fanfare exit, walks
`0x22→0x25` green, and stops fail-closed at `level7_bait_purchase`
(`bait_shop_geometry_unobserved`).

Modules: `zelda_i/level7/{dungeon,graph,entry,path,hops,overworld,spine}.py`.
The historical pond walk remains `level7.overworld.OverworldToLevel7PondController`;
it begins at the start screen and therefore remains recon-only. Whistle is
required to drain the pond. To make `level7-entry` green from here: live-recon
`0x25 (0,141) → bait shop 0x34`, fill `BaitPurchasePlan.shop_cave_xy` /
`shop_geometry_verified` and the shop buy policy, then supply observed OW hops
`0x34 → pond 0x42` plus the Whistle drain and the entry room.

---

## Evidence

- **Hypothesis:** first-quest room/door/stair graph (`level7/graph.py`); Bait
  shop `0x34`; pond `0x42`; all stop room ids.
- **Measured + verified:** post-L6 fanfare leave OW `0x22` `(112,125)` TF
  `0x3F`, keys 2 bombs 8 rupees 42, `selected_item=2`, Whistle 1 Food 0
  Candle 0, 8 HC full (`recordings/l6_exit_ow.json` +
  `recordings/l7p1_l6exit.json`, `--through level6-exit` 2/2 byte-identical).
  `MEASURED_POST_L6_EXIT.verified=True`.
- **Spine-green (continuous power-on):** `level7_post_l6_overworld` walks the
  bait prefix `0x22→0x32→0x33→0x23→0x24→0x25` in 1577f, `writes=0`, from the
  measured leave (`recordings/l7p1_entry_v2.json`, `--through level7-entry`).
  `route_eligible=false` (endpoint `0x25` is not the L7 entry).
- **Fixture-live:** start-based `0x53→0x52` inland-left micro,
  `recordings/l7_dnp_pond_53.json` leftover play `0x52` `(112,181)`.
- **Fixture-live:** post-L6 bait prefix `0x22→0x32→0x33→0x23→0x24→0x25`,
  leftover play `0x25` `(0,141)`, `recordings/l7_bait_from_l6_l7_bait_25.json`.
  `0x24→0x25` RIGHT @ y=141.
- **Recon (2026-09-02, dead-pocket):** `0x33`/`0x23`/`0x24`/`0x25` have no
  south exit; `0x25` only hidden exit is north x≈208→`0x15`. Bait shop `0x34`
  (E4) is entered from the south (`0x44 ↑ 0x34`). Scripts:
  `scratch/sweep_25_armos.py`, `scratch/probe_24_to_shop.py`,
  `scratch/probe_33_south_to_shop.py`. Screens
  `recordings/l7_25enter2_*`, `l7_24probe_*`, `l7_33s_*`.
- Prior 0x53 miss: `recordings/l7_dnp_pond_assisted_v9.json` `(224,173)`
  `hop10_ay`.
- Pond `0x42`, drain, dungeon entry, Red Candle, and shard remain source-only.
- `route_eligible=false` on every fixture. Did not STATUS-promote.
