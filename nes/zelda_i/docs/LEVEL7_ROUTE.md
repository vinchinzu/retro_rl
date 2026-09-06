# Level 7 — The Demon (route notes)

**Status (2026-09-05):** Survival `--through level7` is **spine-green 2/2
from power-on**. Leftover OW `0x42` `(96,93)` TF `0x7F`. Living residual
is L8-A: [`tasks/rr-6o7.1-residual.md`](tasks/rr-6o7.1-residual.md).
Sections below still describe the Phase 1 recon; they are not the live
spine. Do not STATUS.

**Status:** Phase 1 (2026-09-02). The **L6 leave is measured and verified**:
`--through level6-exit` 2/2 → OW `0x22` `(112,125)` TF `0x3F`, keys 2 bombs 8
rupees 42, `selected_item=2` (arrows), Whistle 1, Food 0, Candle 0, 8 HC full.
`MEASURED_POST_L6_EXIT.verified=True`. On a continuous power-on
`--through level7-entry` the post-L6 controller walks **`POST_L6_TO_POND_HOPS`**
(`0x22↓0x32→0x33↑0x23→0x24↑0x14←0x13`). Leftover OW `0x13` `(240,189)`.
`_after_hops` succeeds only on pond `0x42` and fail-closes there. The
`0x22→0x25` bait prefix is a **dead spur**, not the spine default.

At `level7_bait_purchase` the Survival spine now runs
`SurvivalBaitPurchaseController` (`l7_hops(survival=True)`): one disclosed
`ADDR_FOOD` write (`$065D` → 1) in place of the natural 60R buy, since the
natural L6→shop overworld route is a mountain-locked pocket (bead `rr-8t4.4`).
The disclosed rupee **count** top-up (42→60R, `SPINE_L7_RUPEE_RETOPUP`) still
fires before the stage and represents the cost paid. **Clean** keeps
`NaturalBaitPurchaseController` fail-closed. See `docs/ASSIST_CONTRACT.md`.
The spine then fails closed at `level7_pond_drain_entry` (pond `0x42` is
still unreached from the L6 leave; leftover is `0x13`. Natural Whistle drain
is unobserved. The Survival bait prefix is a dead spur, bead `rr-8t4.4`).

The **Demon pond `0x42` overworld screen is reached 2/2** (geometry-only)
by `OverworldToLevel7PondController` from `PostSwordStart`. A **recon
`ADDR_WHISTLE` poke** on that leftover drains the pond and enters L7:
play **`0x79` `(120,205)`** south mouth (`Level7Entrance` pin,
`scratch/pond/probe_l7_pond_drain.py` drain_v2, 825f from `OW_L7Pond`). Stairs
trigger at pond `(96,132)` tile 114. **Not natural-entry** — TF=0, Food=0,
Whistle poked. The bait shop `0x34` and all rooms past entry stay
hypothesis. Spine chapters stay fail-closed (`route_eligible=false`).

**Beads:** `rr-7vc` (closed planning), `rr-dnp` (live pond approach),
`rr-8t4.1` (Wave A + Food fixture), `rr-8t4.2` (L7-B; entry pin captured),
`rr-8t4.4` (natural L6→shop OW route). Do not STATUS-promote.

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
| **Stepladder** | `ADDR_LADDER` (`0x0663`) ≠ 0 | `0x49` full-width water moat (tile `0xF4` ~y120); L4 item, required to reach DIGDOGGER_2 |
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
- **`0x32` has no south exit either** (`scratch/pond/probe_32_pond_to_shop.py`):
  its south edge is solid mountain at x∈{56,80,96,192}; its only exits are
  NORTH (x≈120 → `0x22`, back to L6) and EAST (y≈141 → `0x33`, the fixture
  link). So the entire `0x22 / 0x32 / 0x33 / 0x23 / 0x24 / 0x25` region is a
  **mountain-locked pocket** whose only outlets are `0x22`'s own edges, the
  north-wall gaps on `0x33`/`0x23`/`0x24`, and the hidden `0x25→0x15` passage.
  There is **no direct southward route from L6 to the row-4/5 band** that
  holds the pond `0x42` (E5-ish) and shop `0x34` (E4).
- **Strategic implication for the route owner:** the shop `0x34` and pond
  `0x42` are both entered walking north out of the western forest band
  (`…0x64→0x54→0x44↑0x34`; `…0x54→0x53→0x52→0x42↑`). That band is the
  `LEVEL7_POND_APPROACH_HOPS` territory, reached from **start**, not from L6.
  Options: (a) buy Bait *before* L6 on the way through, then L6, then approach
  the pond from the west; (b) from the post-L6 pocket, exit NORTH and loop
  west/south around the mountains to the forest band; (c) accept a longer
  post-L6 overworld leg. The `0x22→…→0x25` fixture prefix is a dead spur —
  the spine currently walks it "green" but it makes no progress toward Bait.

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

### Recorder warp — the escape from the post-L6 pocket (H1, 2026-09-05)

The `0x22` post-L6 pocket has **no overland outlet** to the pond band: every
edge of `0x22 / 0x32 / 0x33 / 0x23 / 0x24 / 0x25 / 0x14 / 0x13 / 0x12` was
walked or `$6530` tile-mapped dead across three sittings (`0x12→0x02` and
`0x32→0x31→0x41` included; `0x41→0x42` RIGHT is a full-height wall).

The route out is the **Recorder itself**, owned since L5. Blowing it on a
**non-entrance** overworld screen starts a whirlwind-carry cutscene (mode
5→6→7→4→5, no player input) that drops Link on the door screen of a completed
dungeon, cycling by facing. Live cycle facing DOWN from `0x24`:

```text
0x22 (L6) → 0x0B (L5) → 0x45 (L4) → 0x74 (L3) → 0x3C (L2)
```

**`0x45`, the L4 island door, is one screen NORTH of `0x55`** — already on the
green `LEVEL7_POND_APPROACH_HOPS`. That is the join:

```text
0x22 ↓0x32 →0x33 ↑0x23 →0x24   (walk, POST_L6_TO_WARP_HOPS)
0x24 blow ×8 facing DOWN → 0x45  (level7.warp.RecorderWarpController)
0x45 ↓0x55 ↓0x65 ←0x64 ↑0x54 ←0x53 ←0x52 ↑0x42   (WARP_JOIN_TO_POND_HOPS)
```

- **Live 2/2 byte-identical**, post-L6 leave → pond `0x42` `(128,221)` mode 5
  at frame **4913** both trials (`scratch/pond/probe_recorder_warp_full_route.py`,
  tags `rw_full_route_t2` / `_t3`, `writes=0`). Warp determinism confirmed 3/3
  (`rw_cycle_t1`/`t2` plus the earlier `rw_cycle_down2`): 8 blows to `0x45`,
  11 to `0x74`, exactly 3 blows per dungeon-advance after the first. The
  production controller is screen-checked rather than count-locked.
- **`0x45→0x55` is `align_x=128`**, the raft-dock column the whirlwind happens
  to drop Link on — the same hop `level5/overworld.py` already flies live.
  Raft is legitimately owned (L3 item; the leave carries TF `0x3F`).
- **Keep `align_x=128` through `0x55→0x65` too.** The stock hop's
  `align_x=112` assumes the *east* `0x56→0x55` arrival band and drags Link
  LEFT into the mid-screen house/tree mass (tile cols 14-17, rows 8-11);
  that burned a full 30,000f budget at `(128,103)` in `rw_full_route_t1`.
- **`0x74 → 0x64` UP is DEAD**: `0x74`'s entire north edge is mountain across
  all 32 tile columns (`$6530` dump plus a live 10-column sweep, every
  candidate stuck at `y=85`).

No pokes: the Recorder blow is natural play, so this route is Clean-eligible
geometry — unlike the recon `ADDR_WHISTLE` poke it replaces.

### Whistle pond (source)

From bait shop screen: **down×2, left×2, up** → pond (looks like fairy pond
but is not). Equip Whistle on B, use once → water drains → stairs into L7.

| Landmark | Source hops from shop `0x34` | Hypothesized id | Live? |
|----------|------------------------------|-----------------|-------|
| L7 pond / entrance | D×2 L×2 U | **`0x42`** | no |

The executable pond controller skips the unverified shop detour. Geometry-only
(no Whistle required). **2026-09-02: the full pond approach is green** —
`OverworldToLevel7PondController` from `PostSwordStart` reaches the Demon pond
`0x42` **2/2** (`l7_pond_from_start_l7_pond_v7/v8.json`, 3530f, `phase=DONE`,
leftover play `0x42` `(128,221)`; zero deaths / progression / capacity writes;
`route_eligible=false`). The whole hop chain:

```text
0x77→0x78→0x68→0x58→0x57→0x56→0x55→0x65→0x64→0x54→0x53→0x52→0x42
```

Live geometry (Survival, `PostSwordStart`):

- `0x65→0x64` arrives on the east ledge around `(232,109)`; DOWN to the open
  band, LEFT to the north gap at **`x≈60`** (`BAIT_64_GAP_X`;
  `scratch/probe_64_north_to_54.py` — `x≤40` stalls at `y≈93`), then UP to
  `0x54`. *(This was the long-standing `rr-dnp` v10 wall.)*
- `0x54→0x53` is LEFT around `y≈141`.
- **0x53→0x52 (solved):** LEFT inland from the east edge (`x>192`) before
  descending, then LEFT at/below `y≈189` into `0x52` (`pond_53_to_52_action`).
  *(Long-standing `rr-dnp` v9 wall.)*
- **0x52→0x42 (solved):** `0x52` is a boulder field. Climb the open west
  column `x≈48` from the bottom corridor to the mid-band `y≈120`, traverse
  RIGHT to `x≈132`, then UP funnels Link through the wall gap (`~x128`) into
  the `x≈112` north gap to `0x42` (`POND_52_*`;
  `scratch/probe_52_wall.py` — the gap is not at `x=112`).

Evidence: `recordings/l7_pond_from_start_l7_pond_v7.json` / `_v8.json` (2/2)
and `l7_pond_v7_final.png` (the drained-pending Demon pond, blue water, Link
`(128,221)`). Zero deaths; `progression_writes=capacity_writes=0`;
`route_eligible=false`. **Drain + interior (recon, 2026-09-02):** poke
`ADDR_WHISTLE` + B-slot 5 on `OW_L7Pond`, blow, walk the dry bed. Stairs
at `(96,132)` tile 114 → mode 16 → L7 play **`0x79` `(120,205)`**. Pin
`Level7Entrance` (development_only). Natural whistle-carrying drain from
the L6 leave is still `rr-8t4.4` / post-L5 OW.

**Controllers:** the spine uses `level7.entry.PostLevel6OverworldController`
(shared `OverworldHandoff` gate + fixture-live `POST_L6_TO_BAIT_HOPS`), which
refuses every frame until the handoff verifies. `OverworldToBaitShopController`
(ungated post-L6 fixture) and `OverworldToLevel7PondController` (start-based
pond recon) in `level7.overworld` stay recon-only. Isolated
`probe_level7_entry.py` pruned. Whistle is a pond-**drain** gate, not a
geometry-walk gate.

### Live recon goals

1. Reach pond screen without Whistle (map pond geometry only). **done**
2. Save `OW_L7Pond` if pond screen confirmed. **done** (geometry, whistle=0)
3. Drain + enter, confirm `level == 7`, entry room. **done recon** (`0x79`)
4. Save `Level7Entrance`. **done** (whistle-poke pin; not natural-entry)

**Do not** poke Whistle / Food for Clean claims. The entry pin is labeled
`development_only` / `natural_entry=false`.

---

## Interior (source speed route)

RAM room IDs: **entry `0x79` live** (south mouth `(120,205)`); **north dest
`0x69` live** (south mouth `(120,205)`, goriya `0x05` — not Moldorms);
**east of `0x69` = `0x6A` live** (west mouth `(16,141)`, keese `0x1b`, and
the room is **dark**, matching the source `keese_dark` node). Offline
graph: `level7/graph.py` (source ids `0x7xx`; ENTRY, N-path `0x701` and
`KEESE 0x702` have `ram_id`, `evidence=fixture-live`). Prefer bomb walls over
the fifth lock; Hungry Goriya is a Food gate; Red Candle is `ADDR_CANDLE`
1→2 naturally. Key/bomb ledger is in `LEVEL7_KEY_BOMB_LEDGER`.

**`cur_opened_doors` is not a walkability test in L7.** Both live doorways
(`0x79` north, `0x69` east) are `GateKind.OPEN`: black passage on the spawn
frame, byte stays `0` forever. Gating a push on the RIGHT bit is what kept
`0x69` east red for three sittings. In-room traverses use deterministic
waypoints, not the per-pixel occupancy grid — `WALK_SPEED=1`, so four graded
misses on one cell block all four neighbours and BFS stands forever.

Key themes: bomb walls, key shortage, Digdogger re-spawns, hungry Goriya,
“tip of the nose” staircase, Red Candle, forced Digdogger before boss,
Aquamentus.

| Step | Action (source) | Notes |
|------|-----------------|-------|
| Entry | play `0x79` `(120,205)` south mouth **(live)** | north + east doors; water tiles |
| N path | dest `0x69` **(live)**; source said Moldorms | live goriya `0x05`; east is an OPEN doorway |
| R | dest `0x6A` **(live)**, keese `0x1b`, **dark room** | Candle 0 on the pin: unlit |
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
packet, **`verified=True`** (`--through level6-exit` 2/2, `selected_item=2`;
`route_eligible` still `False`). The spine controller
(`level7.pond.PostLevel6OverworldController`) walks `POST_L6_TO_POND_HOPS`
(`0x22↓0x32→0x33↑0x23→0x24↑0x14←0x13`) as its default hops. The measured
leave stands on the `0x22` mouth tile, so re-entry refusal only arms after
Link steps off it (`_left_mouth` latch). First action from the mouth is
DOWN, never UP. On `--through level7-entry` the prefix greens through
`0x13` `(240,189)` and `_after_hops` fail-closes until pond `0x42`.
`level7_bait_purchase` runs the Survival `SurvivalBaitPurchaseController`
(one disclosed `ADDR_FOOD` write, `rr-8t4.4`) and passes; the spine fails
closed at `level7_pond_drain_entry`. `POST_L6_TO_BAIT_HOPS` (`0x22→0x25`)
is a dead spur kept for bait-micro tests. The `0x77`-start pond walk and
the ungated `OverworldToBaitShopController` stay recon-only. Pond probes:
`scratch/pond/`.

`level7/spine.py` exposes only the three plan-level targets:

| `--through` | Internal chapter stages | Current evidence |
|-------------|-------------------------|------------------|
| `level7-entry` | post-L6 overworld; Bait (Survival Food fixture / Clean natural); pond drain/entry | wired; `level7_post_l6_overworld` + `level7_bait_purchase` (Survival) **green**; fails closed at `level7_pond_drain_entry` |
| `level7-red-candle` | entry to Hungry Goriya; tip-of-nose stairs; Red Candle pickup | wired; fails closed |
| `level7` | forced Digdogger; Aquamentus/heart; shard/settled leave | wired; fails closed |

Factories in `level7/hops.py` always return fresh controllers. **The L6 leave
is measured and `verified=True`** — `MEASURED_POST_L6_EXIT`, OW `0x22`
`(112,125)` TF `0x3F`, `selected_item=2` (`--through level6-exit` 2/2). The
post-L6 controller walks `POST_L6_TO_POND_HOPS` through `0x13`. On the **Survival** spine
`level7_bait_purchase` runs `SurvivalBaitPurchaseController` — one disclosed
`ADDR_FOOD` write (bead `rr-8t4.4`; the natural L6→shop route is a
mountain-locked pocket) — plus the `SPINE_L7_RUPEE_RETOPUP` 42→60R count
top-up for the cost paid. **Clean** keeps `NaturalBaitPurchaseController`
fail-closed (`shop_cave_xy=None` → `bait_shop_geometry_unobserved`). Hungry
Goriya fails closed without Food. Red Candle fails closed until
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

`LEVEL7_ENTRY_STOP.screen` is live `0x79` with `evidence=fixture-live`;
Red Candle and complete stay `None` / `hypothesis`. All three fail closed
(`route_eligible=false`; public success accepts only `natural-segment` or
`spine-green`). A room id alone is insufficient.

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
| `OW_L7Pond` | **live** pond `0x42` `(128,221)` from PostSwordStart; whistle=0 |
| `OW_L7BaitShop` | Armos shop screen |
| `Level7Entrance` | **live recon pin** `level==7` play `0x79` `(120,205)`; whistle poked |
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
`POST_L6_TO_POND_HOPS` through `0x13`, applies the Survival Food fixture, and
stops fail-closed at `level7_pond_drain_entry` (pond `0x42` unreached).

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
- **Spine-green (continuous power-on, prefix):** `level7_post_l6_overworld`
  walks `POST_L6_TO_POND_HOPS` `0x22↓0x32→0x33↑0x23→0x24↑0x14←0x13`, leftover
  OW `0x13` `(240,189)`, `writes=0`. `_after_hops` fail-closes until pond
  `0x42`. `route_eligible=false`. The old `0x22→0x25` bait prefix is a dead
  spur (`POST_L6_TO_BAIT_HOPS`).
- **Fixture-live (2026-09-02):** L7 interior prefix `0x79 → 0x69 → 0x6A`
  **2/2** from the `Level7Entrance` pin — north door 251f, `0x69` goriya
  clear + OPEN east doorway, leftover play `0x6A` `(16,141)` west mouth,
  3,809f, `deaths=0`, `progression_writes=capacity_writes=0`, byte-identical
  on both trials (`recordings/l7_room69_east_room69_east_v5.json` / `_v6.json`,
  `scratch/probe_l7_room69_east.py`). `route_eligible=false`.
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

### 2026-09-04 sitting (rr-n91a) — cellar 0x7B B→A dest play 0x29 2/2

From `Level7Interior0DNoseCellarReconFixture` (already mode-9 0x7B right
ladder `(192,93)`): DOWN to y=189, LEFT to x=48, UP left ladder. Never UP
on the source ladder (that is the dead 0x0D return). Dest play **`0x29`
`(96,157)` 2/2** (`20260904_C1`/`C2`, 396f). C1 continued the suffix:
bomb-E → **`0x2A`** Aquamentus type `0x3D` (reuse `Level1AquamentusController`,
`tank_hits=True`, HC 3→4) → E-shutter **`0x2B`** → south-around shard →
fanfare → OW **`0x42` `(96,93)` TF `0x40`** on this TF-0 recon pin.
`MEASURED_POST_L7_EXIT.verified` stays False (not Survival `0x7F`).
`NOSE_CELLAR.ram_id` stays None. 0x0D walk-on still open. Spine factories
stay fail-closed. Policy: `level7/cellar.py`. Probe:
`scratch/probe_l7_7b_cellar_cross.py`. Residual:
`docs/tasks/rr-n91a-residual.md`.

### 2026-09-04 sitting — 0x0D walk-on south-of-gap / south-strip MISS

Pin `Level7Interior0DClearedReconFixture` play `0x0D` `(63,149)`. Zero
position pokes. DOWN from the pin pins **`(64,157)` tile 178** (SW diamond):
y=161-164 south-of-gap band never entered; y=189 (L9 room30 south-strip)
unreachable (ROM 0x0D south is WALL). No UP-push, no dest fixture, no
spine flip. `NOSE_CELLAR.ram_id` stays None. Probe flags:
`scratch/probe_l7_room0d_squeeze.py --south-of-gap` / `--south-strip`.
PNGs: `recordings/l7_0d_walkon_20260904_start.png`,
`recordings/20260904_sog_sog_after_down.png`.

### 2026-09-04 sitting — 0x0D walk-on Trial A/B MISS

y=141 gap **HIT** `(94,141)` 27f. DOWN from there to lure `(96,165)` **MISS**
`(96,157)` tile 178. After RIGHT push, plug `(192,133)` LEFT+UP → `(191,133)`,
RIGHT+UP no move; still boxed y≥133. No dest fixture. Probe
`--trial-a` / `--trial-b`.

### 2026-09-04 sitting — 0x0D plus-corner x=144 DOWN MISS

y=141 to `(144,141)` **HIT**. DOWN between statues x=128/160 **MISS**
`(144,157)` tile 178 (x held 144). No south-face, no dest fixture. Probe
`--plus-corner`.

### 2026-09-04 sitting — 0x0D north-corridor UP MISS

`(144,141)` UP **MISS** `(144,117)` tile 179. Fallback x=176 UP **MISS**
`(176,117)` tile 179. y≤99 not entered. No push-then-walk-on. Probe
`--north-up`.

