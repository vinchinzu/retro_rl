# Overworld doors & key capabilities (first quest)

**Status:** parallel recon wave 2026-08-06 (`rr-2nx` + L3–L9).  
**Rule:** screen IDs marked **verified** are live fceumm facts (assisted OK).
Source-only hex stays labeled until a probe writes `LevelNEntrance.state`.
**Not Clean route-ready** until natural-entry pure from real predecessor.

Primary planning source: [DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md).  
RAM inventory: [ram_map.md](ram_map.md) / `zelda_i.ram`.  
Recon table below is the live door map; dungeon geometry lives in
`LEVEL*_ROUTE.md`.

---

## Dungeon mouths (first quest)

| Level | Name | Door screen (hex) | Entry room | Evidence | Required items (to *enter*) | TF bit | Item inside | Local route doc |
|------:|------|-------------------|------------|----------|-----------------------------|--------|-------------|-----------------|
| 1 | Eagle | **`0x37`** | **`0x73`** | **verified** Clean | wooden sword | `0x01` | Bow (**required** for Gohma; default spine still skips) | [LEVEL1_ROUTE.md](LEVEL1_ROUTE.md) |
| 2 | Moon | **`0x3C`** | **`0x7d`** | **verified** assisted/geometry; Clean health open | wooden sword; TF1 for natural chain | `0x02` | Magical Boomerang | [LEVEL2_ROUTE.md](LEVEL2_ROUTE.md) |
| 3 | Manji | **`0x74`** | **`0x7c`** | **verified** assisted; `Level3Entrance.state` (source ↑L×4 path **blocked** at 0x67) | wooden sword | `0x04` | Raft | [LEVEL3_ROUTE.md](LEVEL3_ROUTE.md) |
| 4 | Snake | **`0x45`** | **`0x71`** | **live** raft dock `0x55` → island `0x45` (rr-0fx) | **Raft** (`0x0660`) | `0x08` | Stepladder | [LEVEL4_ROUTE.md](LEVEL4_ROUTE.md) |
| 5 | Lizard | **`0x0B`** | **`0x76`** | **verified** assisted Lost Hills `0x1B` ↑×4; `Level5Entrance.state` | none to enter | `0x10` | Whistle | [LEVEL5_ROUTE.md](LEVEL5_ROUTE.md) |
| 6 | Dragon | **`0x22`** | **`0x79`** | **verified** assisted; east `0x7a` key room; `Level6Entrance.state` | none required | `0x20` | Magical Rod | [LEVEL6_ROUTE.md](LEVEL6_ROUTE.md) |
| 7 | Demon | **`0x42`** | **`0x79`** | **live** pond drain (recon whistle poke; `Level7Entrance` pin). Bait shop `0x34` still source. Not natural-entry. | **Whistle** to drain; **Bait** inside | `0x40` | Red Candle | [LEVEL7_ROUTE.md](LEVEL7_ROUTE.md) |
| 8 | Lion | **`0x6D`** | TBD | **verified bush OW** assisted (`Level8BushOW`); candle→burn→enter **PARTIAL** (rr-q8a) | **Candle** (`0x065B`) | `0x80` | Book / Magic Key | [LEVEL8_ROUTE.md](LEVEL8_ROUTE.md) |
| 9 | Death Mountain | `0x05` | TBD | source bomb-rock hyp | bombs; full TF `0xFF` inside | — | Red Ring, Silver Arrows | [LEVEL9_ROUTE.md](LEVEL9_ROUTE.md) |

### Source path notes (planning only)

| Level | Walkthrough hops from start `0x77` (or noted origin) | Derived screens |
|------:|------------------------------------------------------|-----------------|
| 3 | ↑, ←×4, ↓, → | `77→67→66→65→64→63→73→**74**` |
| 4 | **live** post-L3 `74→73→63 E@y149→64→65→dock 55` raft↑ island; east heart dock `3F` separate | dock **`0x55`**, island/door **`0x45`**, entry **`0x71`** |
| 4 (alt) | Long ZD path via raft heart island then lake | confirm vs short hyp **live** |
| 5 | Bracelet warp NE → ←×2 Lost Hills → ↑×4 | maze **`0x1B`**, door **`0x0B`** (soft; confirm live) |
| 6 | From L5 `0x0B`: ↓, 0x1B y=141 LEFT after south-around x≈72, ←×6 @y141, 0x15 south band, 0x14/0x23 SE blue ↓, ← ↓ ← ↑ | **`0x22`** **live** (`l6_entry_continuous_v2`) |
| 7 | Bait Armos: ↑ ←×3 ↑×3 → shop `34`; pond: ↓×2 ←×2 ↑ + whistle | shop **`0x34`**, pond **`0x42`** |
| 8 | →×4 ↑×2 → ↓ → + candle bush | **`0x6D`** |
| 9 | → ↑×5 ← ↑×2 ←×2 + bomb left rock | **`0x05`** |

Same hop arithmetic reproduces verified L1 (`…→0x37`) and L2 walkthrough path
(`…→0x3C`); L3–L9 still need emulator confirmation of door tile + entry room.

### Leave → next-mouth stitch

Shared leftover packet: `zelda_i.overworld.stitch.OverworldHandoff` (L8
`PostLevel7Handoff` shape). Full table, dead beliefs,
L1–L8 handoff stitches: see
[Dungeon leave → mouth stitch](#dungeon-leave--mouth-stitch). `verified=False` /
`evidence="hypothesis"` until the predecessor dungeon naturally clears. Do not
treat this as spine-green.

| From | Leave (screen, x/y, mode, TF, items) | To mouth | Enter items | Status |
|------|--------------------------------------|----------|-------------|--------|
| L1 | **`0x37`** ~(112,125) mode 5, TF `0x01` | L2 **`0x3C`** | wooden sword; TF1 | **verified** leave + mouth |
| L2 | **`0x3C`** ~(112,125) mode 5, TF `0x03` | L3 **`0x74`** | wooden sword | **verified** leave + mouth |
| L3 | **`0x74`** ~(128,125) mode 5, TF `0x07`, raft=1 | L4 **`0x45`** via dock `0x55` | **Raft** | **verified** leave + mouth |
| L4 | **`0x45`** ~(128,125) ±4 mode 5, TF `0x0F` | L5 **`0x0B`** | none | **live** leave; mouth verified |
| L5 | **`0x0B`** ~(112,125) ±4 mode 5, TF `0x1F`, Whistle earned | L6 **`0x22`** | none | **live** leave; mouth **verified** |
| L6 | **UNMEASURED** — do **not** assume `0x22`. Spine tip is play `0x09` (56,109) TF `0x1F` (not a leave) | L7 pond **`0x42`** (source); live approach `0x53` (224,173) LEFT-inland-before-DOWN; bait shop `0x34` source | **Whistle**; Bait inside | leave **UNMEASURED**; mouth hypothesis |
| L7 | **UNMEASURED** (expect TF `0x7F`, Candle 2) | L8 bush **`0x6D`** from `0x5D` south x≈48 | **Candle 2** | leave **UNMEASURED**; bush live; burn unsolved |
| L8 | **UNMEASURED** (expect TF `0xFF`, Magic Key, bombs) | L9 Spectacle Rock **`0x05`** bomb | bombs; TF `0xFF`; Magic Key | leave **UNMEASURED**; mouth source/fixture-live |

White sword (5 HC), Magical sword grave `0x21` (12 HC), Bracelet Armos `0x24`
are later OW shortcuts. **Do not grant.**

---

## Key overworld capabilities (non-dungeon)

| Capability | Screen (hex) | Evidence | Requires | Notes / RAM |
|------------|--------------|----------|----------|-------------|
| Wooden sword cave | `0x77` | **verified** | none | start screen NW cave; `ADDR_SWORD` → 1 |
| White sword cave | **`0x0A`** | **verified** live detour from L9 approach | 5 heart containers | plateau; `ADDR_SWORD` → 2 |
| Magical sword grave | `0x21` | source path via bracelet Armos — **TBD live** | 12 hearts; push 3rd-from-left middle gravestone | graveyard; `ADDR_SWORD` → 3 |
| Power Bracelet Armos | `0x24` | source (10 Armos, top-right) — **TBD live** | none | `ADDR_BRACELET` `0x0665`; unlocks boulder warps |
| Blue candle shop(s) | **`0x5E`** O-6 cave (`CandleShop5E`) | **verified** assisted OW path `CANDLE_SHOP_HOPS` + cave UP@x112; fixture-tested buy; **not spliced onto any spine hop table** (blocks heart_h5 0x47) | rupees **60** | `ADDR_CANDLE` `0x065B`; buy touch≈(152,149); L8 bush residual |
| Bait / Food special shop | `0x34` | source Armos top-middle — **TBD live** | 60R | `ADDR_FOOD` `0x065D`; required for L7 Hungry Goriya |
| Whistle pond (L7 mouth) | `0x42` | source — **TBD live** | Whistle | drains water → L7 stairs |
| Raft dock (east heart) | `0x3F` | source path →×8 ↑×4 — **TBD live**; rr-ps7.4.1 walked post-L3 `0x74`→`0x66` (`overworld.raft_heart`) and blocked at `0x66` | Raft | raft↑ optional Heart Container island `0x2F` (`overworld.locations.raft_heart`); dock/launch/pickup still unverified |
| Raft dock (L4 island) | `0x55` → `0x45` | **live** assisted rr-0fx (`level4.overworld`) | Raft | only two first-quest raft docks |
| Ladder heart (coast) | `0x5F` | source — **TBD live** | Stepladder | water platform Heart Container (`overworld.locations.ladder_heart`); Stepladder owned end of L4, unused on OW today |
| Heart container (bomb) | `0x2C` | source — **TBD live** | Bombs | secret take-any cave (`overworld.locations.heart_m3`) |
| Heart container (bomb) | `0x7B` | source — **TBD live** | Bombs | secret take-any cave (`overworld.locations.heart_l8`) |
| Heart container (burn) | `0x47` | source — **TBD live** | Candle | secret take-any cave (`overworld.locations.heart_h5`); ZD Gathering third HC, see `docs/PRE_L1.md` |
| Lost Hills maze | **`0x1B`** | **verified** assisted; enter from `0x1C` W@y140; pocket free then ↑×4 | none | 4th UP → door `0x0B`; see LEVEL5_ROUTE |
| L8 candle bush | **`0x6D`** | **verified** bush pocket; candle buy residual | Candle | burn then enter; see LEVEL8_ROUTE |
| L9 bomb rock | `0x05` | source — **TBD live** | bombs | left rock of pair |
| Bomb capacity upgrades | *inside* L5 / L7 | source walkthrough | 100R each | not OW mouths; raise bomb max 8→12→16 |

---

## Item / progress RAM (door-relevant)

| Item | ADDR | Set by | Gates |
|------|------|--------|-------|
| Triforce bits | `0x0671` | L1–L8 shards | natural order optional; L9 Old Man wants `0xFF` |
| Sword | `0x0657` | caves | combat tier |
| Candle | `0x065B` | shop / L7 red | burn bushes (L8, secrets) |
| Whistle | `0x065C` | L5 | L7 pond; Digdogger |
| Food (bait) | `0x065D` | shop | L7 Hungry Goriya |
| Raft | `0x0660` | L3 | L4 island + raft heart |
| Book | `0x0661` | L8 | wand flames |
| Ring | `0x0662` | shop blue / L9 red | damage reduction |
| Ladder | `0x0663` | L4 | water gaps + ladder heart |
| Magic Key | `0x0664` | L8 | infinite locks |
| Bracelet | `0x0665` | Armos | boulder warps |
| Bombs | `0x0658` | shop / drops | Dodongo, L9 rock, many walls |
| Bow / arrows | `0x065A` / `0x0659` | L1 bow + shop | Gohma eye; Silver Arrows finish Ganon |

Triforce bit map (matches walkthrough):

| Shard | Bit | Dungeon |
|------:|----:|---------|
| 1 | `0x01` | Eagle (**verified**) |
| 2 | `0x02` | Moon |
| 3 | `0x04` | Manji |
| 4 | `0x08` | Snake |
| 5 | `0x10` | Lizard |
| 6 | `0x20` | Dragon |
| 7 | `0x40` | Demon |
| 8 | `0x80` | Lion |
| all | `0xFF` | L9 gate / endgame |

---

## Graph stubs

Planning NamedRoutes (no Clean claims): `zelda_i.route.catalog_later`  
Node id constants: `zelda_i.route.nodes`

Refresh this file when sibling probes land live door screens
(`LevelNEntrance.state` + `LEVELN_ROUTE.md` Evidence section).

## Sources

- Zelda Dungeon: [The Gathering](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/the-gathering/),
  L1–L9 dungeon chapters (linked from walkthroughs doc)
- Local: `docs/research/DUNGEON_WALKTHROUGHS.md`, `overworld/graph.py`, `ram.py`,
  `level3/overworld.py` (L3 path seed)

## Item-gate hops (`rr-iri`)

Full first-quest OW cave/secret catalog (ROM AttrsB dests, open method, vanilla
fill, rando `locations_from_rom`) and enemy-drop farms (`farm_at`,
`five_rupee_farms`, live 0x4A↔0x49 tektite restock): `zelda_i.overworld.locations`.

---

## Dungeon leave → mouth stitch (merged from `docs/tasks/ow-handoff.md`, 2026-09-07)

Packet: `zelda_i.overworld.stitch.OverworldHandoff`. Defaults `verified=False`,
`route_eligible=False`; `complete()` is false until `verified=True` **and**
every measured inventory field is filled.

The full power-on → credits tape (`recordings/BASELINE_20260907.json`) is green,
so every leave below is now walked live; the L7/L8 rows that the original
handoff marked UNMEASURED are superseded by that tape.

| From | Leave pose | To mouth |
|------|-----------|----------|
| L1 | `0x37` ~(112,125) mode 5, TF `0x01` | L2 `0x3C` |
| L2 | `0x3C` ~(112,125) mode 5, TF `0x03` | L3 `0x74` |
| L3 | `0x74` ~(128,125) mode 5, TF `0x07`, raft=1 | L4 `0x45` via dock `0x55` |
| L4 | `0x45` ~(128,125) ±4 mode 5, TF `0x0F` | L5 `0x0B` |
| L5 | `0x0B` ~(112,125) ±4 mode 5, TF `0x1F`, Whistle earned | L6 `0x22` |
| L6 | `0x22` `(112,125)` mode 5, TF `0x3F`, keys 2 bombs 8 rupees 42 Rod 1 Bow 1 arrows 1, 8 HC | L7 pond `0x42`; bait shop `0x34` |
| L7 | TF `0x7F`, Candle 2, Whistle retained | L8 bush `0x6D` from `0x5D` south x≈48 |
| L8 | TF `0xFF`, Magic Key, bombs | L9 Spectacle Rock `0x05`, bomb the left rock |

Cumulative TF after clear: L1 `0x01` … L6 `0x3F` … L7 `0x7F` … L8 `0xFF`.
The L9 Old Man wants `0xFF`. `(112,125)` is the dungeon mouth tile — a fanfare
return lands Link **on** it, so re-entry guards need a `_left_mouth` latch.

### Open: bait-shop approach (`rr-8t4.4` / `rr-8t4.5`)

The live approach attempted was through **`0x53` (224,173)**, LEFT-inland
before DOWN — recorded as a **live fail**. The post-L6 pocket
(`0x22/0x32/0x33/0x23/0x24/0x25`) has no southward outlet to the row-4/5 band
holding pond `0x42` and shop `0x34`; those are reached walking north out of the
western forest band from *start*. This is the blocker for the natural 60R Bait
buy that Phase 6.5 needs.

### Later overworld shortcuts — needed, **do not grant**

| Capability | Screen | Requires | Status |
|------------|--------|----------|--------|
| White Sword cave | `0x0A` (reached from row 0 via `0x1A` north at x=208) | 5 heart containers | **live** — on the L9 entry chapter |
| Magical Sword grave | `0x21` | 12 HC; push the 3rd-from-left middle gravestone | source only; `0x21` currently unreachable (`level7/overworld.py:91-95`) |
| Bracelet Armos | `0x24` | none; top-right of 10 | source only — **but the spine already walks `0x24`** (Phase 5.2) |

### Bait shop `0x34` — measured geometry (recon, single-run)

From the L7-A reference recon, preserved at tag **`recon/rr-8t4.5-bait-shop`**
(commit `bd9ac79f`; its four probe CLIs were not merged into the cleaned tree —
`git show` the tag to re-read them). Single-run, **not 2/2** — re-verify before
routing.

- **`0x34` is the Armos bait shop**: a 3×2 statue grid. The special
  (staircase) Armos is the **LEFT column**, ~`(64,118)` in link coords →
  **mode 16 at `(64,125)`**.
- **Shop cave interior**: Link spawns ~`(112,213)`; merchant type **`0x7a`** at
  `(120,128)`; two type-`0x40` pedestals at x=**72** and x=**168**, y=128.
  Message "BOY, THIS IS REALLY EXPENSIVE!"; dialog settles by frame ~4.
- **Stock**: Key 80 (left), Blue Ring 250 (mid), **Bait 60 (right, x≈168)**.
- A **pre-L6** walkable leg exists: PostSwordStart → western forest band →
  `0x54` → `0x44` → `0x34`, reusing the `OverworldToLevel7Pond` hops to `0x54`
  then a manual north push (`0x54`→`0x44` gap at x≈116, `0x44`→`0x34` gap at
  x≈132). **This is not what `rr-8t4.4` asks for** — that bead wants the
  *post-L6* route from `MEASURED_POST_L6_EXIT`, out of the `0x22` mountain
  pocket.
- **Not done**: the buy touch was never confirmed (food 0→1, −60R).

This is the geometry `NaturalBaitPurchaseController` (`level7/entry.py:107`)
needs to stop failing closed — it wants `shop_geometry_verified=True` plus
`shop_cave_xy` on `BaitPurchasePlan`. See Phase 6.5 in
[CLEANUP_PLAN.md](CLEANUP_PLAN.md).
