# Overworld doors & key capabilities (first quest)

**Status:** planning aggregate for L3–L9 recon (`rr-2nx`).  
**Rule:** screen IDs marked **verified** are live fceumm facts. All other hex IDs
are **source path arithmetic** from Zelda Dungeon walkthrough hops on the
`ADDR_SCREEN` grid (`(row << 4) | col`, start `0x77`) until a probe writes
`LevelNEntrance.state` / updates this table. **Not route-ready** until live.

Primary planning source: [DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md).  
RAM inventory: [ram_map.md](ram_map.md) / `zelda_i.ram`.

---

## Dungeon mouths (first quest)

| Level | Name | Door screen (hex) | Evidence | Required items (to *enter*) | TF bit | Item inside | Local route doc |
|------:|------|-------------------|----------|-----------------------------|--------|-------------|-----------------|
| 1 | Eagle | **`0x37`** | **verified** live / `SCREEN_LEVEL1_ENTRANCE` | wooden sword | `0x01` | Bow (optional) | [LEVEL1_ROUTE.md](LEVEL1_ROUTE.md) |
| 2 | Moon | **`0x3C`** | **verified** geometry probe + walkthrough; Clean walk still gated on health | wooden sword; TF1 for natural post-L1 chain | `0x02` | Magical Boomerang | [LEVEL2_ROUTE.md](LEVEL2_ROUTE.md) |
| 3 | Manji | `0x74` | source path: start ↑ L×4 ↓ → ; `level3_overworld.SCREEN_LEVEL3_ENTRANCE` — **TBD live** enter | wooden sword (potion recommended) | `0x04` | Raft | [LEVEL3_ROUTE.md](LEVEL3_ROUTE.md) *(when present)* |
| 4 | Snake | `0x45` | source hyp: raft from dock `0x55` → island `0x45` (`level4_overworld` / [LEVEL4_ROUTE.md](LEVEL4_ROUTE.md)) — **TBD live** | **Raft** (`0x0660`) | `0x08` | Stepladder | [LEVEL4_ROUTE.md](LEVEL4_ROUTE.md) |
| 5 | Lizard | `0x0B` | source: Lost Hills maze `0x1B` ↑×4 — **TBD live** | none to *enter*; bracelet warp optional shortcut | `0x10` | Whistle (Recorder) | [LEVEL5_ROUTE.md](LEVEL5_ROUTE.md) *(when present)* |
| 6 | Dragon | `0x22` | source path from L5 mouth (↓ L×7 ↓ L ↓ L ↑) — **TBD live** | none required; bracelet warp from near-start helps | `0x20` | Magical Rod | [LEVEL6_ROUTE.md](LEVEL6_ROUTE.md) *(when present)* |
| 7 | Demon | `0x42` | source: whistle pond (`level7_overworld` / [LEVEL7_ROUTE.md](LEVEL7_ROUTE.md)); bait shop `0x34` — **TBD live** | **Whistle** (`0x065C`); **Bait** inside | `0x40` | Red Candle | [LEVEL7_ROUTE.md](LEVEL7_ROUTE.md) |
| 8 | Lion | `0x6D` | source planned bush screen (`level8_overworld`; detour around 0x79) — **TBD live** | **Candle** (`0x065B`, blue shop OK) | `0x80` | Book of Magic, Magical Key | [LEVEL8_ROUTE.md](LEVEL8_ROUTE.md) *(when present)* |
| 9 | Death Mountain | `0x05` | source bomb-rock hyp (`level9_overworld` / [LEVEL9_ROUTE.md](LEVEL9_ROUTE.md)) — **TBD live** | bombs; full TF `0xFF` for Old Man gate inside | — | Red Ring, Silver Arrows | [LEVEL9_ROUTE.md](LEVEL9_ROUTE.md) |

### Source path notes (planning only)

| Level | Walkthrough hops from start `0x77` (or noted origin) | Derived screens |
|------:|------------------------------------------------------|-----------------|
| 3 | ↑, ←×4, ↓, → | `77→67→66→65→64→63→73→**74**` |
| 4 | Short path hyp: ↑ ←×2 ↑ dock `55` raft↑ island; east heart dock `3F` separate | dock **`0x55`**, island/door **`0x45`** |
| 4 (alt) | Long ZD path via raft heart island then lake | confirm vs short hyp **live** |
| 5 | Bracelet warp NE → ←×2 Lost Hills → ↑×4 | maze **`0x1B`**, door **`0x0B`** (soft; confirm live) |
| 6 | From L5 `0x0B`: ↓ ←×7 ↓ ← ↓ ← ↑ | **`0x22`** |
| 7 | Bait Armos: ↑ ←×3 ↑×3 → shop `34`; pond: ↓×2 ←×2 ↑ + whistle | shop **`0x34`**, pond **`0x42`** |
| 8 | →×4 ↑×2 → ↓ → + candle bush | **`0x6D`** |
| 9 | → ↑×5 ← ↑×2 ←×2 + bomb left rock | **`0x05`** |

Same hop arithmetic reproduces verified L1 (`…→0x37`) and L2 walkthrough path
(`…→0x3C`); L3–L9 still need emulator confirmation of door tile + entry room.

---

## Key overworld capabilities (non-dungeon)

| Capability | Screen (hex) | Evidence | Requires | Notes / RAM |
|------------|--------------|----------|----------|-------------|
| Wooden sword cave | `0x77` | **verified** | none | start screen NW cave; `ADDR_SWORD` → 1 |
| White sword cave | TBD live | source (Gathering 1.3) | 5 heart containers | plateau N of start; `ADDR_SWORD` → 2 |
| Magical sword grave | `0x21` | source path via bracelet Armos — **TBD live** | 12 hearts; push 3rd-from-left middle gravestone | graveyard; `ADDR_SWORD` → 3 |
| Power Bracelet Armos | `0x24` | source (10 Armos, top-right) — **TBD live** | none | `ADDR_BRACELET` `0x0665`; unlocks boulder warps |
| Blue candle shop(s) | TBD live | source (Gathering) | rupees | `ADDR_CANDLE` `0x065B`; needed for L8 bush + many secrets |
| Bait / Food special shop | `0x34` | source Armos top-middle — **TBD live** | 60R | `ADDR_FOOD` `0x065D`; required for L7 Hungry Goriya |
| Whistle pond (L7 mouth) | `0x42` | source — **TBD live** | Whistle | drains water → L7 stairs |
| Raft dock (east heart) | `0x3F` | source path →×8 ↑×4 — **TBD live** | Raft | raft↑ optional Heart Container island `0x2F` |
| Raft dock (L4 island) | `0x55` → `0x45` | source hyp (`level4_overworld`) — **TBD live** | Raft | only two first-quest raft docks |
| Ladder heart (coast) | `0x5F` | source →×8 ↑×2 — **TBD live** | Stepladder | water platform Heart Container |
| Lost Hills maze | `0x1B` | source — **TBD live** | none | ↑×4 escapes north to L5 |
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

Planning NamedRoutes (no Clean claims): `zelda_i.routes_later`  
Node id constants: `zelda_i.later_nodes`

Refresh this file when sibling probes land live door screens
(`LevelNEntrance.state` + `LEVELN_ROUTE.md` Evidence section).

## Sources

- Zelda Dungeon: [The Gathering](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/the-gathering/),
  L1–L9 dungeon chapters (linked from walkthroughs doc)
- Local: `docs/research/DUNGEON_WALKTHROUGHS.md`, `overworld.py`, `ram.py`,
  `level3_overworld.py` (L3 path seed)
