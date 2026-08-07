# Dungeon walkthroughs (planning sources)

**Status:** planning accelerators only. Room IDs, transitions, object types,
and stop predicates must be verified live in fceumm before a segment is
route-ready. Claims below are paraphrased from external guides, not emulator
facts.

Primary source (Zelda Dungeon 100% first-quest walkthrough):

| Level | Title | URL |
|------:|-------|-----|
| — | The Gathering (overworld prep) | https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/the-gathering/ |
| 1 | The Eagle | https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-1-the-eagle/ |
| 2 | The Moon | https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-2-the-moon/ |
| 3 | The Manji | https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-3-the-manji/ |
| 4 | The Snake | https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-4-the-snake/ |
| 5 | The Lizard | https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-5-the-lizard/ |
| 6 | The Dragon | https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-6-the-dragon/ |
| 7 | The Demon | https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-7-the-demon/ |
| 8 | The Lion | https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-8-the-lion/ |
| 9 | Death Mountain | https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-9-death-mountain/ |

Secondary: [IGN walkthrough index](https://www.ign.com/wikis/the-legend-of-zelda/Walkthrough),
[RPGClassics dungeon maps](https://tartarus.rpgclassics.com/zelda1/1stquest/dungeonmaps.shtml).

Local Level 1 correlation: [LEVEL1_ROUTE.md](../LEVEL1_ROUTE.md).
Local Level 2 notes: [LEVEL2_ROUTE.md](../LEVEL2_ROUTE.md).

---

## Level 1 — The Eagle (verified locally)

- **OW door (verified):** screen `0x37`.
- **Item:** Bow (optional speed skip).
- **Boss:** Aquamentus (sword).
- **Triforce bit:** `0x01`.
- Full live route: `docs/LEVEL1_ROUTE.md`.

---

## Level 2 — The Moon

Source:
[Zelda Dungeon L2](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-2-the-moon/),
[IGN Dungeon Two](https://www.ign.com/wikis/the-legend-of-zelda/Dungeon_Two).

### Overworld approach (source)

From start screen: right 4, up 2, right 2, up, left, up → dungeon mouth.
Enemies: Octoroks, Tektites, Zolas, Leevers, Moblins.

**Live correlation:** Moon door is overworld screen **`0x3C`**. Post-L1 walk
prefix is verified through **`0x4A`**; suffix `0x4A→…→0x3C` is the active
blocker (health).

Potion shop nearby (source): from dungeon down, right, down, left 2, up along
right side; burn 3rd-from-top bush on left with Blue Candle.

### Interior (source speed route)

| Step | Direction / action | Notes |
|------|--------------------|-------|
| Entry | UP | 12 Keese optional skip |
| N | clear Ropes | door LEFT opens |
| W | clear 6 Ropes | **key** drop; return entry |
| E of entry | clear 5 Ropes | **key**; UP |
| N | 3 Ropes optional | key RIGHT locked |
| E | 6 Gel, **Compass** corner | optional bomb N shortcut |
| N (or bomb) | 5 Red Goriya skippable | key RIGHT; kill-one-kill-all quirk |
| E | 5 Gel, **Map** | optional bomb N |
| N (or bomb) | door seals; clear 5 Rope | **key**; RIGHT |
| E | 3 Blue Goriya + 4 statues | **Magical Boomerang** |
| bomb N / detour | Moldorm | **key**; RIGHT |
| E | traps + 4 Keese | bombs reward; bomb N optional |
| N | clear Ropes (unlock) | RIGHT for Gel rupees + Old Man “Dodongo Dislikes Smoke” |
| N | clear 5 Red Goriya | **bombs** drop; UP boss |
| Boss | **Dodongo** | bombs only (2 mouths); Heart Container |
| E of boss | Triforce shard 2 | center of room |

**Key items:** Magical Boomerang (full-screen). Bombs required for Dodongo.
**Boss:** Dodongo (smoke/bombs). **Triforce bit:** `0x02`.

### Live checklist

- [x] Entry room **0x7d**; north **0x6d** (5 Ropes); west **0x6c** (6 Ropes, key `0x19`)
- [x] Rope object type **0x28** (TYPE_AND_HP when play-settled; spawn ~100f after screen)
- [x] Entry east **0x7e** (5 Ropes + key `0x19`) via diamond-nav; UP→**0x6e** (3 Ropes)
- [x] Magical Boomerang inventory: `0x0674` / `0x0675` (not yet collected live)
- [x] **0x6e RIGHT** → **0x6f** compass gels; **0x6f bomb N @(120,101)** → **0x5f**
- [x] **0x5f LEFT key** → **0x5e** Goriya `0x06` live
- [ ] Live room ID for Magical Boomerang / boss / triforce (past 0x5f RIGHT residual)
- [x] Goriya object type **0x06** on L2 (0x5e); Moldorm still open
- [ ] Bomb placement policy for Dodongo
- [ ] Isolated + natural-entry clear with `triforce & 0x02`

---

## Level 3 — The Manji

Source:
[Zelda Dungeon L3](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-3-the-manji/).

### Overworld (source)

From start: up, left 4, down, right 1. Harder dungeon; potion recommended
(shop: left, up, right of door).

### Interior (source)

- Entry → LEFT → Zols (split to Gel with wooden sword) + key → UP → more Zols + key → UP
- Darknuts (side/back hits only); bombs reward; bomb RIGHT = boss shortcut (skip if need item)
- LEFT → Keese + **Compass** → key LEFT → clear Darknuts (south door) → DOWN
- Many Darknuts; staircase → Keese path → **Raft**
- Backtrack: up, right 2, up → Zols + key → key RIGHT → Map room near boss
- Bubbles (disarm sword), Keese, Zols → UP **Manhandla** (bombs best)
- Heart Container → Triforce shard 3

**Item:** Raft. **Boss:** Manhandla. **Triforce bit:** `0x04`.

---

## Level 4 — The Snake

Source:
[Zelda Dungeon L4](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-4-the-snake/).
Local route plan: [LEVEL4_ROUTE.md](../LEVEL4_ROUTE.md). Scaffold:
`level4_overworld.py`, `scripts/probe_level4_entry.py --plan-only`.

### Overworld prep (source)

- **Gate:** Raft from L3 (`ADDR_RAFT` `0x0660`). Do not poke for Clean.
- Raft heart: from start east 8, north 4 dock → cave Heart Container (vs potion)
  - Hypothesized dock **`0x3F`**, cave **`0x2F`** (hop math; not live)
- Dungeon: raft dock island → building mouth
  - Hypothesized lake dock **`0x55`** (start U L×2 U), island **`0x45`** (not live)
- Full ZD path from L3-area dock: L, U, L×6, D×3, L×3, raft island (source)

### Interior (source)

- LEFT Keese + key; UP Vires (split to red Keese); key RIGHT dark maze → **Compass**
- Ladder of dark rooms + keys; water blocks north until item
- RIGHT clear Vires → Like-Likes + Zols → push left block → **Stepladder** (`ADDR_LADDER` `0x0663`)
- Use ladder over water → Map; optional bomb shortcuts / Manhandla side fight
- Old Man: “Walk Into The Waterfall” (L5 clue)
- Clear Vires + Keese, push block → **Gleeok** (2 heads)
- Heart → Triforce shard 4

**Item:** Stepladder. **Boss:** Gleeok (2-head). **Triforce bit:** `0x08`.
**Live:** none (entry room / door screen TBD).

---

## Level 5 — The Lizard

Source:
[Zelda Dungeon L5](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-5-the-lizard/).

### Overworld prep (source)

- Ladder heart on water platform (east coast)
- Power Bracelet (top-right Armos of 10)
- Magical Sword (graveyard; push 3rd-from-left middle gravestone; need 12 hearts)
- Lost Hills: walk **up four times** to dungeon

### Interior (source)

- RIGHT Pols Voice (sword/arrows) + key
- Dark Gibdo rooms, Dodongo optional (prefer bomb wall skip)
- **Map**; Zol key; Gibdo bombs; Blue Darknuts → staircase
- LEFT more Darknuts → staircase → **Whistle (Recorder)**
- Optional bomb capacity upgrade (100 rupees)
- Digdogger: play Whistle to shrink, sword/bomb finish
- Heart → Triforce shard 5

**Item:** Whistle. **Boss:** Digdogger. **Triforce bit:** `0x10`.

---

## Level 6 — The Dragon

Source:
[Zelda Dungeon L6](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-6-the-dragon/).

### Overworld (source)

West near graveyard. Bracelet warp shortcut from start (middle staircase).
Potion shop: bomb between two staircases one screen SE of door.

### Interior (source)

- RIGHT Orange Wizzrobes + key; LEFT locked (do not waste key on Old Man first)
- **Compass** from Zols; statue/Keese rooms; hard multi-Wizzrobe + Bubble + Like-Like clear
- Mid-dungeon **Gleeok (3 heads)** then Map
- Staircase → **Magical Rod/Wand**
- More Vires / Wizzrobes → staircase to **Gohma**
- Gohma: **one arrow to open eye**; Heart → Triforce shard 6

**Item:** Magical Rod. **Boss:** Gohma (arrow eye). **Triforce bit:** `0x20`.

---

## Level 7 — The Demon

Source:
[Zelda Dungeon L7](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-7-the-demon/).
Local route plan: [LEVEL7_ROUTE.md](../LEVEL7_ROUTE.md). Scaffold:
`level7_overworld.py`, `scripts/probe_level7_entry.py --plan-only`.

### Overworld (source)

- **Gates:** Whistle from L5 (`ADDR_WHISTLE` `0x065C`); Bait mid-dungeon
  (`ADDR_FOOD` `0x065D`). Do not poke for Clean.
- Buy **Bait/Food** (60R special shop: start up, left 3, up 3, middle top Armos)
  - Hypothesized shop screen **`0x34`** (not live)
- Whistle at pond (from shop: down 2, left 2, up) drains water → entrance
  - Hypothesized pond **`0x42`** (not live)

### Interior (source)

- Many bomb walls; only 4 keys for 5 locks (bomb-skip or pre-carry key)
- Digdogger re-spawns (whistle splits into multiples)
- Bomb upgrade #2 (16 bombs)
- Hungry Goriya needs **Bait**
- **Map** + tip-of-nose secret (push block in nose tip room) → staircase
- **Red Candle**; force-clear Digdogger before boss path
- Boss: **Aquamentus** (same as L1); Heart → Triforce shard 7

**Item:** Red Candle. **Boss:** Aquamentus. **Triforce bit:** `0x40`.
**Live:** none (pond / entry room TBD).

---

## Level 8 — The Lion

Source:
[Zelda Dungeon L8](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-8-the-lion/).

### Overworld (source)

From start: right 4, up 2, right, down, right; burn lone bush with Candle.

### Interior (source)

- Optional routes; items not required for credits
- Manhandla early; staircase → **Book of Magic** (wand flames)
- Darknut gauntlets + keys; **Compass** / **Map**
- Gohma (arrow) side room
- Staircase → **Magical Key** (infinite doors)
- Boss: **Gleeok (4 heads)**; Heart → Triforce shard 8

**Items:** Book of Magic, Magical Key. **Boss:** Gleeok 4-head. **Triforce bit:** `0x80`.

---

## Level 9 — Death Mountain

Source:
[Zelda Dungeon L9](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-9-death-mountain/).
Local route plan: [LEVEL9_ROUTE.md](../LEVEL9_ROUTE.md). Scaffold:
`level9_overworld.py`, `scripts/probe_level9_entry.py --plan-only`.

### Overworld (source)

- **Gate:** full Triforce `ADDR_TRIFORCE == 0xFF` for Old Man; bombs for rock.
- From start: right, up 5, left, up 2, left 2; bomb left rock of pair.
  - Hypothesized Spectacle Rock **`0x05`** (hop math; not live)
  - Nearby potion shop hypothesized **`0x04`** (one left)
- Bring full Red Potion (shop one screen left).

### Interior (source) — Magical Key path (summary)

- Full Triforce Old Man gate
- Lanmola; undergrounds; Like-Likes threaten Magical Shield
- **Patra** (orbiting eyes) — first skippable, second drops **Map**
- Staircase → **Red Ring** (`ADDR_RING` planned value 2)
- More Patra / Wizzrobe clears → **Silver Arrows** (`ADDR_ARROWS` planned value 2)
- Final Patra → **Ganon** (must stun then Silver Arrow)
- Princess Zelda / ending (credits stop **unverified** stub)

**Items:** Red Ring, Silver Arrows. **Boss:** Ganon. Routes differ with/without Magical Key.
**Live:** none (rock / entry / ending TBD).

---

## Triforce bit map

| Shard | Bit | Dungeon |
|------:|----:|---------|
| 1 | `0x01` | Eagle (verified) |
| 2 | `0x02` | Moon |
| 3 | `0x04` | Manji |
| 4 | `0x08` | Snake |
| 5 | `0x10` | Lizard |
| 6 | `0x20` | Dragon |
| 7 | `0x40` | Demon |
| 8 | `0x80` | Lion |

---

## Usage rules

1. Prefer this file + Zelda Dungeon URLs when planning a new dungeon.
2. Never promote a room controller until live RAM shows room id, enemy types,
   and a 2/2 isolated stop predicate.
3. Keep speed skips explicit (e.g. L1 skips Bow/Map/Boomerang; L2 may skip
   Compass/Map if keys allow).
4. Dodongo / Digdogger / Gohma require inventory actions (bombs, whistle, arrows)
   — controllers must equip B-item, not only mash A.
