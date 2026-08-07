# Level 7 — The Demon (route notes)

**Status:** planning (gated). No live pond screen, entry room, or Clean
segment. Requires **Whistle** from Level 5 plus **Bait/Food** for the hungry
Goriya. Do not claim pure-first until both are real inventory.

**Beads:** `rr-7vc` (plan + whistle gate).

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
for staircase → special shop → **Bait 60R**.

| Landmark | Source hops from start | Hypothesized id | Live? |
|----------|------------------------|-----------------|-------|
| Bait Armos / special shop | U L×3 U×3 | **`0x34`** | no |

### Whistle pond (source)

From bait shop screen: **down×2, left×2, up** → pond (looks like fairy pond
but is not). Equip Whistle on B, use once → water drains → stairs into L7.

| Landmark | Source hops from shop `0x34` | Hypothesized id | Live? |
|----------|------------------------------|-----------------|-------|
| L7 pond / entrance | D×2 L×2 U | **`0x42`** | no |

From start alone (no shop detour): U L×3 U×3 then D×2 L×2 U → same pond.

**Scaffold:** `level7_overworld.py` — shop + pond hop placeholders,
`has_whistle()`, `has_food()`, refuse-without-cap helpers.

### Live recon goals

1. Reach pond screen without Whistle (map pond geometry only).
2. Save `OW_L7Pond` if pond screen confirmed.
3. With real Whistle: drain, enter, confirm `level == 7`, entry room.
4. Save `Level7Entrance` + `recordings/l7_*_recon.json`.

**Do not** poke Whistle / Food for Clean claims.

---

## Interior (source speed route)

Room IDs **unknown**. Key themes: bomb walls, key shortage, Digdogger
re-spawns, hungry Goriya, “tip of the nose” staircase, Red Candle, forced
Digdogger before boss, Aquamentus.

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

## Boss / Triforce stop predicates (stubs)

```text
level7_boss_cleared  — TBD: Aquamentus dead + HC
level7_complete      — ADDR_TRIFORCE & 0x40
```

Scaffold: `level7_triforce_stop(snap)` → `bool(snap.triforce & 0x40)`.

---

## Checkpoints (planned names)

| State | When |
|-------|------|
| `OW_L7Pond` | Pond screen mapped (Whistle optional) |
| `OW_L7BaitShop` | Armos shop screen |
| `Level7Entrance` | `level==7`, play, entry room |
| `Level7RedCandle` | after Red Candle |
| `Level7Complete` | `triforce & 0x40` |

---

## Scaffold / probe

```bash
uv run python zelda_i/scripts/probe_level7_entry.py --plan-only
uv run python zelda_i/scripts/probe_level7_entry.py --infinite-life --save-state
```

Module: `zelda_i/level7_overworld.py`. Live probe **refuses** without Whistle
unless `--plan-only` (or explicit `--allow-missing-caps` for dock-only walk).

---

## Evidence

- Source walkthrough only.
- No live pond/entry recordings yet.
