# Level 7 — The Demon (route notes)

**Status:** PARTIAL assisted overworld approach. The live controller reaches
`0x53`; the `0x53→0x52→0x42` pond suffix is not yet green. There is no pond
checkpoint, entry room, or Clean segment. **Whistle** from Level 5 gates pond
entry; **Bait/Food** is a separate mid-dungeon hungry-Goriya gate.

**Beads:** `rr-7vc` (closed planning), `rr-dnp` (live pond approach).

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

The executable pond controller skips the unverified shop detour. Its live
prefix is:

```text
0x77→0x78→0x68→0x58→0x57→0x56→0x55→0x65→0x64→0x54→0x53
```

Live geometry through `0x53` (Survival, `PostSwordStart`):

- `0x65→0x64` arrives on the east ledge around `(232,109)`; go DOWN to the
  open band, LEFT to the north gap around `x≈48`, then UP to `0x54`.
- `0x54→0x53` is LEFT around `y≈141`.
- Last failure: on `0x53` at `(224,173)`, direct DOWN toward the hypothesized
  lower west gap `y≈189` remains blocked. The next micro should move LEFT
  inland before descending, then attempt LEFT into `0x52`.

Evidence: `recordings/l7_dnp_pond_assisted_v9.json` and
`recordings/l7_dnp_pond_assisted_v9_final.png`. It reports zero deaths and
`progression_writes=capacity_writes=0`; `success=false`, so no checkpoint was
saved and this is not a route claim.

**Controller:** `level7.overworld.OverworldToLevel7PondController`. Isolated
`probe_level7_entry.py` pruned. Whistle is a pond-entry gate.

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

Isolated `probe_level7_entry.py` pruned. Durable runner (no L7 SpineHop yet):

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1
```

Module: `zelda_i/level7/overworld.py`. Pond walk is
`OverworldToLevel7PondController`; Whistle is required to drain.

---

## Evidence

- Live assisted partial through overworld `0x53`:
  `recordings/l7_dnp_pond_assisted_v9.json`.
- Final `0x53` frame: `recordings/l7_dnp_pond_assisted_v9_final.png`.
- Pond `0x42`, drain, and dungeon entry remain source-only.
