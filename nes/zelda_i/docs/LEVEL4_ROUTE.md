# Level 4 — The Snake (route notes)

**Status:** planning (gated). No live OW door ID, entry room, or Clean
segment. Do not claim isolated-pure or natural-entry until Raft is obtained
from Level 3 for real (no inventory poke in Clean STATUS).

**Beads:** `rr-k0w` (plan + raft gate RAM).

Planning sources (external, not emulator facts):

- [Zelda Dungeon — Level 4: The Snake](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-4-the-snake/)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)
- RAM: `zelda_i/ram.py` (`ADDR_RAFT`, `ADDR_LADDER`, `ADDR_TRIFORCE`)

Every screen id and room id below is **source-hypothesized** unless marked
**(live)**. Live verification is a future probe after L3 Raft.

---

## Gates / required capabilities

| Cap | RAM | Source role |
|-----|-----|-------------|
| **Raft** | `ADDR_RAFT` (`0x0660`) ≠ 0 | Hard gate: dock → island only with Raft from L3 |
| Sword | `ADDR_SWORD` ≥ 1 | Combat |
| Bombs (helpful) | `ADDR_BOMBS` | Optional wall skips / Manhandla |
| Blue Candle (helpful) | `ADDR_CANDLE` | Dark rooms |
| **Stepladder** (dungeon item) | `ADDR_LADDER` (`0x0663`) | Cross water tiles inside L4 (and later OW) |
| Triforce shard 4 | `ADDR_TRIFORCE & 0x08` | Clear stop |

**Predecessor:** Level 3 Manji must drop Raft. Planning-only may document a
**dev inventory poke** of `0x0660` for assisted geometry probes — never for
Clean STATUS or published pure-first evidence.

**Optional OW prep (source):** east-coast Raft Heart Container — from start
east 8, north 4 to dock, walk onto dock (Raft carries north into cave).
Choose Heart over potion.

---

## Overworld

### Hypothesized screens (source hop math)

Overworld id = `(row << 4) | col` (row 0 = north, col 0 = west). Start =
`0x77` (live).

| Landmark | Source path | Hypothesized id | Live? |
|----------|-------------|-----------------|-------|
| East-coast Raft dock (heart) | start E×8 N×4 | **`0x3F`** | no |
| Raft heart cave screen | dock N (auto) | **`0x2F`** | no |
| Lake dock toward L4 island | start U, L×2, U (common short path) | **`0x55`** | no |
| Island after raft (door screen?) | dock N (auto) | **`0x45`** | no |

ZD long path from “previous dungeon” dock (after Raft): left, up, left×6,
down×3, left×3, then raft onto island. Treat as **source only** until a live
trail records the screen sequence; short path above is the usual speed
approach from start once Raft is owned.

**Scaffold:** `level4_overworld.py` — `LEVEL4_DOCK_HOPS` placeholders,
`has_raft()`, `level4_overworld_stop` stubs. Target constants are labeled
`SOURCE_HYPOTHESIS` and must be overwritten after live probe.

### Live recon goals (when free / after Raft)

1. Walk to lake dock screen without Raft (map dock geometry only).
2. Save `OW_L4Dock` if dock screen confirmed.
3. With real Raft: enter island, confirm `level == 4`, mode 5, entry room id.
4. Save `Level4Entrance` + JSON under `recordings/l4_*_recon.json`.

**Do not** poke Raft for Clean claims.

---

## Interior (source speed route)

Room IDs **unknown** until entry probe. Controllers must not hard-code room
ids until live settle.

| Step | Action (source) | Notes |
|------|-----------------|-------|
| Entry | LEFT | 8 Keese → **key** |
| Back E, N | Vires | Wooden sword splits Vire → red Keese; key RIGHT |
| E | Dark maze | Candle; **Compass** |
| Back W, N | key | then LEFT into dark ladder of rooms |
| Dark chain N | keys / water block | North blocked by water until Stepladder |
| E (key) | clear 5 Vire + Keese | open RIGHT; skip useless locked UP |
| E | 2 Like-Like + 2 Zol | push **left block** → stairs → **Stepladder** |
| Back W×2 | ladder over water | key locked north path |
| E | Vires skippable | Map room; optional bomb N rupee room / shortcuts |
| Side path | Manhandla (blocks) | bombs preferred; bomb reward |
| Old Man | “Walk Into The Waterfall” | L5 clue |
| Pre-boss | clear Vires + Keese, push left block | unlock boss door RIGHT |
| Boss | **Gleeok (2 heads)** | fireballs unblockable; detached heads bounce |
| E of boss | center of room | **Triforce shard 4** |

**Key item:** Stepladder (`ADDR_LADDER`).
**Boss:** Gleeok (2-head). Object type id **TBD live**.
**Triforce bit:** `0x08`.

### Policy notes (planning)

- Vire: prefer avoid or accept Keese split; no special B-item.
- Like-Like: stay out of contact (Magical Shield loss).
- Water tiles: after Stepladder, automatic when walking single-tile gaps.
- Gleeok: melee A-spam with movement; no bomb requirement (unlike Dodongo).
- Dark rooms: equip candle on B; one flame per screen with Blue Candle.

---

## Boss / Triforce stop predicates (stubs)

```text
level4_boss_cleared  — TBD: Gleeok absent + room_all_dead / heart drop
level4_complete      — ADDR_TRIFORCE & 0x08  (and mode 18 fanfare settle)
```

Scaffold: `level4_triforce_stop(snap)` returns True only when
`snap.triforce & 0x08` (inventory fact; not a route success claim).

---

## Checkpoints (planned names)

| State | When |
|-------|------|
| `OW_L4Dock` | Dock screen mapped (may lack Raft) |
| `Level4Entrance` | `level==4`, play mode, entry room settled |
| `Level4Stepladder` | after `ADDR_LADDER` |
| `Level4BossCleared` | after Gleeok + HC |
| `Level4Complete` | `triforce & 0x08` |

None exist yet as verified fixtures.

---

## Scaffold / probe

```bash
# Planning dry-run (default): prints caps + hypothesized screens; no emu claim
uv run python zelda_i/scripts/probe_level4_entry.py --plan-only

# Live (requires real Raft in save, or refuse). Optional Survival assist only.
uv run python zelda_i/scripts/probe_level4_entry.py --infinite-life --save-state
```

Module: `zelda_i/level4_overworld.py`.

---

## Evidence

- Source walkthrough only as of planning recon.
- No `recordings/l4_*.json` until live.
- Related RAM already in tree: `ADDR_RAFT`, `ADDR_LADDER`, TF `0x08`.
