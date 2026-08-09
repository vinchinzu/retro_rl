# Level 4 — The Snake (route notes)

**Status:** OW entry **live (assisted)**. Interior still planning. Do not
claim Clean STATUS — Survival assist only for this segment.

**Beads:** `rr-0fx` Z4.1 live entry (done); `rr-5lu` interior residual;
epic `rr-q3n`.

Planning sources (external, not emulator facts):

- [Zelda Dungeon — Level 4: The Snake](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-4-the-snake/)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)
- RAM: `zelda_i/ram.py` (`ADDR_RAFT`, `ADDR_LADDER`, `ADDR_TRIFORCE`)

Every screen id and room id is **source-hypothesized** unless marked
**(live)**.

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

**Predecessor:** Level 3 Manji drops Raft. **Do not** poke `0x0660` for
Clean STATUS or published pure-first evidence.

**Optional OW prep (source):** east-coast Raft Heart Container — from start
east 8, north 4 to dock, walk onto dock (Raft carries north into cave).
Choose Heart over potion.

---

## Overworld

### Live path (assisted 2026-08-08, rr-0fx)

Start: checkpoint **`Level3Complete`** (mode 18, room 0x3d, `raft=1`,
`tf&0x04`). Fanfare settles to OW **`0x74`** ~(128,125).

```
0x74 W@y141 → 0x73 free mid x≈128 → N → 0x63 free south
  → E@y≈145–155 → 0x64 E@y141 → 0x65 N@x112 → dock 0x55
  → N@x≈128 (Raft) → island 0x45 → door UP @x128 → level 4 room 0x71
```

| Landmark | Id | Live? | Notes |
|----------|-----|-------|-------|
| Post-L3 return | **`0x74`** | **live** | Same as L3 door mouth |
| Raft dock | **`0x55`** | **live** | South entry y=221; raft only x≈128 |
| Island door | **`0x45`** | **live** | UP into dungeon |
| Entry room | **`0x71`** | **live** | South mouth ~(120,205) mode 5 |
| East-coast Raft heart dock | `0x3F` | no | Source only |
| Raft heart cave | `0x2F` | no | Source only |

**Traps (live):**

- `0x73`: arrive east edge from 0x74 — free mid before UP.
- `0x63` east: **y∈[145,155]** only. y=141 sticks in bush near x≈144.
- Dock `0x55`: UP at x≤112 never boards; align **x≈128** then UP.
- Do not poke Raft. Not Clean STATUS.

**Module:** `level4_overworld.py` — `LEVEL4_HOPS_FROM_POST_L3`,
`OverworldToLevel4Controller`, `PostL3TriforceSettleController`.
`SOURCE_HYPOTHESIS = False`.

### Runner

```bash
# 2/2 assisted entry + Level4Entrance.state
uv run python nes/zelda_i/scripts/run_level4_entry.py --infinite-life --trials 2 --save-state

# Dock only → OW_L4Dock.state
uv run python nes/zelda_i/scripts/run_level4_entry.py --infinite-life --dock-only --save-state

# Plan dry-run
uv run python nes/zelda_i/scripts/probe_level4_entry.py --plan-only
```

Evidence: `recordings/l4_entry_recon.json` (**2/2 assisted**, ~2173f/trial).
Checkpoints: **`Level3ExitOverworld`**, **`OW_L4Dock`**, **`Level4Entrance`**.

---

## Interior (source speed route)

Room IDs beyond entry **0x71** unknown until interior probe (`rr-5lu`).

| Step | Action (source) | Notes |
|------|-----------------|-------|
| Entry 0x71 | LEFT | 8 Keese → **key** |
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

## Checkpoints

| State | When | Status |
|-------|------|--------|
| `Level3ExitOverworld` | Post-L3 fanfare settle OW 0x74 raft=1 | **live** |
| `OW_L4Dock` | Dock screen 0x55 | **live** |
| `Level4Entrance` | `level==4`, play mode, room 0x71 | **live** |
| `Level4Stepladder` | after `ADDR_LADDER` | planned |
| `Level4BossCleared` | after Gleeok + HC | planned |
| `Level4Complete` | `triforce & 0x08` | planned |

---

## Evidence

- `recordings/l4_entry_recon.json` — **2/2 assisted** entry from `Level3Complete`
- Checkpoints `Level4Entrance`, `OW_L4Dock`, `Level3ExitOverworld` (+ provenance)
- Related RAM: `ADDR_RAFT`, `ADDR_LADDER`, TF `0x08`
- **Not Clean STATUS**
