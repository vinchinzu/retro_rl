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

## Interior (live pure dual-green — rr-5lu / rr-2ysf 2026-08-09/10)

Module: `level4_dungeon.py`. Runner:
`scripts/run_level4_rooms.py`. Evidence:
`recordings/l4_chain_key_pure_chain_to_key.json` (**2/2 pure** ~1278f),
`recordings/l4_clear50_pure_clear_50.json` (**2/2 pure** ~2478f),
`recordings/l4_keyright62_pure_key_right_62.json` (**2/2 pure** ~1133f),
`recordings/l4_clear62_pure_clear_62.json` (**2/2 pure** ~11536f),
`recordings/l4_compass62_pure_compass_62.json` (**2/2 pure** ~471f,
ADDR_COMPASS|0x08 + return 0x61). Not Clean STATUS promote.

### Live graph (from `Level4Entrance`)

```
0x71 entry (empty combat, item 0x03)
  --UP @ x≈120--> 0x61
0x61: 3× Vire type **0x12** (HP 64) → sword split type **0x1c** (slots 10–12)
  --BOMB_UP stand≈(120,105) face UP--> 0x51
0x51: 8× Keese type **0x1b** (TYPE-only) + RoomItemId **0x19** key (keys 0→1 @~136,149)
  --LEFT @ y≈141--> 0x50
0x50: 5× Vire **0x12**  **DEAD-END pocket** (RIGHT only back to 0x51)
0x51 --DOWN @ x≈120--> 0x61
0x61 --KEY-RIGHT @ y≈141 (keys 1→0)--> 0x62
0x62: 5× Vire + RoomItemId **0x16** Compass (dark maze)  **Stepladder residual**
```

| Room | Live? | Enemies | Item / notes | Segment bead |
|------|-------|---------|--------------|--------------|
| **0x71** | **live pure 2/2** | none | Empty mouth; free UP only | `rr-zchy` |
| **0x61** | **live pure 2/2** | 3× `0x12` → split `0x1c` | Clear ~295f; bomb N → 0x51; KEY-RIGHT → 0x62 | `rr-yr77` / `rr-h278` |
| **0x51** | **live pure 2/2** | 8× `0x1b` Keese | Key `0x19` pickup ~ (136,149) | `rr-wqdu` |
| **0x50** | **live pure 2/2** | 5× `0x12` Vire | Dead-end pocket; no progress exit | `rr-2ysf` |
| **0x62** | **live pure enter+clear+compass 2/2** | 5× `0x12` Vire | Compass `0x16` dark maze; pickup ~(136,132); return LEFT→0x61 | `rr-2ysf` / `rr-9so0` |

### Post-compass residual (rr-o0nn live recon 2026-08-10)

Start: **`Level4Compass`** (0x61, `ADDR_COMPASS|0x08`, keys=0, doors=1 RIGHT).

**Live component is closed** — only rooms `{0x71, 0x61, 0x51, 0x50, 0x62}`:

| From | Exit | Dest | Notes |
|------|------|------|-------|
| 0x61 | free/BOMB UP | 0x51 | hole still open post-compass |
| 0x61 | RIGHT | 0x62 | re-enter without key (door bit stays) |
| 0x61 | DOWN | 0x71 | free |
| 0x51 | LEFT @y141 | 0x50 | free |
| 0x51 | DOWN | 0x61 | free |
| 0x51 | **UP** | **sealed** | not a key door (keys poke does not consume) |
| 0x51 | **RIGHT** | **sealed** | same |
| 0x50 | RIGHT | 0x51 | only exit; denser bomb-N scan no open |
| 0x62 | LEFT | 0x61 | only durable exit; bomb stands no open |

Also live-negative: Vire re-clear key farm (8 cycles) **no drops**; candle already 1 on fixtures (not the gate).

**ADDR_LADDER still 0.** Source walkthrough continues dark-chain N / KEY-RIGHT to Like-Like stairs — **no live room id yet** outside this component. Evidence: `recordings/l4_o0nn_*.json`.

### Runner

```bash
# Pure dual-green room segments (no --infinite-life)
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment entry_up --trials 2
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_61 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment bomb_61 --from-state Level4Entrance --trials 2
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment key_51 --from-state Level4Entrance --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment chain_to_key --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_50 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment key_right_62 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment chain_to_62 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment clear_62 --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level4_rooms.py --segment compass_62 --trials 2 --save-state
```

**Traps (live):**

- Source “entry LEFT Keese key” is **wrong** on this seed/path — entry is empty; first key is bomb-N of Vires.
- **0x50 west of first key is a dead-end pocket** — not the Stepladder path.
  Compass is KEY-RIGHT 0x61→0x62; post-compass component stays closed (rr-o0nn).
- Vire split is type **`0x1c`**, not standard Keese `0x1b`; HP stays 0 (type-only) and lands in slots **10–12**.
- Free doorways often show `cur_opened_doors=0` / `open_doorway_mask=0` — do not require door bits for UP 0x71→0x61 or LEFT 0x51→0x50.
- Bomb stand on 0x61: **(120, ~105)** face UP + B; wait blast then push UP.
- Key item id is **0x19 from room entry** (not drop-after-clear); walk mid-room after Keese clear.
- KEY-RIGHT 0x61: hold **y≈141** RIGHT; keys 1→0; vestibule enter ~(16,141).
- 0x62 **dark maze**: open seek fails; use scripted holds
  (`MAZE_62_TO_COMPASS` hold6 → pickup ~(136,132) sets `ADDR_COMPASS|0x08`,
  then `MAZE_62_RETURN_WEST` hold4 → LEFT scroll to 0x61). Center y=141 is
  wall-blocked from the west door — must follow the return corridor.
- Post-compass: only durable exit found is **LEFT→0x61**. No free/bomb east
  exit from 0x62. Stepladder residual is **outside the closed early component**
  (source: dark-chain N / Like-Like stairs) — 0x51 UP is **sealed**, not the
  live next room.
- From `Level4Compass`: KEY-RIGHT door stays open (RIGHT re-enter 0x62, no key).

Checkpoints (dev): `Level4Room61`, `Level4Room61Cleared`, `Level4FirstKey`,
`Level4Room50Cleared`, `Level4Room62`, `Level4Room62Cleared`, `Level4Compass`.

### Source speed route (planning only past compass — not emulator facts)

Room IDs **beyond 0x62 maze** remain source-hypothesized until probed.

| Step | Action (source) | Notes |
|------|-----------------|-------|
| Past first key | KEY-RIGHT dark maze | **live** 0x62 Compass |
| Dark chain N | water block | needs Stepladder |
| Like-Like + Zol | push left block → stairs | **Stepladder** |
| Gleeok (2 heads) | fireballs | E → TF `0x08` |

**Key item:** Stepladder (`ADDR_LADDER`).
**Boss:** Gleeok (2-head). Object type id **TBD live**.
**Triforce bit:** `0x08`.

### Policy notes

- Vire: wooden sword splits → `0x1c`; clear both generations.
- Keese 0x51: TYPE-only liveness (HP stays 0).
- Like-Like (later): stay out of contact (Magical Shield loss).
- Water tiles: after Stepladder, automatic on single-tile gaps.
- Gleeok: melee A-spam; no bomb requirement (unlike Dodongo).

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
| `Level4FirstKey` | 0x51 keys≥1 after Keese clear | **live** |
| `Level4Room50Cleared` | 0x50 Vires clear (dead-end) | **live** |
| `Level4Room62` | KEY-RIGHT enter 0x62 vestibule | **live** |
| `Level4Room62Cleared` | 0x62 Vires clear (compass residual) | **live** |
| `Level4Compass` | after `ADDR_COMPASS & 0x08` | partial / residual |
| `Level4Stepladder` | after `ADDR_LADDER` | planned |
| `Level4BossCleared` | after Gleeok + HC | planned |
| `Level4Complete` | `triforce & 0x08` | planned |

---

## Evidence

- `recordings/l4_entry_recon.json` — **2/2 assisted** entry from `Level3Complete`
- Checkpoints `Level4Entrance`, `OW_L4Dock`, `Level3ExitOverworld` (+ provenance)
- Related RAM: `ADDR_RAFT`, `ADDR_LADDER`, TF `0x08`
- **Not Clean STATUS**
