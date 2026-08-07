# Level 3 — Manji (route notes)

Status: **assisted-entry** (not Clean STATUS)

Assist track only for overworld entry (`UnlimitedHealthAssist` /
`--infinite-life`). Interior pure segments below are **Clean isolated** where
noted. Do not promote natural-entry or Clean gates from this doc alone.

## Overworld

| Field | Value | Evidence |
|-------|-------|----------|
| Door screen | **`0x74`** | **live** — exit spawn from entry; re-enter level==3 |
| Door approach | UP @ **x≈128**, approach from **y≳130** | live exit spawn (128, 125) |
| Entry room | **`0x7c`** | **live** `level==3` mode 5 |
| Checkpoint | `Level3Entrance.state` | `custom_integrations/LegendOfZelda-Nes/` |

### Path from start / post-sword

**Source** (Zelda Dungeon — *not* walkable as stated):

```
From start: up, left 4, down, right 1
→ screens 0x77 → 0x67 → 0x66 → 0x65 → 0x64 → 0x63 → 0x73 → 0x74
```

**Live blocker:** screen **0x67** is a fully enclosed tree pocket (enterable
from 0x77 north) with **no west exit**. Source hop path fails after the first
screen.

**Live prefix** (assisted; pieces verified 2026-08-06):

```
0x77 E@y≈140 → 0x78 N@x≈48 → 0x68 N → 0x58 W@y≈155
→ 0x57 W → 0x56 W@y≈133 → 0x55 S → 0x65
```

**Live door suffix** (from `OW_66` or 0x65; assisted enter succeeded):

```
0x66/0x65 W@y≈141 → 0x64 → 0x63 S → 0x73 E@y≈117 → 0x74
→ tour/UP hunt → level 3 room 0x7c
```

Code: `zelda_i.level3_overworld` (`LEVEL3_PATH_HOPS`, `LEVEL3_DOOR_HOPS_FROM_66`).

Required items to *enter*: wooden sword (potion recommended by walkthrough;
not required for assisted entry).

## Interior (source → live)

| Room id | Enemies | Key/item | Doors | Status |
|---------|---------|----------|-------|--------|
| **0x7c** entry | Static orange sprites (not live RAM types); `obj_count` residual | — | S mouth (exit OW), **W** (corner residual) | **live** entry |
| **0x7b** west of entry | **6× Zol type `0x13`** (HP>0; wooden sword can leave type-0 HP residual) | key **RoomItemId `0x19`** | E back to 0x7c; **N** to 0x6b | **live pure Clean** |
| **0x6b** north of 0x7b | **5× Zol type `0x13`** on **diagonal raised blocks** | RoomItemId `0x19` (key drop **residual** — RoomAllDead often stalls) | S→0x7b; **N→0x5b** after type-0x13 clear | **live pure Clean** (clear + north) |
| **0x5b** north of 0x6b | **3× Darknut type `0x0b` HP64** | — | S→0x6b; **N→0x4b**; E/W blocked | **live graph** (arrival pure; combat residual) |
| **0x4b** north of 0x5b | **3× Zol type `0x13`** | RoomItemId `0x19` | S→0x5b | **graph only** (probe) |
| Further | Darknuts, bombs, Compass, Raft path… | bombs / Raft | — | source only |

### Live room graph (isolated pure)

```
0x7c (entry, ~(120,205))
  -- west: y≈149 band → wall x≤48 → LEFT+UP diagonal -->
0x7b (6× Zol 0x13, key 0x19)  keys 0→1 after clear+pickup
  -- north: UP @ x≈120 (|dx|≤4; wider align sticks at x≈112) -->
0x6b (5× Zol 0x13, diagonal blocks; RoomItemId 0x19 residual)
  -- north after type-0x13 clear: free-explore grid + UP @ x≈120 -->
0x5b (3× Darknut 0x0b HP64)  checkpoint Level3Darknuts
  -- north (open) -->
0x4b (3× Zol + key)  graph only
```

West door residual (fixed for pure): pure **LEFT** sticks at **x≈32**
(`open_doorway_mask==0`, solid door tiles). **LEFT+UP** at the west wall
corner-clips into scroll → room **0x7b**. Approach on **y≈149** (y≈141 alone
often blocks mid-room at x≈112).

North door residual from **0x7b**: **UP** only with **|x−120|≤4**. Align
threshold 8 leaves Link at **x≈112** stuck on the north wall. RIGHT/LEFT/DOWN
from cleared 0x7b do not open (mask==0).

**0x6b geometry:** diagonal raised blocks partition the floor. After Zol clear,
RoomAllDead often stays 0 (type-0 HP leftovers from wooden-sword hits — not
killable as type 0x13). Pure clear uses **type-0x13 liveness only**
(`settle_all_dead=0`). Source key drop not yet reliably collected; north
shutter/door opens for UP once type-0x13 are gone (keys inventory unchanged
in live trials). North exit needs **grid hunt** (not a single waypoint snake).

### Source interior (Zelda Dungeon L3)

- Entry → **LEFT** → Zols (split to Gel with wooden sword) + key → UP → Zols + key → UP
- Darknuts (side/back hits); bombs reward; bomb RIGHT = boss shortcut
- LEFT → Keese + **Compass** → key LEFT → Darknuts → DOWN
- Staircase → Keese path → **Raft**
- Backtrack toward boss: Bubbles, Keese, Zols → UP **Manhandla** (bombs best)
- Heart Container → Triforce shard 3

### Live pure segments (Clean isolated)

#### West key (0x7c → 0x7b)

- Module: `zelda_i.level3_dungeon` (`ROOM_7B_SPEC`, `Level3WestKeyController`)
- Runner: `uv run python nes/zelda_i/scripts/run_level3_west_key.py --trials 2`
- Stop: `level3_room_7b_key_success` (keys≥1, no live Zols, room 0x7b)
- Checkpoint: `Level3WestKey.state` (`--save-state`)
- Evidence: `recordings/level3_west_key_isolated.json` (3/3 Clean lab; door ~319f + combat ~658f)
- Intervention: **Clean**

#### North chain (0x7b → 0x6b → 0x5b) — rr-65w

- Module: `zelda_i.level3_dungeon` (`ROOM_6B_SPEC`, `ROOM_5B_SPEC`,
  `Level3NorthChainController`)
- Runner: `uv run python nes/zelda_i/scripts/run_level3_north_chain.py --trials 2`
- Stop: `level3_reached_5b` (play mode in room **0x5b**)
- Checkpoint: `Level3Darknuts.state` (`--save-state`)
- Evidence: `recordings/level3_north_chain_isolated.json` (**2/2 Clean**;
  door ~275f + combat ~1100f + north exit ~2700f ≈ 4133f)
- Intervention: **Clean** (no health/inventory poke on this segment)
- Recon assist: `--infinite-life` optional if deaths block earlier segments

## Boss / Triforce

| Field | Value |
|-------|-------|
| Boss | Manhandla (bombs preferred) |
| Item | Raft (`ADDR_RAFT=0x0660`) |
| Triforce bit | **`0x04`** |

## Checkpoints

| State | Provenance |
|-------|------------|
| `Level3Entrance.state` | Assisted enter 2026-08-06; `level==3` room **0x7c** ~(120, 205); Survival health poke only |
| `Level3WestKey.state` | Clean isolated pure from Level3Entrance; room **0x7b**, keys≥1, Zols dead |
| `Level3Darknuts.state` | Clean isolated pure from Level3WestKey; room **0x5b**, 3× Darknut 0x0b |

## Residual toward Raft (not pure yet)

1. **0x5b Darknut clear** — side/back hits only; bombs reward (source). Spec
   scaffolded as `ROOM_5B_SPEC` (combat not 2/2 pure).
2. **0x4b** north — 3× Zol + key (live probe); then source Darknut/Compass/
   staircase → **Raft**.
3. **0x6b key pickup** — RoomItemId 0x19 observed; inventory keys did not
   increment in live trials (RoomAllDead residual). North progress does not
   require the extra key.
4. **Natural-entry** from real predecessor still required before Clean STATUS
   promote.

## Evidence

- `recordings/l3_recon.json` — door/entry facts, path notes
- `recordings/level3_west_key_isolated.json` — Clean west-key pure trials
- `recordings/level3_north_chain_isolated.json` — Clean north-chain 0x7b→0x5b
- `recordings/l3_westkey_probe_report.json` — door probes from Level3WestKey
- `recordings/l3_5b_spawn.png` / `l3_north_up_x120.png` — room visuals
- Probe: `uv run python nes/zelda_i/scripts/probe_level3_entry.py --infinite-life --from-state OW_66 --save-state`
- Map-only: `… --from-state Level3Entrance --map-only --infinite-life`
- West key: `uv run python nes/zelda_i/scripts/run_level3_west_key.py --trials 2 --save-state`
- North chain: `uv run python nes/zelda_i/scripts/run_level3_north_chain.py --trials 2 --save-state`

## Sources

- [Zelda Dungeon — Level 3: The Manji](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-3-the-manji/)
- Local: `docs/research/DUNGEON_WALKTHROUGHS.md`, `docs/tasks/PARALLEL_RECON.md`
