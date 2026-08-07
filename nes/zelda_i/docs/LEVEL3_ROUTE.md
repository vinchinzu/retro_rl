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

### Post-L2 (Moon TF) → Manji (assisted LIVE, rr-rnx, 2026-08-07)

After L2 `tf&0x02`, fanfare settles to OW **0x3C** ~(112,125)
(`Level2ExitOverworld`). Reverse Moon door corridor + reverse 0x5C maze, then
west-forest join to door:

```
0x3C S → 0x4C E@y∈[133,145] → 0x4D S@x≈52 → 0x5D W → 0x5C
  [reverse maze] → 0x5B leave bush → 0x5A … → 0x55 S → 0x65
  W → 0x64 W@y≈125–150 → 0x63 S → 0x73 E → 0x74 UP → 0x7c
```

**2/2 assisted** evidence: `recordings/l2_to_l3_assisted.json`.
Runner: `scripts/run_l2_to_l3.py --infinite-life --from-state Level2ExitOverworld`.
Hops: `LEVEL3_HOPS_FROM_POST_L2` / `OverworldPostL2ToLevel3Controller`.

**Traps:**

| Screen | Trap |
|--------|------|
| 0x4C east | **y∈[133,145] only** — y=149 solid forever |
| 0x5C reverse | denser channel waypoints; **no y_band** on 0x5B hop |
| 0x64 west | band **y≈125–150** (y≈109 wall-hug sticks) |

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
| **0x5b** north of 0x6b | **3× Darknut type `0x0b` HP64** | bombs drop on clear | S→0x6b; **N→0x4b** open; **W→0x5a** open; E walk sealed; bomb-R→0x5c | **live** (arrival pure; combat residual) |
| **0x4b** north of 0x5b | **3× Zol type `0x13`** | RoomItemId `0x19` (pickup residual) | S→0x5b; **L KEY→0x4a**; **R KEY→0x4c** map; U blocked | **live** doors; clear spec encoded |
| **0x5a** west of 0x5b | **4× Keese `0x1b`** + 4× blade traps `0x49` | **Compass RoomItemId `0x16`** (inventory bit L3=4) | R→0x5b; **L KEY→0x59**; U→0x4a; D blocked | **live assisted** Compass |
| **0x59** west of compass | **5× Darknut `0x0b`** | — | R→0x5a; **D kill-clear→0x69**; U→0x49 | **live assisted** |
| **0x69** south of 0x59 | **8× Darknut `0x0b`** | — | U→0x59; **R stairs @ y≈141→0x0f** mode 9 | **live assisted** |
| **0x0f** underworld | 4× Keese (HP residual) | **Raft RoomItemId `0x0c`** → `ADDR_RAFT` | mode-9 passage (not cardinal doors) | **live assisted** Raft |
| **0x5c** bomb-R of 0x5b | 3× Darknut `0x0b` | item 0x03 | clear → doors R\|L; **R@y≈141→0x5d**; UP→0x4c | **live** bomb + clear |
| **0x5d** east of 0x5c | 2× Zol + 3× Keese + 3× invuln `0x2b` | item residual | UP→**0x4d** boss (flaky gate residual) | **live** doors residual |
| **0x4d** north of 0x5d | **Manhandla candidate type `0x3c`** (5 slots) + `0x56` proj | HC / TF residual | post-kill residual | **glimpse** assisted (not beaten) |
| **0x4c** north of 0x5c / east of 0x4b | 2× Zol + blade traps | **Map RoomItemId `0x17`** | L KEY→0x4b; from 0x5c UP | **live** |
| **0x49** north of 0x59 | 2× Zol + 3× Keese + 3× `0x2b` invuln | key `0x19` | R→0x4a; S→0x59 | **live** (false-boss trap) |

### Live room graph (isolated pure + assisted past-5b)

```
0x7c (entry, ~(120,205))
  -- west: y≈149 band → wall x≤48 → LEFT+UP diagonal -->
0x7b (6× Zol 0x13, key 0x19)  keys 0→1 after clear+pickup
  -- north: UP @ x≈120 (|dx|≤4; wider align sticks at x≈112) -->
0x6b (5× Zol 0x13, diagonal blocks; RoomItemId 0x19 residual)
  -- north after type-0x13 clear: free-explore grid + UP @ x≈120 -->
0x5b (3× Darknut 0x0b HP64)  checkpoint Level3Darknuts
  -- north OPEN (no clear) -->
0x4b (3× Zol + key 0x19)  -- LEFT KEY --> 0x4a (5 Keese)
                          -- RIGHT KEY --> 0x4c (map 0x17)
  -- west OPEN -->
0x5a (4 Keese + traps, COMPASS 0x16)   *** Raft route ***
  -- LEFT KEY (y≈141 long push) -->
0x59 (5 Darknut; kill opens DOWN bit)
  -- DOWN after clear -->
0x69 (8 Darknut)
  -- RIGHT @ y≈141 only (other y blocked) -->
0x0f mode=9 underworld passage
  -- DOWN y≈189 → RIGHT x≈176 → UP channel → LEFT x≈136 -->
  RAFT (ADDR_RAFT=1)   checkpoint Level3Raft (assisted)
  -- bomb-RIGHT 0x5b @(192,141) --> 0x5c (3× Darknut)
      -- clear → RIGHT @ y≈141 --> 0x5d --> UP residual --> 0x4d Manhandla?
      -- UP --> 0x4c Map
```

### Post-Raft → boss shortcut (assisted LIVE, 2026-08-07)

From `Level3Raft` (mode 9, `ADDR_RAFT=1`, room 0x0f):

```
0x0f reverse channel + NW stairs UP → 0x69
  UP → 0x59
  BOMB_RIGHT @(192,141) → 0x5a     *** walk-RIGHT sealed ***
  RIGHT → 0x5b
  BOMB_RIGHT @(192,141) → 0x5c
  full Darknut clear (doors raw=3) → RIGHT @ y≈141 → 0x5d
  UP → 0x4d Manhandla candidate 0x3c (gate residual)
```

Probe: `scripts/probe_level3_manhandla.py --infinite-life --tag l3_manhandla`.
Evidence: `recordings/l3_manhandla_recon.json`, `l3_manhandla_shortcut.json`,
`l3_manhandla_map_explore.json`.

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

| Field | Value | Evidence |
|-------|-------|----------|
| Boss | **Manhandla** (bombs preferred) | source + LIVE type candidate |
| Boss room | **`0x4d`** (candidate) | assisted glimpse via 0x5d UP |
| Boss object type | **`0x3c`** (candidate; 5 slots + `0x56` proj) | `l3_manhandla_map_explore` |
| False boss | type **`0x2b`** HP240 invuln on 0x49/0x5d | sword/bomb no dmg |
| Prep room | **`0x5d`** east of 0x5c | LIVE RIGHT@y141 after 0x5c clear |
| Item | Raft (`ADDR_RAFT=0x0660`) | LIVE assisted in 0x0f |
| Triforce bit | **`0x04`** | not yet collected assisted |
| Constants stub | `ROOM_L3_BOSS`, `MANHANDLA_OBJECT_TYPE` | `level3_dungeon.py` — **not pure green** |

## Checkpoints

| State | Provenance |
|-------|------------|
| `Level3Entrance.state` | Assisted enter 2026-08-06; `level==3` room **0x7c** ~(120, 205); Survival health poke only |
| `Level3WestKey.state` | Clean isolated pure from Level3Entrance; room **0x7b**, keys≥1, Zols dead |
| `Level3Darknuts.state` | Clean isolated pure from Level3WestKey; room **0x5b**, 3× Darknut 0x0b |
| `Level3Raft.state` | Assisted Survival from Level3Darknuts via Compass west path; `ADDR_RAFT≠0` in 0x0f |

## Residual toward Raft / boss (after assisted LIVE map)

1. **Assisted Raft runner 2/2 LIVE** (Survival) from `Level3Darknuts` → Compass
   west → 0x59/0x69 → stairs → passage → `ADDR_RAFT`. Checkpoint
   `Level3Raft.state`. **Not Clean STATUS.**
   - Module: `Level3RaftPathController` in `level3_dungeon.py`
   - Runner: `uv run python nes/zelda_i/scripts/run_level3_raft.py --infinite-life --trials 2 --save-state`
   - Evidence: `recordings/level3_raft_assisted.json` (**2/2 assisted**, ~6448f/trial)
2. **0x5b Darknut clear pure** — side/back hits; combat residual.
3. **0x4b Zol clear** — spec `ROOM_4B_SPEC` + `run_level3_clear4b.py` (try 2/2).
4. **0x6b key pickup** residual (inventory may not increment).
5. **Manhandla + TF `0x04`** — path to 0x5d LIVE; room **0x4d** / type **0x3c**
   candidate; kill + TF residual (not yet assisted).
6. **Natural-entry** from real predecessor before Clean STATUS promote.

### Traps burned (past-5b + Raft→boss)

| Room | Trap |
|------|------|
| 0x5a LEFT key | Key can **spend without scroll** if y≠141 / short push — need long y=141 LEFT |
| 0x59 / 0x69 | Darknuts **spawn delay** ~75–100f; clear too early → doors stay closed |
| 0x59 DOWN | After live=0, **DOWN bit lags ~40f** (`room_all_dead` ramp); wait for doors&4 |
| 0x59 DOWN push | Align **x≈120** and hold DOWN; chasing y=205 thrash-oscillates on south wall |
| 0x69 RIGHT stairs | Only **y≈141** works; other y bands max-x stick without room change |
| 0x0f passage | South band **UP blocked** except **x≈176** channel; Raft touch ~**(136,141)** after channel |
| 0x0f mid-band | Once on y≈141 corridor, **LEFT to x≈136** — do not re-south if x drifts off channel |
| 0x0f exit | Reverse channel then **NW stairs hold UP** → 0x69 (Level3Raft backtrack) |
| 0x5b LEFT plane | West wall sits at **x≈26** (not 32); push LEFT once x≤48 (do not snap back to x=32) |
| 0x5b RIGHT | Walk sealed; bomb stand **(192,141)** opens 0x5c (recon poke OK) |
| 0x59 RIGHT post-Raft | **Walk sealed** despite door bit; **BOMB_RIGHT @(192,141)** reopens 0x5a |
| 0x5c RIGHT | Only **y≈141** after full Darknut clear (doors raw=3); incomplete clear seals RIGHT |
| 0x5d UP | Boss gate **flaky** (doors often 0); type `0x2b` invuln ≠ Manhandla |
| 0x2b | HP240 invulnerable mover on 0x49/0x5d — ignore for boss fight |

## Evidence

- `recordings/l3_recon.json` — door/entry facts, path notes
- `recordings/level3_west_key_isolated.json` — Clean west-key pure trials
- `recordings/level3_north_chain_isolated.json` — Clean north-chain 0x7b→0x5b
- `recordings/l3_past_5b_recon.json` — **LIVE doors from 0x5b** + compass path hops
- `recordings/l3_raft_recon.json` — **assisted Raft pickup** (`ADDR_RAFT=1`) recon
- `recordings/level3_raft_assisted.json` — **2/2 durable runner** from Level3Darknuts
- `recordings/l3_manhandla_recon.json` — **Raft→boss path** + traps (TF not yet)
- `recordings/l3_manhandla_shortcut.json` / `l3_manhandla_map_explore.json` — 0x5c/0x5d/0x4d
- `recordings/l3_westkey_probe_report.json` — door probes from Level3WestKey
- `recordings/l3_5b_spawn.png` / `l3_north_up_x120.png` — room visuals
- Probe: `uv run python nes/zelda_i/scripts/probe_level3_entry.py --infinite-life --from-state OW_66 --save-state`
- Past Darknuts: `uv run python nes/zelda_i/scripts/probe_level3_past_darknuts.py --infinite-life --tag l3_past_5b`
- Manhandla: `uv run python nes/zelda_i/scripts/probe_level3_manhandla.py --infinite-life --tag l3_manhandla`
- Map-only: `… --from-state Level3Entrance --map-only --infinite-life`
- West key: `uv run python nes/zelda_i/scripts/run_level3_west_key.py --trials 2 --save-state`
- North chain: `uv run python nes/zelda_i/scripts/run_level3_north_chain.py --trials 2 --save-state`
- Raft (assisted): `uv run python nes/zelda_i/scripts/run_level3_raft.py --infinite-life --trials 2 --save-state`
- 0x4b clear: `uv run python nes/zelda_i/scripts/run_level3_clear4b.py --trials 2`

## Sources

- [Zelda Dungeon — Level 3: The Manji](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-3-the-manji/)
- Local: `docs/research/DUNGEON_WALKTHROUGHS.md`, `docs/tasks/PARALLEL_RECON.md`
