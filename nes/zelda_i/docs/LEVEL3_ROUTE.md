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

Code: `zelda_i.level3.overworld` (`LEVEL3_PATH_HOPS`, `LEVEL3_DOOR_HOPS_FROM_66`).

Required items to *enter*: wooden sword (potion recommended by walkthrough;
not required for assisted entry).

## Interior (source → live)

| Room id | Enemies | Key/item | Doors | Status |
|---------|---------|----------|-------|--------|
| **0x7c** entry | Static orange sprites (not live RAM types); `obj_count` residual | — | S mouth (exit OW), **W** (corner residual) | **live** entry |
| **0x7b** west of entry | **6× Zol type `0x13`** (HP>0; wooden sword can leave type-0 HP residual) | key **RoomItemId `0x19`** | E back to 0x7c; **N** to 0x6b | **live pure Clean** |
| **0x6b** north of 0x7b | **5× Zol type `0x13`** on **diagonal raised blocks** | RoomItemId `0x19` (key drop **residual** — RoomAllDead often stalls) | S→0x7b; **N→0x5b** after type-0x13 clear | occupancy BFS dest (block on miss); isolated north still Clean |
| **0x5b** north of 0x6b | **3× Darknut type `0x0b` HP64** | bombs drop on clear | S→0x6b; **N→0x4b** open; **W→0x5a** open; E walk sealed; bomb-R→0x5c | **live** (arrival pure; combat residual) |
| **0x4b** north of 0x5b | **3× Zol type `0x13`** | RoomItemId `0x19` (pickup residual) | S→0x5b; **L KEY→0x4a**; **R KEY→0x4c** map; U blocked | **live** doors; clear spec encoded |
| **0x5a** west of 0x5b | **4× Keese `0x1b`** + 4× blade traps `0x49` | **Compass RoomItemId `0x16`** (inventory bit L3=4) | R→0x5b; **L KEY→0x59**; U→0x4a; D blocked | **live assisted** Compass |
| **0x59** west of compass | **5× Darknut `0x0b`** | — | R→0x5a; **D kill-clear→0x69**; U→0x49 | **live assisted** |
| **0x69** south of 0x59 | **8× Darknut `0x0b`** | — | U→0x59; **R stairs @ y≈141→0x0f** mode 9 | **live assisted** |
| **0x0f** underworld | 4× Keese (HP residual) | **Raft RoomItemId `0x0c`** → `ADDR_RAFT` | mode-9 passage (not cardinal doors) | **live assisted** Raft |
| **0x5c** bomb-R of 0x5b | 3× Darknut `0x0b` | item 0x03 | clear → doors R\|L; **R@y≈141→0x5d**; UP→0x4c | **live** bomb + clear |
| **0x5d** east of 0x5c | 2× Zol→Gel + Keese + 3× invuln `0x2b` | item residual | clear killables (slots **1–12**) → doors **raw=10** → UP→**0x4d** | **live assisted** |
| **0x4d** north of 0x5d | **Manhandla type `0x3c`** (5 heads HP64) + `0x56` proj | HC `0x1A` mid-room | kill → **UP→0x3d TF** | **live assisted** kill |
| **0x3d** north of boss | TF shard (RoomItemId `0x1B`) | — | touch → `ADDR_TRIFORCE&0x04` (mode 18) | **live assisted** |
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
      -- clear → doors raw=3 → RIGHT@y141 / bomb-R --> 0x5d
      -- clear Zol/Gel/Keese (slots 1–12; ignore 0x2b) → doors raw=10
      -- side_path UP --> 0x4d Manhandla 0x3c → bomb kill → UP 0x3d TF 0x04
      -- UP --> 0x4c Map
```

## Spine attach (`rr-4d53.3`)

Watchable main spine is **one continuous Survival session from power-on**
(`run_survival_spine.py`). Isolated L3 runners and `Level3*.state` pins are
geometry libraries. They are **not** spine approvals and cannot close a
`rr-4d53.3*` bead.

`--through level3` stop name **moves with the claimed tip leaf**. Parent
`rr-4d53.3` stays open until TF bit `0x04` is on that same tape.

### Approval (required to close any `rr-4d53.3*` leaf)

1. Command (one trial, same env as the predecessor stop):
   `uv run python nes/zelda_i/scripts/run_survival_spine.py --through level3 --no-video --trials 1`
2. Predecessor is the previous spine stop on that session (never a loaded
   `Level3Entrance` / `Level3WestKey` / `Level3Darknuts` / `Level3Raft` pin).
3. Exact RAM stop for the leaf (table below). No timeout bump in place of a miss.
4. Report: `continuous_emulator_session=true`, `mid_run_state_load=false`,
   `seamed=false`, `progression_writes=0`, `capacity_writes=0`, `deaths=0`.
5. Bomb/key **count** top-up listed in `inventory_assist` is the only allowed
   poke (`docs/ASSIST_CONTRACT.md`). No door poke, no undiscovered items, no
   `max_bombs` write. `--poke-bombs 16` on `run_level3_to_boss` is recon.
6. Evidence: `recordings/survival_spine.json` + `_final.png`. Report `stop`
   field equals the leaf name. Library stages live in `level3/spine.py` (or a
   `level3/` extract); unit tests cover stage names + stop predicate.
7. Occupancy / door-graph first (`zelda_i.walk.predict`). Halt at the first
   unrecoverable miss; do not hunt.

### Not approval

- Isolated `run_level3_west_key.py` / `run_level3_north_chain.py` /
  `run_level3_raft.py` / `run_level3_to_boss.py` / `run_l2_to_l3.py`
- Any `Level3*.state` load, door poke, or `--poke-bombs`
- Seamed compose / clip concat
- Walkthrough or TAS (hypothesis only)

Room leaves are children of in-progress `rr-4d53.3` so `bd ready -l zelda_i`
shows only the unblocked tip. Corridor parents (`.3.1` / `.3.3` / `.3.4`) are
aggregators blocked on their last child — do not claim a parent instead of
the room leaf.

### DAG (claim one ready leaf)

| Bead | Segment | Spine stop | Library | Isolated (not close) |
|------|---------|------------|---------|----------------------|
| `rr-4d53.3.0` | L2 TF → Manji entry | `level3_entrance_0x7c` | `level3_entry_stages` | `run_l2_to_l3` from `Level2ExitOverworld` |
| `rr-4d53.3.1.1` | **closed** 0x7c west key | `level3_west_key_0x7b` | `level3_west_key_stages` (`Level3WestKeyController`) | `run_level3_west_key.py` |
| `rr-4d53.3.1.2` | **closed** 0x7b occupancy dest | `level3_dest_0x5b` | `level3_dest_6b_stages` north_chain | `run_level3_north_chain.py` |
| `rr-4d53.3.1` | parent: live dest 0x5b | same as `.1.2` | both dest stages | north-chain Clean 2/2 |
| `rr-4d53.3.3.1` | **closed** 0x5b LEFT → Compass | `level3_compass_0x5a` | raft `left_to_5a` | `Level3Raft` pin |
| `rr-4d53.3.3.2` | **closed** 0x5a KEY-LEFT y=141 | `level3_west_darknuts_0x59` | raft `key_to_59` | key-waste recon |
| `rr-4d53.3.3.3` | **closed** 0x59 kill DOWN | `level3_south_darknuts_0x69` | raft `clear_59`/`down_to_69` | spawn-lag recon |
| `rr-4d53.3.3.4` | **closed** 0x69 stairs → Raft | `level3_raft` (`ADDR_RAFT≠0`) | raft `stairs_to_0f`/`passage_raft` | `run_level3_raft.py` |
| `rr-4d53.3.3` | parent: 0x5b → Raft | same as `.3.3.4` | `Level3RaftPathController` | Raft 2/2 assisted |
| `rr-4d53.3.2` | **verified Survival** bomb budget | documented Raft top-up 8→16 | farm deferred | isolated poke is recon |
| `rr-4d53.3.4.1` | **verified** Raft backtrack bomb-R → 0x5b | `level3_backtrack_0x5b` | continuous boss path | walk-RIGHT sealed trap |
| `rr-4d53.3.4.2` | **verified** 0x5b clear + bomb-R → 0x5c raw=3 | `level3_shortcut_0x5c` | continuous boss path | HP=0 Darknuts type-live |
| `rr-4d53.3.4.3` | **verified** 0x5c edge thread → 0x5d raw=10 | `level3_prep_0x5d` | continuous boss path | ignore type `0x2b` |
| `rr-4d53.3.4.4` | **verified** Manhandla → TF `0x04` | `level3_triforce_0x04` | continuous boss path | side path from `(80,141)` |
| `rr-4d53.3.4` | **verified** parent: Raft → TF | same as `.4.4` | `continuous_mode`, restores forbidden | isolated suffix recon |
| `rr-4d53.3` | **verified** parent: L2 exit → L3 TF | `level3_triforce_0x04` | full `--through level3` | `l3_tf_continuous_video_v1` |

West key `0x7b`, dest `0x5b`, Compass `0x5a`, `0x59`, `0x69`, and natural
Raft and the boss suffix are live on the spine. The next claim is L3 exit →
L4. Isolated poke-16 cannot close a spine leaf. Compass /
map rooms that are off the Raft route stay optional and do not block TF.

West-key close (2026-08-21): `l3_west_key_spine.json` 1/1 Survival 54589f,
room `0x7b` keys=5 (entry keys=4), bombs=8, `tf=0x03`, west_key 671f
(door 320f LEFT+UP y≈149 + combat 351f 6× Zol last_live=0). `route/chain.py`
`controller_stage_done` accepts string-phase L3 path controllers (enum
`.phase.name` crashed the first attach). Isolated west-key is not this close.

Dest close (`.3.1.2`, 2026-08-21): `l3_dest_0x5b_v12.json` is 1/1 continuous
Survival, 57256f, room `0x5b`, keys=5, bombs=8, TF=0x03, zero deaths and zero
progression/capacity writes. Combat occupancy_patrol held at 1435f / 5 Zol.
UP (v8), LEFT+UP (v9), LEFT (v10), and DOWN (v11) all failed at the north
diagonal pocket; RIGHT escaped it in v12 and reached `0x5b` in 945 exit frames.

Library `zelda_i.level3.bomb_budget` counts Raft→boss spend: verified bomb-R
0x59 and bomb-R 0x5b (stands `(192,141)`), plus an **assumed** Manhandla-heads
estimate (type `0x3c`, 5 heads live; bombs preferred — not TAS-perfect).
Isolated `Level3Raft` stops at bombs=0 so `run_level3_to_boss --poke-bombs 16`
is recon only. The verified Survival spine reaches Raft with bombs=8, then
records a count top-up 8→16 before the boss suffix; the natural farm remains
deferred. No bomb-capacity or undiscovered-item write is allowed.

### Post-Raft → Manhandla → TF (assisted LIVE **2/2**, 2026-08-07)

From `Level3Raft` (mode 9, `ADDR_RAFT=1`, room 0x0f):

```
0x0f reverse channel + NW stairs UP → 0x69
  UP → 0x59
  BOMB_RIGHT @(192,141) → 0x5a     *** walk-RIGHT sealed ***
  RIGHT → 0x5b
  BOMB_RIGHT @(192,141) → 0x5c
  full Darknut clear (doors raw=3) → RIGHT @ y≈141 (or bomb-R) → 0x5d
  clear Zol/Gel/Keese until only 0x2b → doors raw=10 (U|L)
  side_path UP → 0x4d Manhandla type 0x3c (5 heads HP64 + 0x56)
  bomb heads → HC mid-room → UP → 0x3d → TF bit 0x04
```

Durable runner: `scripts/run_survival_spine.py --through level3 --no-video --trials 1`.
Isolated `run_level3_to_boss.py` pruned. Evidence:
`recordings/level3_to_boss_assisted.json` (**2/2** enter+kill+TF,
~21653f/trial). Checkpoints: `Level3Boss`, `Level3Complete`.
Library: `level3.boss_path.Level3BossPathController`.

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
in live trials). North exit: south-mouth LEFT+UP clip (v5 live), occupancy BFS
with diamond thread on no-path, then UP @ x≈120 on the north band. Residual:
`(112,117)` pocket (UP / LEFT+UP / LEFT no-op; next is DOWN).

### Source interior (Zelda Dungeon L3)

- Entry → **LEFT** → Zols (split to Gel with wooden sword) + key → UP → Zols + key → UP
- Darknuts (side/back hits); bombs reward; bomb RIGHT = boss shortcut
- LEFT → Keese + **Compass** → key LEFT → Darknuts → DOWN
- Staircase → Keese path → **Raft**
- Backtrack toward boss: Bubbles, Keese, Zols → UP **Manhandla** (bombs best)
- Heart Container → Triforce shard 3

### Live pure segments (Clean isolated)

#### West key (0x7c → 0x7b)

- Module: `zelda_i.level3.dungeon` (`ROOM_7B_SPEC`, `Level3WestKeyController`)
- Isolated segment CLI pruned. Spine dest 0x5b (includes west key):
  `uv run python nes/zelda_i/scripts/run_survival_spine.py --through l3-dest-6b --no-video --trials 1`
- Stop: `level3_room_7b_key_success` (keys≥1, no live Zols, room 0x7b)
- Checkpoint: `Level3WestKey.state` (`--save-state`)
- Evidence: `recordings/level3_west_key_isolated.json` (3/3 Clean lab; door ~319f + combat ~658f)
- Intervention: **Clean**

#### North chain (0x7b → 0x6b → 0x5b) — rr-65w

- Module: `zelda_i.level3.dungeon` (`ROOM_6B_SPEC`, `ROOM_5B_SPEC`,
  `Level3NorthChainController`)
- Isolated segment CLI pruned. Spine dest 0x5b:
  `uv run python nes/zelda_i/scripts/run_survival_spine.py --through l3-dest-6b --no-video --trials 1`
- Stop: `level3_reached_5b` (play mode in room **0x5b**)
- Checkpoint: `Level3Darknuts.state` (`--save-state`)
- Evidence: `recordings/level3_north_chain_isolated.json` (**2/2 Clean**;
  door ~275f + combat ~1100f + north exit ~2700f ≈ 4133f)
- Intervention: **Clean** (no health/inventory poke on this segment)
- Recon assist: `--infinite-life` optional if deaths block earlier segments

## Boss / Triforce

| Field | Value | Evidence |
|-------|-------|----------|
| Boss | **Manhandla** (bombs preferred) | LIVE kill **2/2 assisted** |
| Boss room | **`0x4d`** | `level3_to_boss_assisted` |
| Boss object type | **`0x3c`** (5 heads HP64 + `0x56` proj) | LIVE + HP drop under bomb |
| False boss | type **`0x2b`** HP240 invuln on 0x49/0x5d | sword/bomb no dmg |
| Prep room | **`0x5d`** east of 0x5c | clear → doors raw=10 → UP |
| TF room | **`0x3d`** north of boss | RoomItemId `0x1B`; UP after kill |
| Item | Raft (`ADDR_RAFT=0x0660`) | LIVE assisted in 0x0f |
| Triforce bit | **`0x04`** | **2/2 assisted** collected |
| Constants | `ROOM_L3_BOSS`, `MANHANDLA_OBJECT_TYPE` | `level3/dungeon.py` — assisted only |

## Checkpoints

| State | Provenance |
|-------|------------|
| `Level3Entrance.state` | Assisted enter 2026-08-06; `level==3` room **0x7c** ~(120, 205); Survival health poke only |
| `Level3WestKey.state` | Clean isolated pure from Level3Entrance; room **0x7b**, keys≥1, Zols dead |
| `Level3Darknuts.state` | Clean isolated pure from Level3WestKey; room **0x5b**, 3× Darknut 0x0b |
| `Level3Raft.state` | Assisted Survival from Level3Darknuts via Compass west path; `ADDR_RAFT≠0` in 0x0f |
| `Level3Boss.state` | Assisted path to 0x4d Manhandla live (from Level3Raft) |
| `Level3Complete.state` | Assisted Manhandla kill + TF bit `0x04` in 0x3d |

## Residual toward Raft / boss (after assisted LIVE map)

1. **Assisted Raft runner 2/2 LIVE** (Survival) from `Level3Darknuts` → Compass
   west → 0x59/0x69 → stairs → passage → `ADDR_RAFT`. Checkpoint
   `Level3Raft.state`. **Not Clean STATUS.**
   - Module: `Level3RaftPathController` in `level3/raft_path.py`
   - Isolated segment CLI pruned. Durable runner:
     `uv run python nes/zelda_i/scripts/run_survival_spine.py --through l3-raft --no-video --trials 1`
   - Evidence: `recordings/level3_raft_assisted.json` (**2/2 assisted**, ~6448f/trial)
2. **Assisted Manhandla + TF `0x04` 2/2 LIVE** from `Level3Raft` (Survival).
   Checkpoint `Level3Complete.state`. **Not Clean STATUS.**
   - Isolated segment CLI pruned. Durable runner:
     `uv run python nes/zelda_i/scripts/run_survival_spine.py --through level3 --no-video --trials 1`
   - Evidence: `recordings/level3_to_boss_assisted.json` (**2/2** enter 0x4d + kill + TF)
3. **0x5b Darknut clear pure** — side/back hits; combat residual (off the
   Raft skip; not a spine blocker).
4. **0x4b Zol clear** — spec `ROOM_4B_SPEC` (optional, not on the Raft TF
   route). Isolated `run_level3_clear4b.py` pruned.
5. **0x6b key pickup** residual (inventory may not increment).
6. **Natural-entry / Clean STATUS** only after the continuous spine TF `0x04`
   tape exists. Isolated pins stay development evidence.

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
| 0x5c RIGHT | Need **doors raw=3** after full Darknut clear; raw=1 false-clear seals walk-RIGHT; bomb-R fallback OK |
| 0x5d UP | Clear **Zol+Gel+Keese slots 1–12** (gel in slot 11 seals shutter); only 0x2b left → doors **raw=10**; then walk-UP |
| 0x2b | HP240 invulnerable mover on 0x49/0x5d — ignore for clear/boss fight |
| TF room | **0x3d UP of boss** (not east); HC mid-room first |

## Evidence

- `recordings/l3_recon.json` — door/entry facts, path notes
- `recordings/level3_west_key_isolated.json` — Clean west-key pure trials
- `recordings/level3_north_chain_isolated.json` — Clean north-chain 0x7b→0x5b
- `recordings/l3_past_5b_recon.json` — **LIVE doors from 0x5b** + compass path hops
- `recordings/l3_raft_recon.json` — **assisted Raft pickup** (`ADDR_RAFT=1`) recon
- `recordings/level3_raft_assisted.json` — **2/2 durable runner** from Level3Darknuts
- `recordings/l3_manhandla_recon.json` — early Raft→boss path recon
- `recordings/level3_to_boss_assisted.json` — **2/2 assisted** Raft→Manhandla→TF `0x04`
- `recordings/l3_manhandla_shortcut.json` / `l3_manhandla_map_explore.json` — 0x5c/0x5d/0x4d
- `recordings/l3_westkey_probe_report.json` — door probes from Level3WestKey
- `recordings/l3_5b_spawn.png` / `l3_north_up_x120.png` — room visuals
- Entry: `uv run python nes/zelda_i/scripts/run_l2_to_l3.py --infinite-life --from-state Level2ExitOverworld`
- Isolated L3 segment CLIs pruned. Durable runner:
  `uv run python nes/zelda_i/scripts/run_survival_spine.py --through level3 --no-video --trials 1`
  (prefix hops: `--through l3-entry` / `l3-dest-6b` / `l3-raft`)

## Sources

- [Zelda Dungeon — Level 3: The Manji](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-3-the-manji/)
- Local: `docs/research/DUNGEON_WALKTHROUGHS.md`
