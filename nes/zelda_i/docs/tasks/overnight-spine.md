# Overnight Survival spine — hop queue (manager)

Parent is **observer only**. Do not STATUS. Do not overwrite Clean M5.
`--no-video` on hop trials. Encode **one** watchable MP4 only after L9
credits greens. Manager does **not** implement path controllers or start a
spine trial. Worker owns the current hop.

**Bead:** `rr-tne2` (in_progress until L6 TF `0x20`; then child beads for
L7/L8/L9). Living residual: `docs/tasks/rr-tne2-residual.md`. This file is
the overnight hop queue — do not mint a second residual.

**Lock:** if `overnight-lock.json` has `status=running`, do not start a
second `run_survival_spine`. Spine to L6 is ~220k frames (~60 min).

**Halt:** 3 serial reds on the SAME checkbox → BLOCKED residual, retarget
that hop (see 3-red table). Occupancy halt at first miss. One change per
trial.

---

## File-split GATE (before any new `--through`)

Refuse a new knob on a file **≥800** lines. Split **before** wiring.

| File | LOC now | Gate |
|------|--------:|------|
| `nes/zelda_i/level6_spine.py` | **785** | **Done** (suffix `level6_spine_suffix.py` 652). Do not grow it back. |
| `nes/zelda_i/survival_spine.py` | **714** | **Done** (`level4_spine.py` 534). Do not grow it back. |

New hop policy lives in a **new file**. Do not attach a through by growing
a file back over 800. `level6_stairs3a_ne.py` is 601 (dedicated red).
stairs3a-ne71 is a **new sibling**. Prefix is **clear3a**, not a red leftover.

- [x] Split `level6_spine.py` (<800)
- [x] Split `survival_spine.py` (<800)
- [x] Wire `--through level6-clear3a` (1/1; play `0x3A` `(144,141)`)
- [x] Wire `--through level6-stairs3a` (**BLOCKED** 3 reds; push yes, idle tile 119)
- [x] Wire `--through level6-stairs3a-71` (**BLOCKED** 3 reds; RIGHT on 119 at x=184)
- [x] Wire `--through level6-stairs3a-ne` (**BLOCKED** 3 reds; UP on 119 at x=160)
- [ ] Wire `--through level6-stairs3a-ne71` from **clear3a** leftover (dest RAM)

---

## Tip (live)

**Leave predecessor (real):** `--through level6-clear3a` 1/1
`l6_clear3a_continuous_v1`: play `0x3A` `(144,141)` rod=1 keys=4 bombs=8
TF=`0x1F` map=`0x0A` bow=0 arrows=0. Center 0x68 unpushed. Tape 219,649f.
PNG: `recordings/l6_clear3a_continuous_v1_final.png`.

**BLOCKED (this halt):** `--through level6-stairs3a-ne` 3 serial reds.
**No v4.** Push live (`112,144→136`). RIGHT+DOWN around y=149 **live**
to AROUND_X. v3 leftover `0x3A` **`(160,147)`** tile **119** last_dir=UP.
Tile 119 is at **x=160**, not only x=184. NE 0x68 live `(208,96)`. East
door **open** — do not walk. Dest stayed `0x3A`. Did not reach `0x71`.

stairs09 analog: south-face NE 0x68, UP onto **`0x71`** at `(208,93)`.
Do **not** UP on 119. **LEFT around** tile 119 at ~x=160, then continue
to `(208,96)`. Start leftover is still **clear3a**. 0x28/0x38/0x39-west
leave stays abandoned.

---

## Bow / Gohma decision (do not grant items)

Gohma needs **one wooden arrow to the open eye**. Assist contract forbids
granting bow/arrows. L1 bow was **skipped**. Leftover ~39R cannot buy 80R
arrows. `$0656` B-item: **2=arrows**.

### Walkthrough dest of 0x3A stairs (source, not live)

Zelda Dungeon L6 + IGN Dungeon Six + StrategyWiki (first quest):

1. Clear 0x3A (Like-Like + Wizzrobes) — **live**.
2. Push the **center/left block** → staircase.
3. **Underground passage** (4 Keese) — **not Gohma**. Mode-9 analog to Rod
   cellar `0x75`.
4. Emerge **NE of the dungeon map** (Zol + Like-Like + Bubble) — avoid,
   go **south**.
5. South: Vires + key. West: Wizzrobes + blade traps. **KEY-UP → Gohma**.
6. Heart → north TF shard 6 (`0x20`).

So `level6-stairs3a` dest is a **passage**, not the boss. Do not invent
the Gohma room id. Live dest is RAM only.

### Warn — one-way into Gohma unarmed

- **0x3A stairs themselves** are source-a **two-mouth cellar** (Rod-class).
  Taking them to *see dest* is OK **if dest is mode 9** and a return exists
  (pattern `level6-exit75`).
- **If dest is a play room with no return stairs**, that is the one-way
  trap into the Gohma *wing* unarmed. Do **not** take that warp. Leave L6
  from `0x3A` for the bow detour.
- **The hard trap is KEY-UP into Gohma without bow+arrows.** Gohma cannot
  die to sword/rod. Boss room typically seals until the kill. Survival
  refill does not open the door. Death is not a route (mode 17 suspends
  assist). Do **not** fight Gohma. Do **not** KEY-UP unarmed.

### WHEN to leave L6 (before vs after dest)

**Fired then reversed:** dest unknown after 3 reds on `level6-stairs3a`
→ leave toward `0x79` **abandoned**. `#1b` RIGHT on 119 at x=184
**BLOCKED**. `#1c` UP on 119 at **x=160** **BLOCKED**. Retry: LEFT around
tile 119, then south-face NE 0x68 `(208,96)` UP onto `0x71`. Do not UP
on 119. Do not walk east door. Do not fight Gohma.

| Dest (RAM) | Leave now? | Action |
|------------|------------|--------|
| mode 9 cellar | **no** (not yet) | Look; **return to 0x3A** (exit analog). Then bow detour from 0x3A. |
| play room + return stairs | after return | Return to 0x3A, then leave. |
| play room, no return | **yes, do not warp** | Abort stairs; leave from 0x3A. |
| dest unknown after 3 reds | leave **abandoned** | Retry `#1d` LEFT around 119 at x=160, then NE 0x68 UP onto `0x71`. |

### Natural bow + arrows (no poke)

1. **Exit L6** from **clear3a** `0x3A` `(144,141)` unpushed (rooms
   already cleared). See leave-path table under the current hop. Keys=4.
   Occupancy halt at first miss. Do not poke doors.
2. OW `0x22` → L1 door **`0x37`** (verified). Enter **`0x73`**.
3. Verified L1 speed route to **`0x23`**. Then **west branch**
   (`LEVEL1_ROUTE.md`; room id live-TBD — do not invent). Stairs →
   `ADDR_BOW=1`.
4. Farm rupees to **≥80** (have ~39). OW Octoroks. Do not poke rupees.
5. Buy arrows **80R** at a merchant that stocks them. Gathering: start
   right×3, up, right (hyp **`0x6B`**, **not live**). Live candle shop
   **`0x5E`** is Shield 160 / Key 100 / Candle 60 — **no arrows**. Probe
   the shop; do not invent a live screen.
6. Return OW → L6 `0x22` UP `0x79`, re-walk the cleared path to `0x3A`,
   take stairs, KEY-UP Gohma **with B=arrows**.

Forbidden pokes on this whole detour: `ADDR_BOW`, `ADDR_ARROWS`, doors,
keys, Map, Whistle, TF bits, rupees, undiscovered items.

---

## Current worker hop (one checkbox)

**Abandon 0x28/0x38 leave.** **Abandon 0x39-west leave.** `#1c`
stairs3a-ne **BLOCKED** 3 reds. Push live; RIGHT+DOWN to ~x=160 **live**;
UP halt on tile 119 at `(160,147)`. NE 0x68 live `(208,96)`. **No
stairs3a-ne v4.** **No stairs3a-71 v4.** **No stairs3a v4.**

**Halted 2026-08-25.** Scheduler deleted. Spine trial killed. Last
committed hop: `#1d` `level6-stairs3a-ne71` **v1 red 1/3**. Do not claim
v2/v3.

**Resume `#1d` `level6-stairs3a-ne71`.** Real predecessor: **clear3a**
play `0x3A` `(144,141)`. **Reuse live push + RIGHT+DOWN to ~x=160.**
Then **LEFT around tile 119** (not UP), continue to NE 0x68 `(208,96)`,
south-face UP onto `0x71` at `(208,93)`. Dest **RAM** mode 9 **or** play
≠ `0x3A`. Do **not** UP on 119. Do **not** walk east door. Do **not**
invent/fight Gohma. Do **not** poke bow/arrows. Occupancy halt at first
miss.

| Field | Value |
|-------|--------|
| `--through` | `level6-stairs3a-ne71` |
| Leftover start | **clear3a** play `0x3A` `(144,141)` rod=1 keys=4 bombs=8 bow=0 arrows=0 TF=`0x1F` map=`0x0A` |
| Stop | dest **from RAM**: mode 9 **or** play ≠ `0x3A`. Do not invent room id / Gohma. |
| Forbidden | `ADDR_BOW` / `ADDR_ARROWS` / doors / keys / Map / Whistle; stairs3a-ne v4; stairs3a-71 v4; stairs3a v4; UP on 119 at x=160; walk east door; fight Gohma; invent Gohma |
| One change | LEFT around tile 119 at ~x=160, then NE 0x68 UP onto `0x71`. New file. Do **not** compose through stairs3a-ne. Do not grow spine files ≥800. |

```bash
QT_QPA_PLATFORM=offscreen UV_CACHE_DIR=/tmp/retro_rl_uv_cache \
  uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-stairs3a-ne71 --no-video --trials 1 --tag l6_stairs3a_ne71_continuous_v1
```

- [x] `level6-clear3a` greens play `0x3A` `(144,141)` — **prefix for this hop**
- [x] `level6-stairs3a` **BLOCKED** 3 reds. Push live. Idle tile 119. No v4.
- [x] `level6-stairs3a-71` **BLOCKED** 3 reds. RIGHT on 119 at x=184. No v4.
- [x] `level6-stairs3a-ne` **BLOCKED** 3 reds. UP on 119 at x=160. No v4.
- [ ] `level6-stairs3a-ne71` dest RAM mode 9 or play ≠ `0x3A`

### Leave path `0x3A` → `0x79` DOWN (split into room hops)

**Leave toward `0x79` abandoned** (0x28/0x38/0x39-west RAM-sealed).
Do not push 0x68 on a leave hop. Occupancy halt at first miss. Current
hop is stairs, not this table.

| Step | Edge | Type | Return / hop | Flag |
|------|------|------|--------------|------|
| 1 | `0x3A` LEFT → `0x39` | kill-door after Vire; 0 keys | **live** (north39 west_align; **west39 v3 enter**) | tile 118 east spawn; leftover LEFT 0px tile 119 replans |
| 1b | `0x39` reclear + WEST | PNG-open, RAM sealed; `doors=9` N+E; west bit never set | dest stayed `0x39` `(124,133)` | **abandoned** (`level6-west39-reband` 3/3); y=133 x≈124–128 y-dead; skip reband/upclip/west39 v4 |
| 2 | `0x39` UP → `0x29` | kill-door (Vires respawn) | **spine 1/1** `level6-north39` leftover `(120,205)` | skip this leave — 0x19/0x18/0x28 chain abandoned |
| 3 | inland `0x29` | tile 244 @ y=157 | **clipped** LEFT+UP / west-wall UP / RIGHT+UP | inland29 v1 RIGHT box; v2 greens |
| 4 | `0x29` UP → `0x19` | forward south was **KEY** (4→3) | **spine 1/1** keys **4→4** already-open | do not poke; Map sprite optional |
| 5 | `0x19` LEFT → `0x18` | east shutter walkable y=141, no key | **spine 1/1** `level6-west19` leftover `(208,141)` keys 4→4 | do **not** KEY-UP `0x09` |
| 6 | `0x18` DOWN → `0x28` | occupancy LEFT y=141 then DOWN | **spine 1/1** `level6-south18` leftover `(120,77)` | 35 diamond misses replanned; north hole skip |
| 7 | `0x28` → `0x38` | kill-door south; reclear opens center | **dead-end** (0x38 three-way sealed) | skip; do not compose leave through `clear28-south` |
| 8 | `0x38` S/E/W | south 170, east 223, west 222 | PNG-open, RAM sealed, doors=0 | **abandoned** |
| 7b | `0x28` east | PNG-open, RAM sealed tile **223** | reclear opened **south** not east | skip; do not re-push |
| 7c | `0x28` west | mouth **undated** (never reached) | LEFT y=93 / LEFT+DOWN boxed x=96 | skip west28 v4 |
| 7d | `0x28` west aisle x=64 | occupancy DOWN x=120 stuck `(120,93)` tile 178 | aisle never reached | skip aisle-west28 v4; **0x28 leave dead** |
| 9 | `0x48` DOWN → `0x58` | forward UP **free** | expected **free** DOWN | next room hop |
| 10 | `0x58` DOWN → `0x68` | forward UP **free** | expected **free** DOWN | next room hop |
| 11 | `0x68` DOWN → `0x78` | kill-door after 0x78 clear | **reclear TBD** | occupancy halt; do not poke |
| 12 | `0x78` RIGHT → `0x79` | **free** | **free** | do **not** LEFT from `0x79` (KEY) |
| 13 | `0x79` DOWN → OW `0x22` | south mouth **residual** | later hop | first spine OW exit |

Do not invent Gohma. East of `0x29` is **sealed**. East of `0x3A` after a
stairs push is **forbidden**. Coordinate clip only after a live miss.

---

## Hop queue (planned `--through` names — not wired, not green)

Names below are **queue labels**. None of L7/L8/L9 OW modules are
spine-green. Do not claim `level7_overworld.py` / `level8_overworld.py` /
`level9_*` as continuous Survival.

Forbidden on every hop: doors, keys, undiscovered items, Map, Whistle,
bow/arrows grants, TF pokes.

### A. Finish L6 TF `0x20` (`rr-tne2`)

| # | `--through` | Leftover start | Stop predicate | One-change checkbox |
|---|-------------|----------------|----------------|---------------------|
| 0 | *(split)* | n/a | both spine files <800 | [x] `level6_spine.py` 785; [x] `survival_spine.py` 714 |
| 1 | `level6-stairs3a` | `0x3A` `(144,141)` unarmed | mode 9 **or** new play room RAM | [x] **BLOCKED** 3 reds (push yes, idle tile 119). No v4. |
| 1b | `level6-stairs3a-71` | **clear3a** `0x3A` `(144,141)` | dest **RAM** mode 9 **or** play ≠ `0x3A` | [x] **BLOCKED** 3 reds. RIGHT on 119 at `(184,147)`. No v4. |
| 1c | `level6-stairs3a-ne` | **clear3a** `0x3A` `(144,141)` | dest **RAM** mode 9 **or** play ≠ `0x3A` | [x] **BLOCKED** 3 reds. UP on 119 at `(160,147)`. No v4. |
| 1d | `level6-stairs3a-ne71` | **clear3a** `0x3A` `(144,141)` | dest **RAM** mode 9 **or** play ≠ `0x3A` | [ ] **current** — LEFT around 119 at x=160, then NE 0x68 UP onto `0x71` |
| 2a | `level6-exit3a` | dest of #1 if mode 9 | play-ready `0x3A` again (Rod-class return) | [x] skipped (dest unknown) |
| 2b | *(abort warp)* | `0x3A` if dest is one-way play | still `0x3A`; no CheckWarp | [x] dest unknown after 3 reds |
| 3 | `level6-exit-ow` | **clear3a** `0x3A` `(144,141)` unpushed | OW `0x22` mode 5 from `0x79` DOWN | [x] **BLOCKED** 3 reds (full leave too big). No v4. Split below. |
| 3a | `level6-north39` | clear3a `0x3A` | play `0x29` enter-stop `~(120,205)` keys=4 | [x] **1/1** `l6_north39_continuous_v1` tape 238,608f hop 18,959f |
| 3b | `level6-inland29` | **north39** `0x29` `(120,205)` | play `0x19` enter-stop; clip not occupancy UP @ x=120 | [x] **1/1** `l6_inland29_continuous_v2` hop 412f tape 239,020f keys 4→4 |
| 3c | `level6-north29` | inland `0x29` | play `0x19` (KEY **LIVE-TBD**) | [x] **skipped** — 3b already landed `0x19` already-open |
| 3d | `level6-west19` | **inland29** `0x19` `(120,205)` | play `0x18` enter-stop | [x] **1/1** `l6_west19_continuous_v1` hop 1,559f tape 240,579f shutter free |
| 3e | `level6-south18` | **west19** `0x18` `(208,141)` | play `0x28` enter-stop | [x] **1/1** `l6_south18_continuous_v1` hop 264f tape 240,843f |
| 3f | `level6-south28` | **south18** `0x28` `(120,77)` | play `0x38` enter-stop | [x] **BLOCKED** 3 reds. Never recleared. No v4. |
| 3f2 | `level6-aisle28` | **south18** `0x28` `(120,77)` | play `0x38` enter-stop | [x] **BLOCKED** 3 reds. West aisle not the mouth. No v4. |
| 3f3 | `level6-clear28-south` | **south18** `0x28` `(120,77)` | play `0x38` enter-stop | [x] **1/1 dead-end** — skip as leave prefix |
| 3g | `level6-south38` | **clear28-south** `0x38` `(120,93)` | play `0x48` enter-stop | [x] **BLOCKED** 3 reds. PNG-open + mask=0 + DOWN no-op. No v4. |
| 3g2 | `level6-clear38-south` | **clear28-south** `0x38` `(120,93)` | play `0x48` enter-stop | [x] **BLOCKED** 3 reds. Reclear analog **false**. No v4. |
| 3g3 | `level6-bomb38-south` | **clear28-south** `0x38` `(120,93)` | play `0x48` enter-stop | [x] **BLOCKED** 3 reds. Bomb-south **false**. No v4. |
| 3g4 | `level6-east38` | **clear28-south** `0x38` `(120,93)` | dest **RAM** enter-stop (do not invent `0x48`) | [x] **BLOCKED** 3 reds. Boxed south of y=141. No v4. |
| 3g5 | `level6-east38-lane` | **clear28-south** `0x38` `(120,93)` | dest **RAM** enter-stop (do not invent `0x48`) | [x] **BLOCKED** 3 reds. East RAM sealed tile 223. No v4. |
| 3g6 | `level6-west38` | **clear28-south** `0x38` `(120,93)` | dest **RAM** enter-stop (do not invent room id) | [x] **BLOCKED** 3 reds. West tile 222 sealed. No v4. |
| 3e2 | `level6-east28` | **south18** `0x28` `(120,77)` | dest **RAM** ≠ `0x28`/`0x38` | [x] **BLOCKED** 3 reds. East tile 223 sealed. Reclear ≠ east. No v4. |
| 3e3 | `level6-west28` | **south18** `0x28` `(120,77)` | dest **RAM** ≠ `0x28`/`0x38` | [x] **BLOCKED** 3 reds. West mouth never reached (x=96 box). No v4. |
| 3e4 | `level6-aisle-west28` | **south18** `0x28` `(120,77)` | dest **RAM** ≠ `0x28`/`0x38` | [x] **BLOCKED** 3 reds. Aisle never reached (tile 178). No v4. |
| 3w | `level6-west39` | **clear3a** `0x3A` `(144,141)` | dest **RAM** ≠ `0x3A`/`0x29` | [x] **BLOCKED** 3 reds. v3 enter 0x39 live. No v4. |
| 3w2 | `level6-clear39-west` | **clear3a** `0x3A` `(144,141)` | dest **RAM** ≠ `0x3A`/`0x29`/`0x39` | [x] **BLOCKED** 3 reds. LEFT+DOWN dated ~3px. No v4. |
| 3w3 | `level6-west39-upclip` | **clear3a** `0x3A` `(144,141)` | dest **RAM** ≠ `0x3A`/`0x29`/`0x39` | [x] **BLOCKED** 3 reds. y=133 LEFT grind dated. No v4. |
| 3w4 | `level6-west39-reband` | **clear3a** `0x3A` `(144,141)` | dest **RAM** ≠ `0x3A`/`0x29`/`0x39` | [x] **BLOCKED** 3 reds. y=133 y-dead. 0x39-west sealed. No v4. |
| 3h | `level6-south48` | `0x48` | play `0x58` enter-stop | [ ] one room |
| 3i | `level6-south58` | `0x58` | play `0x68` enter-stop | [ ] one room |
| 3j | `level6-south68` | `0x68` | play `0x78` enter-stop | [ ] kill-door reclear TBD |
| 3k | `level6-east78` | `0x78` | play `0x79` | [ ] RIGHT free; do not LEFT (KEY) |
| 3l | `level6-exit79` | `0x79` | OW `0x22` mode 5 DOWN | [ ] was the old #3 stop |
| 4 | `l1-reenter` | OW `0x22` | L1 play `0x73` | [ ] door `0x37` |
| 5 | `l1-bow` | L1 `0x73` | `ADDR_BOW=1` via **west of `0x23`** | [ ] do not invent room id |
| 6 | `ow-farm80` | post-bow (OW or L1 exit) | rupees **≥80** natural | [ ] no rupee poke |
| 7 | `ow-buy-arrows` | rupees≥80 | `ADDR_ARROWS≥1`; do **not** use `0x5E` | [ ] live shop residual |
| 8 | `l6-reenter` | arrows owned | L6 play `0x79` | [ ] OW `0x22` UP |
| 9 | `l6-return-3a` | `0x79` cleared path | play `0x3A` | [ ] no combat if still clear |
| 10 | `level6-stairs3a` *(armed)* | `0x3A` bow=1 arrows≥1 | same dest as #1 | [ ] block may already be pushed |
| 11 | `level6-emerge` | dest of #10 | play NE-wing room from RAM | [ ] source: skip combat, go south |
| 12 | `level6-south-key` | emerge leftover | dest south + key if source | [ ] Vire/key source |
| 13 | `level6-west` | after south-key | dest west from RAM | [ ] traps+wizz source |
| 14 | `level6-gohma` | west leftover, **keys≥1, bow+arrows** | Gohma dead from RAM (do not invent room id). B=`2` | [ ] **never unarmed** |
| 15 | `level6` / TF | post-Gohma + heart | `ADDR_TRIFORCE & 0x20`; then fanfare settle | [ ] closes `rr-tne2` **only then** |

3-red retarget: #1 stairs3a **BLOCKED** (push yes, idle 119). Leave
toward `0x79` **abandoned**. #1b stairs3a-71 **BLOCKED** (RIGHT on 119
at x=184). #1c stairs3a-ne **BLOCKED** (UP on 119 at x=160). #1d LEFT
around 119 then NE 0x68 UP onto `0x71` from **clear3a**. Do not
stairs3a-ne v4. Do not invent/fight Gohma. Do not poke. #14 unarmed →
bow detour **after dest**.

### B. L7 TF `0x40` (child bead after `rr-tne2` closes)

Whistle **owned** (L5). Need **Bait 60R** (Hungry Goriya). Pond drain
`0x42` (source). Scaffold: `level7_overworld.py`,
`scripts/probe_level7_entry.py`. Live pond prefix reaches **`0x53`**;
`0x53→0x52→0x42` is **not** green. Bait shop `0x34` **not** live.

Post-L6 leftover is expected **OW `0x22`** after TF fanfare (Zelda 1
warps to the dungeon mouth). Do **not** run the start-`0x77` pond table
as if it were the spine leftover.

| # | `--through` | Leftover start | Stop | One-change |
|---|-------------|----------------|------|------------|
| 16 | **`level7-pond`** (first L7 hop) | post-L6 OW `0x22` (confirm RAM) | pond screen from RAM (source hyp `0x42`; do not invent if dest ≠) | [ ] not start-based hops |
| 17 | `level7-bait` | before Hungry Goriya; rupees≥60 | `ADDR_FOOD≠0` at source shop `0x34` (Armos top-middle) | [ ] 60R farm; shop not live |
| 18 | `level7-drain` | pond + Whistle | stairs / `level==7` entry room RAM | [ ] B=whistle; refuse missing cap |
| 19 | `level7-…` interior | L7 entry | bomb walls, Digdogger (whistle), bait Goriya, Red Candle, Aquamentus | [ ] room ids unknown |
| 20 | `level7` | post-boss | `ADDR_TRIFORCE & 0x40` | [ ] Red Candle expected |

First L7 hop = **`level7-pond`** from post-L6 leftover, not bait (bait
can precede Hungry Goriya; pond is the mouth).

### C. L8 TF `0x80` (child bead)

Need **Candle** to burn bush **`0x6D`**. After L7, **Red Candle** is the
dungeon item (source) — skip `0x5E` 60R buy **if** `ADDR_CANDLE≠0`.
Scaffold: `level8_overworld.py` (`LEVEL8_BUSH_HOPS`,
`CANDLE_SHOP_HOPS`). Bush OW from **start** is assisted-green; **not**
spine-green from post-L7 leftover. Fire→stairs on 0x6D **not** closed.

| # | `--through` | Leftover start | Stop | One-change |
|---|-------------|----------------|------|------------|
| 21 | **`level8-bush`** (first L8 hop) | post-L7 leftover (likely OW `0x42`) | play `0x6D` (verified bush pocket) | [ ] path from leftover, not start |
| 21b | `level8-candle` | only if candle=0 | `ADDR_CANDLE≠0` via `0x5E` 60R **or** L7 red | [ ] skip if L7 dropped red |
| 22 | `level8-burn` | `0x6D` + candle | mode-16 mouth / `level==8` entry room RAM | [ ] burn residual |
| 23 | `level8-…` interior | L8 entry | Book / Magical Key optional; Gleeok 4-head | [ ] rooms unknown |
| 24 | `level8` | post-boss | `ADDR_TRIFORCE & 0x80` | [ ] |

First L8 hop = **`level8-bush`** (reach `0x6D`); burn is the next
checkbox if candle is already owned.

### D. L9 Ganon / credits (child bead)

Gate: **full TF `0xFF`**. Spectacle Rock **`0x05`** (live recon);
entrance room **`0x76`** (live, fixture). Bomb **left** rock. Interior
Old Man blocks without full TF. Silver Arrows required for Ganon.
Scaffold: `level9_overworld.py` (`LEVEL9_ROCK_HOPS`), `level9_path.py`,
`level9_stairs.py`, `level9_patra.py`, `level9_ganon.py`. Backward Patra
`0x52` → Ganon `0x42` → Zelda `0x32` → credits is **fixture recon**,
`route_eligible=false`. Natural interior **unbuilt**.

| # | `--through` | Leftover start | Stop | One-change |
|---|-------------|----------------|------|------------|
| 25 | **`level9-entry`** (first L9 hop) | post-L8 leftover (likely OW `0x6D`) | `level==9` play **`0x76`** from rock `0x05` bomb | [ ] not start-based `LEVEL9_ROCK_HOPS` |
| 26 | `level9-…` interior | `0x76` + TF `0xFF` | Red Ring, Silver Arrows (`ADDR_ARROWS==2`), Magical Key path preferred | [ ] do not poke TF/Silver |
| 27 | `level9-patra` | natural pred of `0x52` | live Patra `0x52` body `0x47` + 8×`0x25` | [ ] 0x03 stairs cellar `0x77` left is live **recon only** |
| 28 | `level9-ganon` | Patra north | `$0672≠0` then Zelda `0x32` | [ ] stun + Silver Arrow B=`2` |
| 29 | `level9-credits` | Zelda room | mode `0x13` updating, submode 3 (credits) or 4 (final page) | [ ] **then** encode the one MP4 |

First L9 hop = **`level9-entry`** (OW to `0x05` bomb → `0x76`).

---

## 3-red retarget (manager)

| Stuck hop | Retarget |
|-----------|----------|
| `level6-stairs3a` | **FIRED.** v1 occupancy box tile 118; v2 hold-UP past hole, east door open; v3 idle tile **119** still mode 5. Push of center 0x68 **solved**. No v4. |
| Dest is play + no return | Abort warp; stay `0x3A`. Leave toward `0x79` **abandoned**. |
| `level6-exit-ow` | **FIRED.** Full leave too big. v1 tile 118 east spawn; v2 kill-door `north_push`; **v3 dated miss for `#3b`:** occupancy UP @ x=120 from `(120,205)` boxes `(120,157)` tile 244. No v4. |
| `level6-north39` | **1/1 green.** Leftover play `0x29` `(120,205)`. Do not re-prove. |
| `level6-inland29` | **1/1 green** v2. Leftover play `0x19` `(120,205)` keys 4→4. v1 RIGHT box `(48,109)` not a 3-red halt. |
| `level6-west19` | **1/1 green.** Leftover play `0x18` `(208,141)` shutter free. Do not re-prove. |
| `level6-south18` | **1/1 green.** Leftover play `0x28` `(120,77)`. Do not re-prove. |
| `level6-south28` | **FIRED.** Never recleared. Center south sealed (mask=0). No v4. |
| `level6-aisle28` | **FIRED.** West aisle not the mouth. No v4. |
| `level6-clear28-south` | **1/1 dead-end** into 0x38. Do not compose leave through it. |
| `level6-south38` | **FIRED.** PNG-open + mask=0 + DOWN no-op. No v4. |
| `level6-clear38-south` | **FIRED.** Reclear analog **false**. No v4. |
| `level6-bomb38-south` | **FIRED.** South bomb **false**. No v4. |
| `level6-east38` | **FIRED.** Boxed south of y=141. No v4. |
| `level6-east38-lane` | **FIRED.** East tile 223 sealed. No v4. |
| `level6-west38` | **FIRED.** West tile 222 sealed. 0x38 three-way sealed. No v4. |
| `level6-east28` | **FIRED.** East tile 223 sealed. Reclear ≠ east. No v4. |
| `level6-west28` | **FIRED.** West mouth never reached (x=96 box). No v4. |
| `level6-aisle-west28` | **FIRED.** v1 occupancy DOWN leftover boxed `(120,79)` tile 118. v2 mouth_step then DOWN halt `(120,82)`. v3 2px overshoot then stuck `(120,93)` tile **178**. West aisle never reached. 0x28 **cannot leave toward 0x79**. No v4. **Abandon 0x28/0x38 leave.** |
| `level6-west39` | **FIRED.** v1 leftover LEFT 0px `0x3A` `(144,141)` tile 119. v2 west mouth `(32,93)` tile 200 occupancy_stand timeout. **v3 enter live:** `0x39` `(208,141)` then reclear started; occupancy WEST miss `DOWN` `(144,109)` tile **118**. Dest stayed `0x39`. West mouth undated. No v4. |
| `level6-clear39-west` | **FIRED.** Enter+reclear+y=141 live. v1 occupancy y=141 LEFT `(142,141)` tile 119. v2 LEFT+DOWN clip live; LEFT 0px `(139,141)` tile 119. v3 LEFT+DOWN clip live; LEFT 0px leftover `(136,141)` tile **117**. ~3px per clip. West mouth never reached. `doors=9` N+E. No v4. Next: `#3w3` LEFT+UP at dated `(136,141)` (east39 reverse), not LEFT+DOWN. |
| `level6-west39-upclip` | **FIRED.** v1 LEFT+UP `(136,141)` live; occupancy LEFT leftover `(133,133)` tile 116. v2 LEFT+DOWN ~3px y-dead; LEFT leftover `(130,133)` tile 116. v3 LEFT+UP ~5px y-dead; occupancy LEFT leftover `(125,133)` tile **118**. West of the y=141 statue. Dest stayed `0x39`. No v4. Next: `#3w4` DOWN onto y=141 at x≈125 then occupancy LEFT. |
| `level6-west39-reband` | **FIRED.** y=133 at x≈124–128 y-dead for DOWN / RIGHT+DOWN / LEFT+DOWN. Dest stayed `0x39` `(124,133)`. `doors=9` N+E; west bit never set. PNG-open RAM-sealed. **Abandon 0x39-west leave.** No v4. Next: `#1b` stairs3a-71. |
| `level6-stairs3a-71` | **FIRED.** v1 clip live; false occupancy_halt leftover `(114,149)` tile 116. v2 push live `112,144→136`; TO_NE y-first UP 0px leftover `(72,165)` tile 116. v3 RIGHT to x=184 live; leftover `(184,147)` tile **119** RIGHT 0px. Stairs revealed. NE 0x68 `(208,96)`. East door open. Dest stayed `0x3A`. No v4. Next: `#1c` south-face NE 0x68 UP onto `0x71`. |
| `level6-stairs3a-ne` | **FIRED.** v1 leftover `(114,149)` tile 116 first RIGHT after push knockback. v2 leftover `(122,149)` tile 118 skip v1 halt; DOWN 0px; halt RIGHT. v3 leftover `(160,147)` tile **119** last_dir=UP. RIGHT+DOWN around y=149 **live** to AROUND_X. Tile 119 at **x=160**, not only x=184. NE 0x68 live `(208,96)`. East door open. Dest stayed `0x3A`. No v4. Next: `#1d` LEFT around 119, then NE 0x68 UP onto `0x71`. |
| `level6-stairs3a-ne71` 3 reds | Occupancy halt at first miss. Reuse live push + RIGHT+DOWN to ~x=160. LEFT around tile 119 (not UP). Continue to NE 0x68 `(208,96)`, south-face UP onto `0x71`. Do not walk east door. Dest RAM mode 9 or play ≠ `0x3A`. |
| `level6-south68` | Kill-door reclear TBD; do not poke. |
| `level6-north29` | **Skipped** — 3b already in `0x19` already-open. |
| `l1-bow` west of 0x23 | Occupancy halt; do not invent cellar id; glance PNG. |
| Arrow shop | Do not poke; do not use `0x5E`; probe Gathering path. |
| `level6-gohma` unarmed | **Illegal.** Jump to bow detour. |
| L7 pond `0x53` | Already live miss: LEFT inland before descending (LEVEL7_ROUTE). |
| L8 burn 0x6D | Candle-owned first; do not poke candle. |
| L9 rock | Confirm leftover screen; do not poke TF. |

---

## Roles

| Role | Owns |
|------|------|
| Observer (parent) | spawn, 90m bump, video at the end |
| Manager | hop queue, bead comments, bow/L7–L9 plan, 3-red retarget. No path controllers. No spine trials. |
| Worker | one hop: policy, tests, `--no-video` trial, residual, `bd export` |

---

## Non-claims

Did not STATUS-promote. Did not overwrite Clean M5. Did not poke
doors/keys/bow/arrows/undiscovered items. Did not grant Map/Whistle.
Did not close `rr-tne2`. Isolated BFS banned. Did not claim L7 pond,
L8 burn, or L9 natural entry as spine-green. Did not start a second
spine trial.
