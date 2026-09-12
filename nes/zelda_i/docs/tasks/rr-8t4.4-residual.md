# rr-8t4.4 — natural L6 → bait shop `0x34`

Living Survival residual. Do not STATUS. Do not add Food/bomb/key pokes.

## This sitting (2026-09-12) — bead `rr-npv.8` Clean 0x33 0-hit

Claimed `rr-npv.8`. Tight dump: natural prefix hp `0x22` through
clear43, then 0x33.

Dump before (peel): 2 hits at (80,173) `combat_evade_peel`, drops 0.
Hit 1962: slot 1 at (80,155) cheb 18 **in UP hitbox**, slot 3 at
(88,173) cheb 8 in pad. Sandwich on the key row.

One change: `Room33ScoopController._combat` holds (120,117), A in
place when the sword box is true and cheb >= 16, away inside the pad,
no south chase.

Dump after: `ok=True` 2128f glance `0x33 (103,173)` hp `0x22` keys 1
full, hits 0, drops 0.

Natural-entry: `clear33_key` green 2128f hp 34=`0x22` hits {} live 0/3.
Then `clear23_key` red: death `0x23 (64,157)` m17 hp `0x20`, Goriya
`0x06` N+W, `combat_wait`, end 15463. Entered 0x23 at lo>=2. Do not
retune 1-heart 0x23.

Glance 0x33 leave: play `0x33` `(103,173)` mode 5 TF `0x00` keys 1
health `0x22` lo==hi deaths 0. Next leftover is 0x23. `route_eligible=false`.
Do not STATUS.

## This sitting (2026-09-12) — bead `rr-npv.8` Clean 0x33 peel leftover

Claimed `rr-npv.8` (already in_progress). Clean power-on
`run_level1_complete --natural-entry --trials 1`, no assist.

Repro (baseline, occupancy tree): `failed=clear33_key` room 0x33
`(94,173)` mode 5 TF `0x00` keys 1 health `0x21` deaths 0,
`0x33_needs_heart` heart_wait=180, stage 2443f / end 13091.
Hits: `0x2a_N` combat_backstep d=8, `0x2a_E` combat_engage d=8.
`tuning.evades=0`. Prefix last_health `0x22`.

Root cause: `GenericDungeonRoomController._combat` never called
`evader.decide`. Tracker/DamageLog were report-only. Link walked into
the 16px BODY pad. After live==0 the seed has no `0x60`/`0x22` heart.

Fix that landed: `CombatTuning.evade` (default False). ROOM_33 on.
`_combat` honors `threat.decide` and will not chase inside
`MIN_DODGE_BODY`. Peel does not slash-walk. Global evade (every room)
timed out `exit42`; do not turn it on repo-wide.

Remeasure: still `clear33_key` red, leftover `(98,173)` mode 5 keys 1
health `0x21`, `0x33_needs_heart` 5198f / end 15846, evades=407, live
0/3. Hits: `0x2a_E` then `0x2a_W` during `combat_evade_peel` at d=8/5
(`dodgeable` False). Same leftover class. Kite/always-away starved the
kill and died (5 hits, live 3/3). 1-heart 0x23 stays blocked.

Glance: play `0x33` `(98,173)` mode 5 TF `0x00` keys 1 bombs 0 health
`0x21` lo!=hi deaths 0. `route_eligible=false`. Do not STATUS.

## This sitting (2026-09-10) — bead `rr-ps7.3`

Claimed `rr-ps7.3`. Leftover-relative L2 OW walk on `0x4C` east mouth.

Policy (`overworld/path.py`): UP/DOWN hop with `align_x`, leftover on the
east/west edge: walk toward `align_x` on a walkable y. Occupancy miss
(true no-move, not a 2px slide) → block cell → y-peel; no path → stand.
Never RIGHT at `x≥232` (scrolls to `0x4D`). Inland hops stay
`align_and_push`.

### Power-on `--through level2-entry` (3/3 this sitting)

Stop at first red: `clear23_key` L1 play `0x23` `(144,149)` mode 5 TF `0x00`
keys 0 bombs 0 rupees 10 health `0x22` lo==hi, `occupancy_patrol` 4627
misses / 6000f. Not the `0x4C` hop. `set_state=0`. Tag `l2_entry_rrps73`.

### Isolated ROM (Level1ExitOverworld, `door_path=True`)

- Natural door hops: L2 play `0x7d` `(120,205)` mode 5 TF `0x01` keys 0
  bombs 0 health `0x33` lo==hi, deaths 0, `food_writes=0` / progression
  writes 0, hop_10_3c then `level2_path_stop`.
- East-mouth knock `y=157` at `(240,133)` arrival: first action LEFT, then
  UP peel, enter L2 `0x7d` `(120,205)` same glance. ~700f after knock.

Unit: leftover `(240,157)` hop UP `align_x=112` first action LEFT, never
RIGHT, never `unstick_wait`. On-column still pushes UP.

Clean campaign: `docs/tasks/rr-npv-clean-parallel.md`. Do not STATUS.

## Shop hop (wired, untested live)

Dedicated `--through level7-bait-shop`: Recorder warp join peels north at
`0x54` → `0x44` → shop `0x34`. No `ADDR_FOOD` write on that hop.

Hypothesis (untested):

```text
0x22 ↓0x32 →0x33 ↑0x23 →0x24
0x24 blow DOWN → 0x45
0x45 ↓0x55 ↓0x65 ←0x64 ↑0x54 ↑0x44 ↑0x34
```

Gaps (OVERWORLD_DOORS): `0x54→0x44` x≈116, `0x44→0x34` x≈132.

## Dead

- `0x22/0x32/0x33/0x23/0x24/0x25` south to row-4
- `0x33` RIGHT at y=141 into `0x34`
- New Food/bomb/key writes to skip `0x4C`
- Poke-pin `Level6ExitOverworld` as leave proof
- Hold UP at `0x4C` east mouth `x≥232` (trees)
- Occupancy 1px-grade on OW 2px UP slide (oscillated 157↔155)
- Isolated `engage_distance=64` on natural L1 0x23
- 1-heart 0x23 chase/contact/maze patrol (blocked class)
- 0x33 `contact_backstep` 16→24 alone (still entered 0x23 at lo=1)
- 0x33 scoop-if-low without holding key-DONE (leave `0x21`, 0x23 death)
- 0x33 heart-wait on `key_got` while Stalfos live (hit to `0x20`)
- 0x33 180f `_patrol` from `(80,165)` (never sat on the drop)

## This sitting (2026-09-11) — walk key-tile `(96,173)` after live==0

Chase/contact on lo≤1 stays **blocked**. Do not retune 0x23 combat.
Hypothesis: after `live==0`, walk to reward tile `(96,173)` and linger
(key + any 0x60 heart/fairy); `heart_wait` only once on-tile; fail-closed
`0x33_needs_heart` if lo<hi. Units: 56 passed. One ROM `--natural-entry
--trials 1`. `--no-video`. Do not poke.

| try | leftover | live | frames |
|-----|----------|------|--------|
| aisle x<=96 | timeout `0x23` `(108,157)` m5 hp `0x21` | 3 | 6000 |
| 0x33 backstep 24 | death `0x23` `(138,157)` m17 hp `0x20` | 3 | 441 |
| 0x33 scoop-if-low | death `0x23` `(138,157)` m17 hp `0x20` | 3 | 441 |
| hold-DONE on key_got | fail `0x33` `(88,165)` m5 hp `0x20` | 1 | 2295 |
| wait only live==0 | fail `0x33` `(80,165)` m5 hp `0x21` | 0 | 2420 |
| walk key-tile x-first (this) | timeout `0x33` `(88,165)` m5 hp `0x21` | 0 | 6000 |

clear53 last_health **`0x22`**. clear33 last_health **`0x21`**,
`heart_wait=0`, `last_live_enemies=0`, phase FIGHT, notes
`at_entry_door`/`target_room_playable` (never `0x33_needs_heart`).
**0x23 not entered.** Walked RIGHT from `(80,165)` toward `(96,173)`;
stuck at `(88,165)` (key under Link). dx==dy so x-first mashed RIGHT
into the east block; never on-tile so wait never armed. PNG
`recordings/level1_complete_t0_natural.png` (2/3 hearts, standing on
key). JSON end_frame 16648 `failed=clear33_key`. 0x52 diamond **green**.
`route_eligible=false`.

New miss class **1/3** (not blocked): greedy x-first to `(96,173)`
walled at y=165 x=88. Occupancy: miss → block cell → replan; no path →
stand.

Pin glance (this ROM): play `0x33` `(88,165)` mode 5 TF `0x00` keys 1
bombs 0 health `0x21` lo!=hi deaths 0. 1-heart 0x23 chase still
**blocked**.

## Leftover

Clean power-on: L1 `0x33` `(88,165)` mode 5, TF `0x00`, keys 1, bombs 0,
health `0x21`, deaths 0, `last_live_enemies=0`. 0x23 **not entered**.
clear53 health **`0x22`**. clear33 last_health **`0x21`**. Next hop:
y-first DOWN off y=165 to 173 then RIGHT to `(96,173)`, or linger/scoop
at leftover xy (key is here); do not mash RIGHT at y=165; do not resume
lo≤1 chase/contact. `route_eligible=false`. Do not STATUS.
