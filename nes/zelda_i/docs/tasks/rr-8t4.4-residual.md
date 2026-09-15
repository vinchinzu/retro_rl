# rr-8t4.4 — natural L6 → bait shop `0x34`

Living Survival residual. Do not STATUS. Do not add Food/bomb/key pokes.

## This sitting (2026-09-15) — the 20R budget, and rocks are half the resets

`rr-doua.1` still in_progress. Two new scratch tools: `bomb_budget.py` (the
arithmetic) and `probe_contact.py` (a 48-frame ring buffer dumped on every
`$04F0` arm, so a collider is named rather than guessed).

**20R costs 36 unbroken kills** in expectation, 46 with no luck, 128 with no
streak at all on row-0 octoroks. `$0627 == 16` is tested before
`$0050 >= 10`, so a clean streak pays at 10, 26, 36, 46 — the fairy spends
six kills of 5-rupee progress. One pass of the walk plus `0x48` is 32 bodies
worth 12.0R random; it only clears 20R with a 26-kill streak. **`0x4A`'s six
tektites are row 1 (0.891 R/kill) — 5.3R of the corridor's 8.4R** — and the
route note said to skip them.

**Two of the four contacts are `rock_projectile` (slot 11, hp 0).** The old
probe filtered `0 < hp < 200` and could not see them. "The hunt walks onto
the bodies" was half the story.

`_engage` now reacts to `_closest_body` (not the held target), swings with
its direction (`nes_action(face, "A")`) so there is no 17-18px dead band, and
never peels inside `MIN_DODGE_BODY`. Live `contact7` `ok=True` room `0x4a`,
1R, 10/9 kills, best 3, 2 resets — both named defects gone from the contact
list, but the **headline is down** vs `contact1` (3R, best 7) and a 1-vs-1
rupee count on a deterministic spine is not a controlled comparison.

`ScreenHunter.shield` (tracker + `threat.assess` + face the `approach_side`)
is wired and tested but **defaults off**: every live walk with it on ran Link
out of hearts on `0x49` (`contact4`..`contact6`, mode 17, byte-identical
under three gatings). Suite 1518 passed. Do not STATUS. Do not overwrite M5
18909f.

## This sitting (2026-09-15) — assist off + sword-reach hunt

`rr-doua.1` still in_progress (20R bombs). `--through pre-l1` now strips
assist even if the caller passed one (CLI too). Live 1/1 green, tag
`prel1_reach`, `assist=None`, `set_state=0`, end 4926f.

Walk 3978f leftover play **`0x4A` `(0,141)`** mode 5 TF `0x00` keys 0
bombs 0 rupees **3** health **`0x20` 0/3** lo!=hi, `hits_taken=2`. Hunt:
11/9 kills, `streak_best` **7** (was 4), `streak_resets` **2** (was 6),
`hurt_events` 4, `damage_taken` 5 (visible now that the refill is gone),
`rupees_banked` 3. Screens: clear 0x77/0x58, skip peahats on 0x68/0x59,
`hunt_hurt_49`. Forced 5-rupee still did not fire (peak 7 < 10).

Sword-reach stand + A pulse is in `overworld/hunt.py` (`sword_stand`,
peel inside `MIN_DODGE_BODY`, slash in-place, no occupancy walk onto the
sprite). Remaining contact is the two resets and the 0x49 hurt-retire.
Do not STATUS. Do not overwrite M5 18909f.

## This sitting (2026-09-15) — pre-L1 rupee streak is contact, not "no drops"

`rr-doua.1` still in_progress (20R bombs). Hunt walk 1/1 green, 14 kills, 0
rupees, `streak_best` 4, `streak_resets` 6. Cause is measured: hunt
occupancy-walks onto the sprite; `Link_BeHarmed` zeros `$50`/`$627` and
grants `$04F0=24`; a wooden octorok chip is `$0670` `$80`; Survival assist
writes `$0670` back to `$FF` the same frame, so `damage_taken` and
`assist.damage_events` stay 0. Probe `scratch/probe_kill_streak.py`: every
real reset has iframes 24 / knockback 32 / hp `0x22`/`$FF`. Random table on
this corridor is Baxter A (red octorok 31%), not group B. `hurt_events` is
the census. Next: sword-reach hunt so the 10-kill 5-rupee can fire. Do not
STATUS. Do not overwrite M5 18909f.

## This sitting (2026-09-14) — Gathering 4.5.1 (`rr-ps7.4` / `pre_l1`)

Occupied-lane LEFT/RIGHT + 0x48 `align_x=120` closed the named 0x37→0x4A
prefix deaths. Combined Clean from `Level1ExitOverworld`: **ok 4612f**,
leftover play **`0x4A` `(155,141)`** hp **`0x31` 1/4**, `hits_taken=0`.
DOWN occupied-lane reverted (left the 0x38/0x48 columns). Stand cap 8f.
The 1/4 leftover is the 0x4A farm (`farm_below_hearts=3` after a hop-local
drop, tektite wave one-shot, `farm_gave_up` twice), not those hits.
Baseline evade-on was 3/4 in 2907f. Door suffix unchanged; 0x4C body still
on the `align_x=112` column. `path.py` 1004 LOC — do not grow it.

Ladder next open is **`pre_l1`** (The Gathering). 4.5.1 leftover this
sitting: play **`0x68` `(48,198)`** mode 5 sword 1 hp `0x22` 3/3 hits 0
(`bush68_stand`, 17 occupancy misses). Row-6 east from the L1 north column
is dead. Next hop is `0x59` SOUTH → `0x69`. `--through pre-l1` is wired
(boot only, no `clear53`). Do not STATUS.

## This sitting (2026-09-14) — 0x49 occupied-lane (LEFT/RIGHT only)

`OverworldPathController.occupied_lane` default off; on for
`OverworldToLevel2Controller`. Before a LEFT/RIGHT hop travel step, if the
intended y-lane sits in a tracked body's `MIN_DODGE_BODY` pad, replan to a
parallel y. Already in-pad still yields to evade. No path → stand ≤8f then
the hop gets the frame. DOWN occupied-lane left the 0x38/0x48 columns
(added hits) and was reverted.

Live prefix (this policy, **without** the 0x48 align_x=120 change): 0x49
hit gone (`hop5_lane` 71f). 0x48 leever still at f=997. Prefix 5955f hp
`0x32` 2/4 on 0x4A (farmed). Door suffix then `east_death` on 0x59 — 2/4 vs
baseline 3/4, do not treat as a new 0x4C census. Units
`tests/test_ow_occupied_lane.py`. `path.py` 1004 LOC. Do not STATUS.

## This sitting (2026-09-14) — 0x48 DOWN hop strafes onto x=120

Named next action from the occupancy_stall leftover. `align_and_push` now
keeps `align_x` on a DOWN hop at y≥205 (UP at y≤80). Production
`LEVEL2_PATH_HOPS` / `LEVEL2_DOOR_HOPS` 0x48→0x58 is `align_x=120` (was
112, which is the rock 8px west of the gap). Units in
`tests/test_ow_align.py`. Live prefix `Level1ExitOverworld` → 0x4A:
**2072f, 1 hit** (0x49 f=1758); 0x48 hit gone. Baseline evade-on was 2907f
2 hits. Occupied-lane (LEFT/RIGHT only) then removed the 0x49 hit on a
separate trial; DOWN occupied-lane left the 0x38/0x48 columns and was
reverted.

Combined (align_x=120 + occupied_lane LEFT/RIGHT, 2026-09-14): prefix
**ok=True 4612f** leftover play `0x4A` `(155,141)` mode 5 hp `0x31` **1/4**
TF `0x01`, `hits_taken=0` at stop, 14 evades. Notes include
`farm_start_4a_2` / `farm_gave_up_1` twice — default `farm_below_hearts=3`
kicked in on 0x4A after a hop-local drop, and the tektite wave is still
one-shot. Named 0x48/0x49 prefix deaths did not fire. Door suffix not
re-censused. Do not STATUS.

## This sitting (2026-09-14) — 0x4C census

Census only. One Clean trial, no pokes, no `--infinite-life`, no path retune.
`route_eligible=false`. Do not STATUS.

```
PYTHONPATH=nes:. QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/probe_level2_suffix.py \
  --from-state Level1ExitOverworld --farm-hearts 0
```

**ok=False** `fail=door_death`. Prefix 2907f (0x37→0x4A). Door controller 1674f.
Notes `['farm_h3','rejoin_59','east_5a_h2','clear_2_to_2','door_death']`.
Inland leftover, not the Survival east-mouth timeout.

Last playable: play **0x4C `(121,133)`** mode 5 facing W TF `0x01` keys 0
bombs 0 rupees 6 health **`0x30`** lo!=hi (0/4) `$0670=122`. Death: **0x4C
`(120,133)`** mode 17 hp `0x30` 0/4. Arrived 0x4C already 0/4 (lost 1 on
0x59→0x5A, 1 on 0x5D→0x4D). The body spent the partial.

Hop **5** → 0x3C UP `align_x=112`, no y_band, `stuck=0`. Reason tail 30×
`hop5_ax` walking x 160→121 at y=133. Occupancy walker does not arm inland.
Killer: slot 1 octorok_fast `0x08` at `(112,124)` vx=0 vy=+0.8 cheb 9 **in
pad**, TTC 0, `dodgeable=False`, body. Four more octoroks live, no rocks.

**`body_undodgeable`**. Same 0x49 shape: `vx=0` so stand TTC is "safe" until
Link is in the pad, then evade yields and `hop5_ax` keeps LEFT. Link walks
into it. Next: occupied-lane on hop 5 so it does not LEFT onto a body on
the door column. Do not retry an in-pad peel or an idle lane wait.

Pre-L1 6 HC (`docs/PRE_L1.md`) would have had budget left after the two
suffix hits. That is a hearts answer, not a skip of the occupied-lane.

## This sitting (2026-09-12) — `rr-ps7.3` opt-in evade on the L2 prefix

Named action from the section below ("wire an opt-in threat step into
`OverworldPathController`") is **landed and measured**. Not a STATUS claim.
No pokes, no assist. Compose from `Level1ExitOverworld`.

`evade: bool = False` on `OverworldPathController` (the `CombatTuning.evade`
shape), `True` only on `OverworldToLevel2Controller`. It runs in `step`
*before* `_do_hop`, after the transition / mode / farm gates; the tracker is
observed every frame; `bounds` and `blocked_dirs` are the four `EDGE_*`
scroll lines (a wrong step out here changes screen, it does not bump a
wall); no occupancy grid reaches `_can_move`.

| `0x37`→`0x4A` prefix | frames | hits | leftover |
|---|---|---|---|
| evade off (control, byte-identical to before) | 5010 | 2 (`0x48` 997, `0x49` 1855) | `0x4A` (155,141) hp `0x32` **2/4** |
| evade on (3/3 identical) | **2907** | 2 (`0x48` 997, `0x49` 1813) | `0x4A` (130,101) hp `0x33` **3/4** |

56 parries, 20 evades, 68 stuck-peels. The heart is banked because the
prefix now reaches `0x4A` with the tektite wave still alive
(`farm_start_4a_2` → `farm_ok_3`); the door suffix then **skips** the farm
(`start_filled=3`, `farm_skipped`).

Composed `probe_level2_suffix --farm-hearts 0` (2/2): was `farm_short` on
`0x4A`; now clears the farm gate, rejoins `0x49`→`0x59`→`0x5A`, clears the
corridor, runs the `0x5C` maze and dies on **`0x4C` (120,133)** mode 17 with
5 live octoroks — `notes=['farm_h3','rejoin_59','east_5a_h2','clear_2_to_2',
'door_death']`. Four screens further than the old stop, but a death, not a
named stop. `route_eligible=false`. Do not STATUS.

### Neither named hit was removed — both are undodgeable by a TTC model

- **`0x48` f=997 leever `0x10`.** It sits at *exactly* `(112,205)`,
  `vx=vy=0`, tracked age 475 frames before it surfaces: stand TTC is 0 and
  `dodgeable` is False for the whole approach. It has no approach velocity —
  it surfaces under Link. The real cause is a **wedge**: `align_and_push`
  drops its `align_x` rule below y=205 (`80 < link_y < 205`), so Link
  hammers DOWN into the wall 8 px west of the x=120 gap for **159 frames**,
  and `track_stuck` never arms the unstick ladder because he flips
  112↔113 every frame. The hit is the price of the stall.
- **`0x49` octorok `0x8`.** Already inside the 16 px pad the first frame it
  registers (`|dx|=4`, `|dy|=15` → TTC 0, not dodgeable); the frame before,
  standing is safe past the 32-frame horizon because its velocity points
  away. Link walks into it; it never walks into Link.

### Measured dead ends (reverted)

- **Unconditional in-pad peel** (the dungeon answer): a 0x38 octorok matched
  the 1 px peel for 45 frames and hit anyway — 4 hits, 0/4 hearts, 10271
  frames; with parry added, death on `0x3b` two screens off-route.
- **Lane guard** (idle while the hop's own next step walks into a pad):
  alone it timed out at 25000 frames on `0x48`; in the full variant 763 idle
  frames and more hits. Removed.
- **Tolerant wedge detector** (2 px / 24 frames + 20-frame license) does
  unwedge `(112,205)` — stall gone — but costs a hit lower on the same
  column: 3 hits, 1/4 hearts. The in-pad peel stays gated on the strict
  `stuck` counter.

Next action: the `0x48` hit is a **lane** problem, not an evade problem —
give the `0x48` DOWN hop a way back to the x=120 gap column at y≥205 (the
`80 < link_y < 205` guard in `align_and_push` is why it cannot), or an
occupancy lane. The `0x49` hit needs an occupied-lane test, not TTC. Do not
re-try an in-pad peel or an idle lane wait out here.

## This sitting (2026-09-12) — `rr-ps7.3` the 0x4A farm is not a farm

Compose from `Level1ExitOverworld` (OW `0x37` `(112,125)` hp `0x33`,
3/4). Not a STATUS claim. No pokes, no assist.

**The named next action ("make the 0x4A farm bank a heart from 1
filled") is not reachable. 0x4A is a one-shot screen.** Measured, not
inferred:

| round trip off 0x4A | back on 0x4A |
|---------------------|--------------|
| depth 1 `0x49` → back | `raw=[]`, `world_kill_count` still 4 |
| depth 2 `0x49`→`0x59`→`0x49` → back | `raw=[]`, still 4 |

The neighbours are alive on the same visits (`0x49` one `0x9`, `0x59`
four `0x1a`), so this is 0x4A, not the object read. Once the opening
tektite wave is killed the screen stays dead for the rest of the run.

Two bugs were hiding that. Both fixed:

- **`farm_leave` was one frame.** It pressed the restock direction once
  and then zeroed `empty_frames`, so the `farm_wait` oscillation
  ("RIGHT if x<160 else LEFT") undid the single step and the screen
  never scrolled. At4A `--min-filled 4`: 1348 `farm_wait` / 15
  `farm_leave` / **0 restocks**, peak 3, 96 occupancy misses. The
  screen was never restocked *once* in the life of this controller.
  Now latched (`leaving`): occupancy-walk to the `LEAVE_GOALS` edge cell
  on the y=141 lane, then `farm_leave_push` until the scroll. The
  `_is_entering_screen` guard no longer bounces the leave back inward at
  x<36 — that is exactly where the scroll has to happen. Measured after:
  5 restocks / 2500f.
- **`run_clean_door_from_env` broke the farm loop on `screen != 0x4A`,**
  which would have ended the farm on its first restock scroll anyway.
  Now it keeps the neighbour.

With both fixed the farm still banks nothing, because the screen is
dead — so it now gives up instead of burning the budget:
`farm_screen_dead` after one restock that sees no prey, same soft-ok
policy as the timeout. 2500f timeout → **536f**.

### The real blocker is the prefix heart ledger

`--farm 0` prefix, `Level1ExitOverworld` → `0x4A`, 2054f, deterministic
across 3 runs:

| f | screen | event | hp |
|------|--------|-------|-----|
| 0 | `0x37` | entry | `0x33` 3/4 |
| 997 | `0x48` | **HIT** `(112,205)`, leever `0x10` at `(112,205)` | `0x32` |
| 1855 | `0x49` | **HIT** `(160,138)`, octorok `0x8` at `(160,130)` | `0x32` |
| 2054 | `0x4A` | leave `(0,141)` | `0x32` 2/4 |

Both hits are body contact on the hop lane: the `0x48` DOWN column
`align_x=120` walks onto a surfacing leever, and the `0x4A` RIGHT lane
`align_y=141` walks under an octorok. `overworld/path.py` never calls
`threat.decide` — `walk_or_swing` slashes a hitbox and faces a contact,
but nothing steps off a body. Same shape as the fixed L1 rooms.

So the budget is: 3/4 out of L1, −2 on the prefix, +0 available on
0x4A, and the door suffix wants 3 (and dies at 3 anyway, per the
earlier At4A sitting). Farming cannot close it. The lever is the two
prefix hits.

Composed leg now stops named instead of dying two screens later
(`probe_level2_suffix --from-state Level1ExitOverworld`):
`ok=False notes=['farm_h2', 'farm_short']`, glance play `0x4A`
`(155,141)` mode 5 hp `0x32` 2/4 TF `0x01`, farm 536f
`farm_screen_dead hearts=2/3`. `run_clean_door_from_env` fails closed
on `farm_short`. 1346 tests pass. `docs/STATUS.md` untouched.
`route_eligible=false`. Do not STATUS.

Next action: kill the two prefix hits. Wire an opt-in threat step into
`OverworldPathController` (the `CombatTuning.evade` shape — default
off, on for `LEVEL2_PATH_HOPS` only; global evade timed out `exit42`).
Do **not** spend another sitting tuning the 0x4A farm.

## This sitting (2026-09-12) — bead `rr-ps7.3` Clean L1 leave → 0x4A

Claimed `rr-ps7.3` (already in_progress). Compose from M5 leftover
(mode 18, L1 0x36, TF 0x01, hp 0x31). Do not STATUS.

Settle 707f lands OW 0x37 (112,125) hp 0x33 (TF heal). Survival
`door_path=True` then died `farm_chase` on 0x48 leevers at (186,93)
mode 17, empty hearts. The generic `farm_below_hearts` hook treated
leevers as a farm.

What landed:
- `worth_heart_farm`: octorok / moblin / tektite only. 0x48 leevers
  and 0x58 mixed groups do not divert. Rupee farms may still use
  leevers.
- `LEVEL2_PATH_HOPS` to 0x4A is green 1801f, leftover play 0x4A
  `(0,141)` hp `0x31` TF `0x01`.
- `run_clean_door_from_env` rejoin uses `farm_below_hearts=0` and
  fails closed if not on 0x59. A 1-heart leftover on 0x4A used to
  farm instead of LEFT, then the east loop RIGHT-scrolled into 0x4B.

0x4A farm from that west mouth: timeout 2500f hearts 1/3, peak 1,
occupancy misses 30, waypoint 0, leftover `(72,104)`. Isolated At4A
is `(0,149)` hp `0x32` and farms to 3 in 146f; the suffix still dies
(first 0x5D, then 0x5A `farm_start_5a_2` when the door hook was on).
Door suffix stays `farm_below_hearts=0`. Do not chase leevers. Do not
RIGHT off 0x4A.

Glance: play `0x4A` `(0,141)` mode 5 TF `0x01` keys 0 bombs 0 health
`0x31` lo!=hi deaths 0. Next hop is the 0x4A heart farm from 1 filled,
or arrive with 2 like At4A. `route_eligible=false`. Do not STATUS.

## This sitting (2026-09-12) — Clean M5 L1 Triforce

Not a spine claim. `run_level1_complete --natural-entry --trials 2`:
both `ok=True`, `triforce=0x01`, end 19416, ~27s.

Ledger (same chain): `clear45_key` 1568f 0 hits, then Aquamentus heart
container `2:0+127 -> 3:1+127`, then TF. Upstream spend unchanged:
`clear52` 1 `0x1b_E`, `clear23_key` 1 `0x5c_W`, `clear44` 2
`0x06_N`+`0x5c_E`. 0x33/0x43/0x45 0-hit.

What landed:
- `_scoop_heart` yields on an unreachable drop (standable-cell goal +
  20-frame cached BFS). Death gone; heart on the 0x45 west column still
  unbanked.
- `OccupancyWalker.next_dir` walks toward the box from an OOB start
  (door column x=32 vs xmin=40).
- FIXED_INVENTORY collect rebuilds the leftover grid (combat scars).
- Collect skips a waypoint when manhattan has not dropped in 48 frames.
  The 0x45 stall was (144,141) for 7666f in a 3px y-loop that never
  tripped in-place stuck.

Measured dead end: grid-aware `ReactiveEvader._can_move` (1px dest,
12px lookahead, peel rank, peel filter). Every variant that changed
0x23 evade buttons capped `clear23_key` at 6000f. Reverted.

`engine.py` is 1225 LOC (soft max ~1000). Do not extract a sibling.
L2-L9 occupancy controllers inherit the OOB walk and collect stale-skip;
not swept this sitting.

Glance is not leave proof for M5. Next Clean leftover is
`l1_exit_ow_l2` (`rr-ps7.3` still claimed on the Survival spine).
Do not STATUS.

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

0x23 dump (lo>=2, after 0x33 0-hit): death `(64,157)` m17. Hit 402
`combat_engage` Goriya cheb 6 on the corridor; then `combat_wait` with
Goriya at `(64,149)` cheb 8 (water north). Boomerang `0x5c` present.
Hold-slash at `(120,157)` was 0-hit until it walked the south door
(timeout 6000f live 3/3) or, with UP-from-mouth, died to `0x5c` at
2.4 px/f (`dodgeable` False). Reverted. Do not retry corridor hold-slash
against Goriya boomerangs. 1-heart 0x23 path untouched.

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
