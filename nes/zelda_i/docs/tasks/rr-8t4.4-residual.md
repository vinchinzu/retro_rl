> Historical lane note. The live sitting is the gathering prefix in [PRE_L1.md](../PRE_L1.md). This file stays because the clean-tip ladder or a route doc still cites it. It is not the current plan.

# rr-8t4.4 — natural L6 → bait shop `0x34`

Living Survival residual. Do not STATUS. Do not add Food/bomb/key pokes.

## This sitting (2026-09-20) — 0x7D carries the 0x7E band; bombs=4

`rr-ttyu.3` stays in progress.  One change: the hop that leaves 0x7D now
carries `SCREEN_7E_EAST_BAND (137,145)`, the same corridor the 0x7E→0x7F
hop already had.  0x7B/0x7C stay `SCREEN_ANY_ROW_BAND`.  `l1` already
said 0x7E want_y=133 stood at 131 and never scrolled; `pre_l1_topup_live`
died there `(40,131)` mode 17 with 19R.  Do not restore ANY_ROW on that
hop.  Do not restore tektite beam-stand.  Rollout still opt-in.

`--through pre-l1 --no-video --trials 1 --rollout` (`pre_l1_7e_band1`):
`ok=True`, `failed_stage=None`, `set_state=0`, assist=null, pokes off.
Walk 8700f → 0x6F with 20R; topup 1f no-op; buy 483f `ADDR_BOMBS>=1`.

Glance (cave leftover, not play): **0x6F `(120,149)`** mode **11** TF
`0x00` keys 0 bombs **4** rupees 20 health **`0x21` 2/3** lo!=hi
deaths 0.  Buy leftover xy `(118,149)` `rupees_at_buy=20` — wallet still
reads 20 on the stop frame; bombs=4 is the stop.

0x7D: 424f, **0 hits**, hearts 1.988→1.988, `band_down` 20 +
`band_down_dodge` 8, spit_duck 117 (was 1236f / 3 hits / 0.484 out).
0x7E: 1384f, 4 octorok_fast, +1R, one `0x55` S, scooped a heart
1.988→2.996.  0x7F two rocks 2.996→1.992.  0x79 5/5, 0x7A 3/4 +5R,
0x7C 6 leevers / 8R.  24 kills, 20R banked, `hop_escape` 593f on 0x78
unchanged.  Occupancy misses 29 on 0x7F (`hop_occ`); still crossed.

Narrow `test_pre_l1` + `test_zd_map` + `test_ow_align`: **34**.  M5 not
re-measured (shop_p7 hop table only).  Do not STATUS.

### Handoff — resume in this order

1. Keep `rr-ttyu.3`.  Flagged arm now greens natural bombs.  C3
   acceptance is still flag-off vs flag-on damage on 0x7B / 0x7C / `0x55`
   plus M5 unregressed — this sitting did not A/B the reactive default.
   The 0x7E band is on the shared hop table, so flag-off gets it too.
2. Do not restore ANY_ROW on the 0x7D→0x7E hop, a fixed y=133 row, or
   tektite beam-stand.
3. Pre-L1 `ADDR_BOMBS>=1` greened on this one flagged tape.  Planner
   owns STATUS.

Full evidence: [`../PRE_L1.md`](../PRE_L1.md), “0x7D carries the 0x7E band”.

## Previous sitting (2026-09-17) — melee the hoppers; 0x6F at 18R; buy still red

`rr-ttyu.3` stays open.  Live headed watch of the flagged arm
(`pre_l1_c3_rollout7d`) died 0x7D / 0R / 6236f.  The UP/DOWN dance, expired
rupees, and skipped tektite/leever kills were the hunt, not the rollout:
beam-standing a tektite's row (229/274 y-reversals on 0x79/0x7A), chase
ignoring floor drops while the wave was up (`hop_scoop` 0), 0x7C 1/6
leevers.  Flagged 0x7D contacts: `007` hunt_lane `(145,85)`, `009` hunt_lane
`(19,101)`, final `(57,109)` mode 17; `threat_duck` still 293f on 0x7D.

Hunt change only (tektites off `BEAM_STAND_KINDS`; no 2 px UP/DOWN align;
uncontested drop scooped mid-wave).  `pre_l1_c3_melee1`: **0x6F 18R**,
`scoop_rupee` 239, 0x7C **6 leevers / 13R**, no 0x79/0x7A beam-stand.
Tektite clears did not improve (2/5, 1/4).  Then 22403f
`shop_p7_hunt_settle` in cave mode 11 — hunt declined, destination not
`done`, timeout 30000f, bombs=0.  Rollout still opt-in.  M5 not re-measured.

Headed `--through pre-l1 --rollout` (`pre_l1_topup_live`) killed 0x79
**5/5** tektites and 0x7A 3/4 + 5R, then **died 0x7E `(40,131)` mode 17
with 19R**, 22 kills, bombs=0.  Topup never ran.  Arrival-short stop was
wired but untested.  Next was the 0x7E crossing (this sitting).

Full evidence: [`../PRE_L1.md`](../PRE_L1.md), “melee the hoppers”.

## Previous sitting (2026-09-17) — C3 rollout lowers damage; `0x55` unchanged

`rr-ttyu.3` is claimed and remains open.  The real power-on trace probe now
saves entry/hit/~250f PNGs and compact Link/body/projectile RAM for 0x7C/0x7D.
`pre_l1_c3_trace1` reproduced the retained control (death 0x7D, 13R, 6996
walk frames) with `assist=null`, pokes off, `set_state=0`, and `ram_writes=0`.
Predecessor: play 0x77 `(64,77)`, wooden sword, health `0x22`/`0xFF` 3/3,
0 bombs/R/keys, TF `0x00`, no other item.

The baseline itself enters 0x7D at y=133.  The rejected fixed-row tape changed
timing/drop state, then spent 385f `scoop_heart`, spawned a second Zora, and
took three body contacts on top of two rocks plus one `0x55`; the row itself
was not the cause.  Control 0x7D damage is three projectiles.

One flagged C3 hypothesis removed the rollout arm's pre-ROM sword yield (417
frames in the first A/B).  STAND is already the control plan: a safe stand
still declines with `no_gain`; a measured walk may now beat a swing during the
zero-velocity muzzle window.  Default reactive policy is unchanged.

`ab_rollout_sword1_measure` (fatal-contact observer fixed) versus the retained
reactive control: total damage 1023→767 units / 8→6 hits; 0x7B 128→0; 0x7C
385→257; walk 6996→6236f.  No new stall.  Ledger: 483 replans, 2396 rollouts,
57,504 frames rolled, 169 claimed frames.  But `0x55` stays **3→3**, all three
on 0x7D, and the tape dies there with 0R before 0x6F.  No route promotion;
natural bombs remain red.  C3 acceptance is not met.

Narrow C2+C3 gate: **229/229**.  Clean M5 confirmation remains frame-perfect:
`ok`, `prefix_ok`, room `0x24`, TF `0x01`, frame **18909**.

### Handoff — resume in this order

1. Keep `rr-ttyu.3` as the only claimed bead.  Instrument the flagged arm's
   three 0x7D `0x55` contact windows with the same PNG/object trace, including
   the fatal contact.  `threat_duck` still owns 517 frames (293 on 0x7D) above
   the rollout; establish whether that precedence hides the moving-shot
   window before changing it.
2. Do not retry either fixed-row policy, change another budget, add random
   jitter/timeout, or add a shop-specific exception.  The rollout arm remains
   opt-in; the default reactive/M5 path stays unchanged.
3. Full pre-L1 acceptance is still natural `ADDR_BOMBS>=1`, no assist/pokes/
   inventory/progression/capacity writes.  0x6F arrival is intermediate only.

Full evidence and artifact names are in [`../PRE_L1.md`](../PRE_L1.md),
“This sitting — C3 sees the bodies; `0x55` is still red.”

## Previous sitting (2026-09-17) — fixed y=133 lane rejected; baseline is 13R

Three unassisted power-on `--through pre-l1` tapes; `assist=None`, pokes off,
`set_state=0`.  The arbiter-wired control `pre_l1_arbiter1` died on 0x7D
with **13R / 21 kills**.  Hop/hunt rung census is live and the former 0x7C
lane/push oscillation is gone.  The bill is now 0x7C 1154f/three hits plus
0x7D 977f/three hits.

`pre_l1_zora_lane1` forced y=129..137 from x>=192 into both Zora screens:
0x7C became 625f/hitless, but 0x7D took six hits and the tape died with 7R.
`pre_l1_7c_entry1` delayed the same row choice to 0x7B x>=232 and did not
force 0x7D; 0x7C instead took five hits and the tape died with 13R.  Both
policies were reverted.  The final C2 arbiter/hunt/path gate passed
**166/166**.  One confirmatory Clean natural-entry M5 trial also remained
frame-perfect: `ok`, `prefix_ok`, room `0x24`, TF `0x01`, frame **18909**.
It confirms rather than replaces the existing 2/2 claim.

Three serial reds means stop this checkbox.  Next action: capture transition
screenshots/RAM for 0x7C→0x7D and form a screen-specific 0x7D crossing; do
not restore a fixed y≈133 rule and do not add a whole-screen `align_y`.

### Handoff — resume in this order

`rr-ttyu.2` is closed.  Its counter criterion was reconciled without changing
the data's meaning: `rung_census` is the arbiter-owned frame-winner count; the
six older fields remain per-screen budgets or branch-entry accounting and are
not presented as winner censuses.

1. Next sitting, claim the newly unblocked `rr-ttyu.3`.  Instrument the real
   power-on walk for 0x7C/0x7D transition PNGs plus compact Link/object RAM.
   Use the rollout/reactive seam for one evidence-based 0x7D hypothesis.
2. Never retry `pre_l1_zora_lane1` or `pre_l1_7c_entry1`, never add a fixed
   coast `align_y`, and never repeat `pre_l1_arbiter1` unchanged.
3. Acceptance is still natural bombs bought (`ADDR_BOMBS>=1`), no assist,
   pokes, state loads, progression writes, or capacity writes.  Arrival with
   20R is an intermediate boundary, not completion.

Commands and required M5/trace fields are in [`../PRE_L1.md`](../PRE_L1.md),
“Next session — instrument C3 before changing policy.”

## This sitting (2026-09-16) — `bomb_topup` ran; still 2R from a 0R arrival

`scratch/probe_topup.py` t1→t3. Assist ON, hunter off on the walk, then
`RupeeTopUpController`. Glance t3: play **0x6F `(0,141)`** mode 5 TF `0x00`
keys 0 bombs 0 rupees **2** health **`0x22` 3/3** lo==hi deaths 0.
`progression_writes=0` / `capacity_writes=0`. Not Clean. Unassisted leftover
is still `pre_l1_topup1` (died 0x7D, 6R) — walk code did not change, so that
tape is still the measurement.

t1 retraced both neighbours on the reverse arrival edge (0x5F 87f peak_live
0; 0x6E 106f peak_live 0). `RupeeTopUpController._extra_hop_action` holds
(`topup_hold`) on a back hop until `hunter.done`. t2 hunted both (15 kills,
2R) but 0x6E spent 214f `occupancy_stand` on the east line (bush maze). t3
treats that stand as no claim so the inward step is the sand corridor: 0x6E
kills 2→4, occupancy_stand 214→0, **17 kills 2R**. Both neighbours still
`hunt_budget_*` retire with bodies left. `streak_best` 5.

Two one-shot six-body waves cannot bank 20R from a 0R arrival without the
10-kill 5-rupee. The top-up is the 19R-plus-one gap, not a farm. Units in
`tests/test_topup.py` (13). Do not STATUS. M5 18909f still stale from the
previous sitting's `path.py` / `tracking.py` (untouched this sitting).

### Handoff — next, in order

1. **Unassisted walk still has to survive to 0x6F.** `pre_l1_topup1` died
   0x7D, 6R. `pre_l1_anyrow1` is the one pass that arrived (14R, 28 kills)
   and died in the destination stand — that stand is now unit-tested to
   finish. The corridor still costs ~3 hearts (Zora spit first). Judge on
   `reason_by_screen`, not on one rupee count.
2. **Top-up 600f budget retires both neighbours.** 0x5F 7 kills / 2R, 0x6E
   4 kills / 0R. Raising `screen_max_frames` on the topup hunter is the
   next topup-only lever; it still will not bank 20R from a 0R arrival.
3. **0x6E is a bush maze.** Do not BFS occupancy there. The hold walks the
   measured sand corridor; a real 0x6E fight needs a measured bush lane,
   same shape as 0x79 beach.
4. **Re-measure M5 Clean** before any STATUS (`path.py` / `tracking.py`
   changed last sitting). `run_level1_complete.py --natural-entry --trials 2`.
5. **0x78 burns a 593-frame `hop_escape`** every unassisted run
   (`stall_escape_78_80_133`). No damage, 10s of wall clock.
6. `overworld/arbiter.py` is still unwired.

## Superseded (2026-09-16) — 0x6F arrived once; top-up wired, not run

Six unassisted `--through pre-l1` runs, tags `pre_l1_scoop1` → `pre_l1_topup1`.
Full detail in [`docs/PRE_L1.md`](../PRE_L1.md).

| tag | leftover | R | kills | what it says |
|-----|----------|---|-------|--------------|
| `pre_l1_beam4` (before) | died 0x7C | 19 | 24 | baseline |
| `pre_l1_anyrow1` | died **0x6F** | 14 | 28 | **all nine hops**; died in the destination stand |
| `pre_l1_bound1` | timeout 0x7C | 11 | 19 | strike wedge, 2 hits all run |
| `pre_l1_wedge1` | timeout 0x7C | 6 | 19 | lane/push oscillation |
| `pre_l1_grind1` / `pre_l1_topup1` | died 0x7D | 6 | 21 | grind cap fires, corridor still costs the hearts |

Landed: `reason_by_screen` census; `SCREEN_ANY_ROW_BAND` on the three
every-row hops; `ScreenHunter._strike_budget`; `_grinding` per-hop-per-screen
cap; `destination_hunted` True in guard; `cleared` split from `done` so a
cleared screen scoops money; `combat.heal_wanted`; a `hunt_heal` rung above
the beam with `HUNT_PICKUP_RADIUS` / `HUNT_HEAL_MAX_FRAMES`;
`ObjectTracker(shot_history=2)` and `path._shot_first` for the Zora muzzle
hold; `overworld/topup.py` + the `bomb_topup` stage with both 0x6F neighbours
measured. 1658 tests green. M5 Clean **not** re-measured.

## Superseded (2026-09-15) — structure pass; leftover still 0x7C

This sitting is a structure pass on the bomb-run, not a new live ROM
leftover. Census and lanes stay in [`docs/PRE_L1.md`](../PRE_L1.md).

Unassisted `--through pre-l1` (tags `pre_l1_beam3` / `pre_l1_beam4`,
reproduced 2/2): died **0x7C `(192,85)`** at 7500f hop 5. **19R** /
24 kills / streak 17 / 2.996 hearts over 6 hits (**4 of 6 from
`0x55`**). 5R of 24R dropped still on the floor. 0x6F not arrived.
Stop is `ADDR_BOMBS >= 1`.

M5 Clean 2/2 18909f unregressed. Tests include `beam`.

Next lever (do not implement this sitting): scoop the 5R
(`scoop_rupees=True`, now above stall-escape); then `0x55` spit
via `threat._FACING_AXIS` 0x03.

Do not STATUS. Do not overwrite M5 18909f. Do not add Food pokes.

## Superseded (2026-09-15) — retest: shop 0x6F never arrived

`--through pre-l1 --no-video --tag pre_l1_retest` 0/1, assist=None,
`set_state=0`. Sword cave green 749f. `bomb_walk` **timeout 30000f**
on hop 3 RIGHT `0x7B`. Glance: play **0x7A `(128,126)`** mode 5 TF
`0x00` keys 0 bombs 0 rupees **2** health **`0x22` 3/3** lo==hi
deaths 0. Hunt 8 kills (4 on 0x78, 4 on 0x7A), streak_best 7, 2R
dropped and banked, 0 left. 0x7A spent 27501f after the 600f hunt
budget; 53 occupancy misses at overlay y≈126. Buy never started.

Same-day earlier hunting trial (`test_pre_l1`) got further: death
**0x7D** 15R 18 kills, still 5R short. Hunt-off geometry probe
(`pre_l1_geo`) cleared 0x7A then **died 0x7B `(65,133)`** mode 17
1R 1/3. Only hop 0x79→0x7A uses beach y=165; 0x7A→0x7B is still
overlay `align_y=131`. 0x6F cave mouth is not live. Do not STATUS.

## Superseded (2026-09-15) — Map-1 south coast is the walk

`shop_p7` lives in `overworld/shop_p7.py`. Dest is **0x6F**. Live hops:
`0x77 → 0x78 → 0x79 → 0x7A → … → 0x7F → 0x6F`. 0x79 east is the beach
(y≈165), not LEFT out the west mouth and not the L8/candle corridor.
`overworld/bomb_shop.py` stays the later 0x4A cave. `gathering.py` is
only the Composer row (sword + walk + buy).

Unassisted leftover: 0x7D, 15R, 18 kills; 5R short of the pack. Do not
re-join via `0x68` / `0x5C` / `0x5E`.

## Superseded (2026-09-15) — 0x7B join torched; dest is 0x5E south

`rr-doua.1` still in_progress. Older join: Map-1 into the bowl, hunt the
four blue tektites, LEFT out the same mouth. Coast work has since
replaced that opening. The other worktrees are different beads (0x4A
bomb buy, raft heart, burn heart, npv.8), not this join.

The 0x6B leftover was the tree wall, not the bowl. Dest (hunt off):
bowl out-and-back green; skirt live through `0x68/0x58/0x59/0x69/0x6A`
RIGHT `0x6B`; `0x6B` DOWN align_x=120 dest-red play **0x6B `(120,189)`**
mode 5 (`DEAD_6B_SOUTH`). Occupancy from the west mouth missed a south
cell east of x=176 (`max_x_blocked=175`). `l8_6b_exits` has UP to 0x5B
and LEFT to 0x6A, no east, no south. 0x7B north is a bomb wall and this
errand has no bombs yet. 0x6C east is a bush pocket. 0x5E east is a tree
wall (t2 leftover `(224,141)`).

The 0x69/0x6A/0x6B/0x7B skirt is torched. One path: bowl, then the live
L8/candle corridor from 0x78 (`LEVEL8_BUSH_HOPS[1:-1]` + 0x5E), then dest
`0x5E` DOWN `0x6E` RIGHT `0x6F`. Hunt/scoop/respawn/laps unchanged. 0x5C
is maze transit. 0x6F cave mouth is not live; buy stage is still the
0x4A factory.

Glance: not re-run live this edit. Next leftover is 0x5E south or 0x6F
arrival. Do not lead with 0x68.

Do not STATUS. Do not overwrite M5 18909f.

Older sittings below that name `0x4A` as the pre-L1 dest are inland prefix,
not this walk.

## Superseded (2026-09-15) — A-edge is first-class; scoop is the 20R gap (old inland prefix)

`rr-doua.1` still in_progress. Composer path is `pre_l1_stages`: sword_cave
→ bomb_walk → bomb_buy. `--through pre-l1` strips assist. `laps=0` on the
Composer row.

ButtonsPressed is an edge. `ScreenHunter._strike` presses A one frame, then
idles; `_approach`'s blocked-align fallback goes through the same edge.
Unit tests pin both producers, plus `hunt_reopen` / `_at_stop` (funded
mid-lap stops; unfunded mid-lap does not). `CombatLedger.report` now has
`rupees_dropped` / `rupees_left` so the scoop gap is a census, not a probe.

Live leftover is still `shop_need_20_have_8`: 17/17 kills, 0 hits, 3/3
hearts, 15R on the floor, 8R banked. A lap is wired and tested and stays
off: same 8R banked. Next Composer change is the scoop, not more screens.

Do not overwrite M5 18909f. Re-measure L1 after this prefix greens.

## Superseded (2026-09-15) — the 20R budget, and rocks are half the resets (old inland prefix)

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

## Superseded (2026-09-15) — assist off + sword-reach hunt (old inland prefix)

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

## Superseded (2026-09-15) — pre-L1 rupee streak is contact, not "no drops" (old inland prefix)

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

## Superseded (2026-09-14) — Gathering 4.5.1 (`rr-ps7.4` / `pre_l1`) (old inland prefix)

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
