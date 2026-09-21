# Pre-L1 loadout

Primary walkthrough: [Zelda Dungeon — The Gathering](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/the-gathering/)
(2015-01-23). Secondary: [IGN Preparation](https://www.ign.com/wikis/the-legend-of-zelda/Preparation)
(2025-07-16). First quest only. Grid: `screen = (row << 4) | col`, start
`0x77` = H8.

M5 Clean is still power-on → L1 Triforce on 3 containers and the wooden sword
(18909f, TF `0x01`). This prefix is the combat-budget answer: gather **before**
the 0x37 mouth, then re-enter L1 with 6 containers and the White Sword. Do
not overwrite the 18909f claim.

No pokes. Do not STATUS from a pin.

## This sitting (2026-09-20) — 0x7D carries the 0x7E band; bombs=4

`rr-ttyu.3` stays in progress.  `pre_l1_topup_live` died 0x7E `(40,131)`
mode 17 with 19R: y=131 is the dead 133 row (`l1` want_y=133 stood at 131
and never scrolled), and 0x7D was `SCREEN_ANY_ROW_BAND` so the walk could
enter 0x7E there.  The 0x7E east hop already used `SCREEN_7E_EAST_BAND
(137,145)`; picking that drift up *after* the scroll is the death.

One hop-table change, hunt/rollout/M5 untouched, rollout still opt-in:

* Hop that leaves 0x7D (target 0x7E) now carries `SCREEN_7E_EAST_BAND`.
* 0x7B / 0x7C stay `SCREEN_ANY_ROW_BAND`.  0x7E→0x7F keeps the same band.
* Do not restore ANY_ROW on that hop.  Never retry a fixed y=133 coast row.

`--through pre-l1 --no-video --trials 1 --rollout` (`pre_l1_7e_band1`),
`set_state=0`, assist=null, pokes off:

| measure | topup_live (red) | 7e_band1 |
|---------|-----------------:|---------:|
| leftover | death 0x7E `(40,131)` m17 19R | **cave 0x6F `(120,149)` m11, bombs=4** |
| bombs / rupees | 0 / 19 | **4 / 20** |
| walk frames | 7742 fail | **8700 ok**, then topup 1f, buy 483f |
| 0x7D | 1236f, 3 hits, 0.484 out | **424f, 0 hits**, 1.988, `band_down` 20 |
| 0x7E | 107f, 1 `0x55` E, death | 1384f, 4 octorok_fast, +1R, heart 1.988→2.996 |
| 0x79 / 0x7A | 5/5 + 3/4 +5R | same 5/5 + 3/4 +5R |
| `0x55` hits | 5 | 3 (two on 0x7C, one S on 0x7E) |
| 0x78 `hop_escape` | 593 | 593 |

Glance: cave **0x6F `(120,149)`** mode **11** TF `0x00` keys 0 bombs **4**
rupees 20 health **`0x21` 2/3** lo!=hi.  Arrival-short stop fired: walk
had 20R so topup was a 1-frame no-op.  Buy leftover `(118,149)` still
reads 20R on the stop frame; `success_addr=$0658` bombs=4.

Narrow `test_pre_l1` / `test_zd_map` / `test_ow_align`: **34**.  M5 not
re-measured (shop_p7 hops only; Clean mouth does not walk this table).

### Next C3 action

Keep `rr-ttyu.3`.  Do not STATUS.  Flagged arm greened natural
`ADDR_BOMBS>=1` on this one tape; C3 still wants flag-off vs flag-on
damage on 0x7B / 0x7C / `0x55` and an M5 confirm if the walker moves.
The 0x7E band is shared, so the reactive default now leaves 0x7D on that
corridor too — do not treat that as a rollout promotion.  Do not restore
ANY_ROW on the 0x7D→0x7E hop.

## Previous sitting (2026-09-17) — melee the hoppers; bank the drop; 0x6F at 18R

`rr-ttyu.3` stays in progress.  The flagged arm was watched live
(`pre_l1_c3_rollout7d`): death on 0x7D, 0R, 6236 walk frames, same ledger as
`ab_rollout_sword1_measure`.  The window showed three hunt failures the
rollout does not own.

1. **UP/DOWN spam.** 0x79/0x7A reversed y 229/274 times.  `BEAM_STAND_KINDS`
   parked Link 56 px off in a blue tektite's row; they hop off that row, so
   the stand goal flipped every hop.  `hunt_79_beam` 150f + `hunt_7a_beam`
   161f.  `_approach` also aligned UP/DOWN on `dy=2` inside sword reach
   (0x79 133↔135).
2. **Rupees expired.** `hop_scoop` was 0.  Chase ignored every floor drop
   while any body was alive; path scoop is 48 px and the beam stand is 56.
3. **Easy kills skipped.** 0x79 3/5 tektites, 0x7A 2/4, both retired at the
   600f hunt budget.  0x7C killed 1/6 leevers.  0x7D died to three `0x55`
   (entry PNG `005`, hits `007`/`009`, final `011` at `(57,109)` mode 17).

One hunt change, default reactive/M5 path untouched, rollout still opt-in:

* Tektites are not a beam-stand kind (leevers still are).
* Align the short axis only when `|cross| > lane_tol` (6).
* Chase banks a drop inside `HUNT_PICKUP_RADIUS` when no live body is in
  that drop's `MIN_DODGE_BODY` pad.

`pre_l1_c3_melee1` (flagged, headless, `set_state=0`, `ram_writes=0`):

| measure | live rollout7d | melee1 |
|---------|---------------:|-------:|
| leftover | death 0x7D 0R | **0x6F 18R**, then 22403f `shop_p7_hunt_settle` timeout |
| bombs | 0 | 0 |
| `hop_scoop` / `scoop_rupee` | 0 / 0 | 272 / 239 |
| 0x7C leevers | 1 kill, 0R | **6 kills, 13R banked** |
| 0x79/0x7A tektites | 3/5 + 2/4 | 2/5 + 1/4 (still retire) |
| `hunt_*_beam` on 0x79/0x7A | 150+161 | **0** |
| 0x7A twerk2 | 274 | 167 |
| 0x79 twerk2 | 229 | 244 (occupancy still chases hops) |
| 0x78 twerk2 | 696 | 696 (`hop_escape` unchanged) |

The 18R is two short of the pack.  The walk then entered cave mode 11, hunt
declined (`mode != PLAY`), and `_after_hops` idled until 30000f because
`destination_hunted` was still false at 2 hearts / 18R.

Follow-up (same sitting): arrival-short now **stops the walk** so
`bomb_topup` can leave 0x6F and fight 0x5F/0x6E.  Tektites keep the chase
until they are gone or the 2400f destination cap — retiring them at 600f
was the skip.  Headed `--through pre-l1 --rollout` (`pre_l1_topup_live`):
0x79 **5/5** tektites in 1024 hunt frames, 0x7A 3/4 and **5R**.  The extra
time on those screens changed 0x7C–0x7E: death on **0x7E** `(40,131)` mode 17
with **19R**, 22 kills, bombs=0, `failed=bomb_walk`.  Topup never ran.
Damage 1023 / 9 hits (five `0x55`).  `set_state=0`.  Bomb buy remains red.
Rollout stays opt-in.

Narrow hunt/beam/arbiter/pre-l1/topup/path/spine: **251**.  M5 was not
re-measured: `ScreenHunter` is the shop_p7/topup walk, not the Clean mouth.

### Next C3 action

Do not restore tektite beam-stand.  0x79 no longer skips the wave; the bill
moved to 0x7E.  Arrival-short stop is untested on this tape.  Next is a
0x7E crossing that keeps the 19R, or a walk that still reaches 0x6F after
the longer tektite chase.  Keep the reactive default and M5 path unchanged.

## Previous sitting (2026-09-17) — C3 sees the bodies; `0x55` is still red

`rr-ttyu.3` is claimed and remains in progress.  Before changing policy,
`scratch/probe_walk_trace.py` was extended to audit the real power-on walk and
save a PNG plus compact Link/object RAM on entry to 0x7C/0x7D, on every
non-fatal hit, and every 250 frames.  A fatal contact is also captured on the
next run; the first trace predated that terminal-hook correction.  Each object
row carries type/state/x/y plus the controller's measured velocity.

`recordings/scratch_walk_trace/pre_l1_c3_trace1.json` reproduced the retained
control exactly: death on 0x7D, 13R, 6996 walk frames.  The audit is
`assist=null`, `allow_pokes=false`, `set_state_count=0`, `ram_writes=0`.  Its
real predecessor is play 0x77 `(64,77)`, wooden sword, 0 bombs/R/keys, TF
`0x00`, health `0x22` + partial `0xFF` (3/3), with no other item.

The capture rejects the old explanation more strongly: the baseline also
enters 0x7D at **y=133**.  The fixed-row tape did not die because y=133 is an
intrinsically bad entrance; shortening 0x7C changed the wave/drop timing.  It
then spent **385 frames `scoop_heart`** on 0x7D, spawned two Zoras rather than
the control's one, and took three body contacts in addition to the same two
rocks and one `0x55`.  The control's three 0x7D hits are all projectiles.

Exactly one C3 hypothesis was tried.  The first rollout A/B had recorded
**417 `rollout_yield_to_sword` frames**: the arm declined before asking the ROM
whenever the hunter could swing.  That is the blind window described by the
bead, because a shot still on the Zora muzzle has zero tracked velocity and
`_shot_first` cannot exempt it.  On the flagged arm only, the rollout now
compares walking against its existing STAND control even during a sword
window.  A safe stand still returns `no_gain` and falls through to the
reactive/hunt rungs; the default reactive arm is unchanged.

The corrected measurement is `scratch/ab_rollout_sword1_measure.json`.  The
first report omitted the fatal hit because the observer discarded mode 17;
the observer now records that contact and retains the full hunter screen
ledger.  The repeat changed observation only and is byte-identical in route
frames/census.

| measure | retained reactive | flagged rollout | delta |
|---------|------------------:|----------------:|------:|
| total damage units / hits | 1023 / 8 | 767 / 6 | -256 / -2 |
| 0x7B damage | 128 / 1 | 0 / 0 | -128 / -1 |
| 0x7C damage | 385 / 3 | 257 / 2 | -128 / -1 |
| `0x55` hits | 3 | 3 | **0** |
| bomb-walk frames | 6996 | 6236 | -760 |

The rollout report is explicit: 483 replans, 2396 rollouts, 57,504 simulated
frames, 169 claimed frames (22 replans + 147 holds), and no new stall.  It
reaches 0x7D with the same 1.492 hearts by spending less life earlier but
collecting no replacement heart, then takes three `0x55` hits and dies with 0R.
It did **not** reach 0x6F or buy bombs, so the arm stays opt-in and nothing is
route-promoted.  C3 acceptance is not met because `0x55` damage is unchanged.

The C2+C3 narrow gate is **229/229**.  Clean natural-entry M5 was re-measured
after the flagged change: `ok`, `prefix_ok`, room `0x24`, TF `0x01`, frame
**18909**.  This confirms rather than replaces the existing 2/2 claim.

### Next C3 action

Do not retry a fixed row or tune another budget from the red.  First explain
the three 0x7D contacts against the active rung: the flagged tape still gives
`threat_duck` 517 frames (293 on 0x7D) above the rollout, and all remaining
0x7D damage is `0x55`.  Capture the flagged arm's three contact windows with
the same PNG/object rows, including the fatal hit, then decide whether the
rollout must measure the moving-shot window as well as the zero-velocity
muzzle window.  Keep the reactive default and M5 path unchanged.

## Previous sitting (2026-09-17) — the arbiter is measurable; y=133 is not a route

Three unassisted power-on `--through pre-l1` tapes, all `set_state=0`, no
assist and no inventory write.  The wired arbiter now reports both hop and
hunt rung censuses; the old 0x7C two-pixel lane/push split is absent.  The
committed-policy control (`pre_l1_arbiter1`) is the current leave: death on
**0x7D**, 13R, 21 kills, eight hits.  It spent 1154f / three hits on 0x7C
and 977f / three hits on 0x7D.

Two narrow entry-lane hypotheses were measured and reverted:

| tag | change | result |
|-----|--------|--------|
| `pre_l1_zora_lane1` | y=129..137 from x>=192 into both 0x7C and 0x7D | 0x7B/0x7C hitless, but 0x7D took six hits; death with 7R / 16 kills |
| `pre_l1_7c_entry1` | same band only for the last 8 px of 0x7B | 0x7C took five hits; death on 0x7D with 13R / 18 kills |

The first tape proves the asymmetry: y≈133 can shorten 0x7C (625f, zero
hits) but is a lethal octorok lane on 0x7D.  The second proves that merely
choosing it later does not stabilize 0x7C against the live leever/Zora wave.
Do not restore either exit-row rule.  After three serial reds this checkbox
is blocked for the sitting.  Next work needs a screen-specific 0x7D crossing
hypothesis from transition screenshots/RAM, not another fixed coast row.

### C2 closeout — the winner census is not the budget ledger

`rr-ttyu.2` closed after the exact narrow gate passed **166/166** and one
confirmatory Clean natural-entry M5 trial returned `ok`, `prefix_ok`, room
`0x24`, TF `0x01`, and frame **18909**.  The trial confirms the existing 2/2
claim; it does not replace it.

The original counter criterion needed one honest correction.  The arbiter's
`rung_census` is the frame-winner accounting and cannot drift from the action
returned.  The six legacy fields are not all censuses: some are per-screen
budgets and others count branch entry even when that branch yields the frame.
They therefore remain beside `rung_census` under their original semantics
instead of being relabelled or mechanically derived from it.

The closeout commands were:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  nes/zelda_i/tests/test_arbiter.py \
  nes/zelda_i/tests/test_hunt.py \
  nes/zelda_i/tests/test_ow_path.py -q
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_level1_complete.py --natural-entry --trials 1
```

## This sitting (2026-09-16) — the weapon was measured; half the wave was asleep

Four measured facts, each one a rung that was acting on a model nobody had
checked against the ROM. The walk still dies on the same half-heart, but
0x7B — the screen that has taken three to five of every tape's hits — is
down to **one hit and a banked heart**.

### A turn and a swing cannot share a frame

`hunt._a_edge` pressed `nes_action(face, "A")`: the face the rung wants and
the A edge, together. `scratch/probe_turn_swing.py` (`turn4`) stands Link on
0x77 and asks for 64 perpendicular turns from a *walking* Link:

| press | turned | did not |
|---|---|---|
| `dir+A` (combined) | 42 | **22** |
| `dir`, then `dir+A` | 50 | 14 |
| `dir` held until the facing agrees | **64** | 0 (1-4 frames, max 4) |

When the turn is refused the blade still goes out — along the **old** facing,
which by construction is an axis the body is not on. That is a guaranteed
miss plus 13 frames of `$00AC != 0` with a body closing. Live, it is not
rare: the gate-off tape (`zoffJ`, byte-identical to last sitting's `zfixE`)
pressed 81 times and **30 of them went out off-face**.

So `_a_edge` spends the frame turning and presses when `$0098` agrees,
capped by `HUNT_TURN_CAP` (a body crossing a diagonal can ask for a new face
every frame — that is a dance, not a turn). `common.swing_or_turn` is the
same rule for the walk's periodic swing.

**Not the shot.** A blade that goes out the wrong way is a miss *and* a pin;
a *beam* that goes out the wrong way is a screen-long projectile down some
other lane, and these screens are full of lanes. Gating it too (`zfixG`)
fired 4 beams where the baseline fired 36, lost full health on 0x78 and cost
half the kills. `_beam_action` passes `turn_first=False`.

### The blade has a near end, and `pad <= MIN_DODGE_BODY` is not a sword rule

`scratch/probe_blade.py` (`blade1`) ledgers every A press of a walk against
the hp drops in the next 16 frames, with each body's offset written in Link's
own frame (`fwd` along the facing, `lat` across):

| nearest body at the press | presses | landed |
|---|---|---|
| `fwd` >= 10 | 41 | 11 |
| `fwd` <= 9 | **13** | **1** |
| `abs(lat)` >= 16 | 4 | 0 |

The sword is an *object the ROM places in front of Link*, so a body
overlapping him is not in front of anything. Every landed press was
`8 <= fwd <= 20`, `abs(lat) <= 12`. `in_sword_hitbox` has no minimum and
`_at_contact` accepted `pad <= MIN_DODGE_BODY` on its own — which is the
softlock rule, not a sword rule — so the contact rung spent 13 pinned frames
at a body that was already touching Link. Four of the eight hits in `zhit6`
have exactly that shape (f=4981: two swings DOWN at a leever 4 px below, the
leever walks in; f=6633: a press at one **1 px** away).

`blade_lands` is `in_sword_hitbox` plus `HUNT_BLADE_MIN_FWD` (10), and inside
it the answer is the peel, not the press — `hunt_*_close_peel`. The hit
census moved accordingly: `slash_recover` owned 5 of 7 contacts in `zhit6`
and **1 of 6** in `zhit7`.

### Standing on a drop is not instant

Live 0x7B (`zhit6` f=4837-4856): Link idles 3 px from a 1-rupee for **twenty
frames** before the ROM hands it over, and a leever closes 10 → 8 and takes
the heart. `common.scoop_toward_drop` idled at `dist <= 4` with no idea what
else was on the screen. The drop keeps for hundreds of frames and the wave
does not, so `_stand_on_drop` hands the frame back whenever a threat is
inside `MIN_DODGE_BODY`.

### Half of 0x7B was asleep

`ObjState` 0 on a leever is the whole dormant phase, and **every layer that
reads "body" was reading it as one**. Measured across five contact tapes:

| leever state | frames still / moved | hp drops | frames touching Link (<=8 px) | of those, armed `$04F0` |
|---|---|---|---|---|
| 0 (under the sand) | 3844 / 45 | **0** | 49 | 4 (iframe carry-over) |
| 1-2 (the rise) | 526 / 0 | 3 | 0 | - |
| 3 (up) | 1941 / 935 | 15 | 111 | 33 |

A state-0 leever cannot move, cannot be cut, and has never hurt Link. Six of
them sit on 0x7B and seven on 0x7C. The blade swung at them (13 pinned frames
each), the evader stepped away from them into the ones that were up, and
`closest_body` handed the whole contact ladder a sand mound.
`combat.dormant_body` now filters them out of `overworld_threat_objects`, out
of `attackable`, and out of `hunt.closest_live_body`; the rise (states 1-2)
is still a threat, so the warning is unchanged.

### The tapes

All unassisted `--through pre-l1`, `scratch/probe_screen_tables.py`.

| tag | leftover | R | kills | hits / hearts | what changed |
|-----|----------|---|-------|---------------|--------------|
| `zoffJ` | died 0x7D | 19 | 23 | 8 / 4.00 | last sitting's `zfixE`, re-measured through the new ablation knob |
| `zfixH` | died 0x7D | 6 | 20 | 8 / 3.99 | turn gate, blade only |
| `zfixK` | died 0x7C | 15 | 21 | 8 / 4.00 | + the blade's near end |
| `zfixL` | died 0x7C | 13 | 21 | 8 / 4.00 | + the scoop yields in the pad |
| `zfixM` | died 0x7D | 13 | 22 | 8 / 4.00 | + dormant leevers are not bodies |
| `zfixO` | **died 0x7D** | 13 | 21 | 8 / 4.00 | + the walk's own swing waits for the facing |

`zfixO` is the state of the code: 81 blade presses with 30 off-face became
**51 with 1**. Read it screen by screen, not on the total — **0x7B went from
3-4 hits and 1.50-2.01 hearts to one hit, 0.50, and Link *healed* there**
(2.50 → 3.00 in), the first heart this walk has ever banked on the coast, and
0x7C went 2185 → 1154 frames. The eight hits did not go away. They moved
down the coast to 0x7C and 0x7D, and half of them are now the Zora
(`fireball_or_statue_projectile` ×3, `rock_projectile` ×2, leever ×2,
tektite ×1).

### What the next sitting should not do

- **Do not score a change on where the tape died.** Every variant this
  sitting spent 8 hits and ~4.00 hearts and the leftover screen moved with
  the reshuffle. Read `hits_by_cause` per screen and the rung that owned the
  frame (`probe_contact.py` + the window census), not the rupee count.
- **Do not paint a firing line on the Zora.** The open question from last
  sitting is answered: the spit is *aimed*, not axial. Across five tapes its
  motion was 466 frames horizontal, 248 diagonal, 172 vertical, and it shares
  the Zora's row in only 247 of 886. There is no row to stand out of;
  distance and the duck are the only answers.
- **0x7C as a transit screen is not free.** `zfixN` (`--transit 0x7c`) cut it
  to 1425 frames and 4 hits but handed 0x7D four more: 10 hits, 5.00 hearts,
  9R. The money and the damage are the same twelve leevers.

The next lever is **time on a Zora screen**. 0x7C and 0x7D cost `zfixO` 2131
frames and six of its eight hits, and the hunt owns only 566 of them — the
rest is the hop, the duck and the scoops. A spit lands about once per 350
exposed frames and neither the duck nor a wall memory has ever changed that
rate; what has never been tried is arriving on those two screens with a lane
picked so the crossing is short.

Do not STATUS. **M5 Clean re-measured after all of this and is unmoved:**
`run_level1_complete.py --natural-entry --trials 1` → ok, `prefix_ok`, room
36, TF `0x01`, **end_frame 18909** — the same frame as the live 2/2 claim,
with `combat.py`, `hunt.py`, `common.py` and `path.py` all changed under the
walker. One trial, so it confirms the claim rather than replacing it.

## Previous sitting (2026-09-16) — the Zora is never a fight; the walk reaches 0x6F

**Never engage the Zora. Always dodge and run.** That is now a rung, not a
disposition, and it is the first thing the walk asks every play frame:
`OverworldPathController._spit_duck` steps off any *closing* unblockable shot
inside `_SPIT_DUCK_RADIUS` (96 px) before the evader, the hunt or the hop get
the frame. Nothing below that line can answer a `0x55` — `prey.SKIP_TYPES`
never chases a Zora, `beam` never shoots one, `behaviors.shield_blocks` says
the small shield does not stop the spit, and `threat.assess` scores a muzzle
that has not launched as *safe* because it is not moving. The old rung that
was supposed to cover this (`evade_shot_over_sword`) never got the frame:
`evade_yield_to_sword` owned **687 of 6384** walk frames (`zhit1`) because
the leever screens keep a body in the blade box almost continuously, and
those are the same screens the Zora shares.

`common.walk_or_swing` also stops *turning* toward one: `prey.SKIP_TYPES` now
filters the nearest-enemy hint and `_off_axis_face`, so a Zora can never own
Link's face. A free swing the travel direction was already pulsing still
lands; that costs nothing and is not engagement.

### The dodge was walking into the shot

`common.answer_projectile` sidestepped perpendicular to the **travel** axis
and then *flipped the step at the screen edge*. On a wall that reverses it
into the shot: live 0x7C (`zhit1` f=5806), Link pinned at x=16 walking DOWN
with the spit 19 px east and 9 px south, stepped RIGHT three frames running
and took it. It now crosses the bearing to the *shot* (`common.perpendicular`,
moved out of `hunt` so one geometry serves both layers) and answers `None`
when neither side has room — the push is honest, a step that closes the gap
is not.

`perpendicular` also stopped asking "does one 2 px step stay in the box" and
started asking "is there a **pad** of room this way": a sidestep only clears
a hitbox once Link has walked `MIN_DODGE_BODY`, so a side with 3 px of wall
left is not an escape. That is the old edge flip stated as what it was for,
and unlike the flip it still answers when Link is already inside the margin.

### Two things a shot dodge must not do

- **Press into terrain.** `_EVADE_BOUNDS` is the scroll rectangle and knows
  nothing about the coast's rocks. Live 0x7B (`zhit2` f=4684): eight frames
  of UP at (48, 133) against a rock while a leever closed 12 px → 8, and
  that was the walk's *first* hit — at full health, streak on 10. It is the
  most expensive frame on the corridor: one `$0670` chip takes the sword
  beam away, so every screen after it is melee, and the forced 5-rupee at
  ten kills dies with it. So the wall is **measured**: `_SPIT_DUCK_STILL_CAP`
  frames that move Link nowhere (and are not `link_busy` — the ROM pins him
  for the whole sword animation) retire that direction, keyed by
  `(direction, 16 px cell)`. Screen-wide was worse than nothing: `zfixB`
  wrote off both sides of 0x7B and then took **eight hits on one screen**.
- **Walk through a body.** The duck is a walk. `_body_first` hands the frame
  back when a body is already inside `MIN_DODGE_BODY`, where a sidestep
  cannot clear it anyway and the blade and the peel are the real answers. A
  body *further out* than the shot still yields to the shot — that part of
  `evade_shot_over_sword` was always right.

### The tapes

All unassisted `--through pre-l1`, `scratch/probe_screen_tables.py`.

| tag | leftover | R | kills | hits / hearts | note |
|-----|----------|---|-------|---------------|------|
| `zbase1` | died 0x7E | 12 | 26 | 7 / 3.50 | committed + lane no-gain |
| `zfixA` | died 0x7E | 11 | 19 | 6 / 3.00 | duck in, fireball hits 3 → 1 |
| `zfixB` | died **0x7B** | 8 | 15 | 8 / 4.00 | screen-wide wall memory: don't |
| `zfixD` | **0x6F** (walk ok) | 12 | 32 | 9 / 4.52 | first arrival by committed code |
| `zfixE` | died 0x7D | **19** | 23 | 8 / 4.00 | `_body_first`; 0x7B 3.01 → 1.50 |

`zfixD` is the first pass where `bomb_walk` returns **ok** — the walk reached
0x6F, which `pre_l1_anyrow1` did once with code that no longer exists and the
committed walk had never done. `zfixE` then banks **19 of the 20 rupee**
price: 0x7B 6R → 7R at half the damage, 0x7C 0R → 7R. It dies on 0x7D at
0.48 hearts, one rupee short, with `duck_wall_7d_up` / `duck_wall_7d_down` —
both sides of the bearing are rock in that cell, so the rung correctly stops
claiming frames and the shot lands.

**The gap is one heart, not one rupee.** 0x7B and 0x7C are the money (12 row-1
leevers, 0.891 R/kill with the streak in it) *and* the damage, and the walk
arrives on the last half-heart either way. The next lever is the one the
census keeps naming and nothing has touched: `hunt_*_slash_recover`. Link
swings at a body inside the contact pad but **not** in the sword hitbox
(`_at_contact` accepts `pad <= MIN_DODGE_BODY` on its own), misses, and the
ROM pins him for the animation while the body closes the last 8 px — every
leever and octorok contact in `zhit2` / `zhit3` has that shape.

Do not STATUS. **M5 Clean re-measured after these changes and is unmoved:**
`run_level1_complete.py --natural-entry --trials 1` → ok, `prefix_ok`, room
36, TF `0x01`, **end_frame 18909** — the same frame as the live 2/2 claim,
with `path.py` and `common.py` both changed under the walker. One trial, so
it confirms the claim rather than replacing it.

## Previous sitting (2026-09-16) — `pre_l1_anyrow1` is not this walk

**The walk did change.** `pre_l1_anyrow1` (the only tape that ever reached
0x6F) was written 10:42:39. `overworld/hunt.py` was edited 10:51:03 and
`overworld/path.py` 10:56:50, and both landed in `7b364caf` at 11:18 —
strictly between that tape and `pre_l1_topup1` (11:12:45). The fixes priced
off anyrow1's own census went in *after* it and were never re-measured
unassisted. So "the one pass that arrived" is not a statement about code
that exists, and the committed walk has arrived on 0x6F **zero** times.

`pre_l1_repro1` re-ran the committed walk: 10211 frames, died 0x7D, 6R, 21
kills — the `pre_l1_topup1` report byte for byte, including `end_frame`.
The emulator is deterministic and the walk is reproducible; the difference
from anyrow1 is code, not noise. The 10:41 `hunt.py` / `path.py` are gone
(no stash, no dangling blob from 2026-09-16), so anyrow1 cannot be restored
by reverting — it has to be re-earned.

| | anyrow1 (gone) | committed | + lane no-gain |
|---|---|---|---|
| leftover | 0x6F, `shop_p_hunt_settle` | died 0x7D | died **0x7E** |
| rupees | 14 | 6 | **12** |
| kills | 28 | 21 | **26** |
| walk frames | 8238 | 9263 | **6384** |
| hearts spent | 5.996 (12 hits) | 3.496 (7 hits) | 3.496 (7 hits) |

The heal caps are **not** the regression. `HUNT_PICKUP_RADIUS` 400 and
`HUNT_HEAL_MAX_FRAMES` 100000 (`pre_l1_uncap1`) died *earlier*, on 0x7C
with 8R, and 0x79 still cost 1029 frames rather than anyrow1's 1037 — so
the flip that separates the two tapes is not in that pair. Reverted.

### 0x7C was a two-pixel stand-off, not an alternation

`scratch/probe_walk_trace.py` (tag `w1`) runs the committed walk unassisted
and records `(frame, screen, x, y, reason)` every frame — `reason_by_screen`
is a histogram and cannot say whether Link moved while a rung owned him. It
reproduces the walk exactly (9263 frames, 0x7D, 6R). On 0x7C:

- **3593 of 4301 frames at x ∈ {24, 25}, y=109** — the far *west* of a screen
  hop 5 crosses eastward. 1797 frames on x=24, 1796 on x=25.
- The peel steps LEFT, the plain push steps RIGHT: `hop_lane` 1795 against
  `hop` 1982, one pixel each, for ~60 minutes of game time.
- `_grinding` ends it at 4000 frames by dropping the lane branch — and Link
  then walks x=25 → 240 in **215 frames**. The screen was always ~220
  frames wide. The stand-off was the whole bill.

Neither existing cap can see it. `_OCCUPIED_LANE_STEER_CAP` counts
*consecutive* non-travel steers and every push frame takes the `not blocked`
early return, which zeroes it; `_lane_stand` is zeroed on the same returns;
`track_stuck` sees a Link who is moving. Only travel-axis **progress**
separates a peel going around a body from a tug-of-war, so that is what
`OverworldPathController._lane_no_gain` measures: lane frames this
`(hop_index, screen)` has owned since the best pixel it has reached toward
the exit, capped at `_OCCUPIED_LANE_NO_GAIN_CAP` (120 — a full vertical
traverse at 1 px/frame with nothing to show for it), then latched off for
that visit. The latch is per visit on purpose: handing the branch back on
the first pixel the push wins restarts the stand-off one pixel east, and
215 px at one cap per pixel is slower than the 4000-frame floor it beats.

`pre_l1_nogain1`: 0x7C **4301 → 923** frames, `hop_lane` 1795 → 61,
note `lane_nogain_5_7c`. The freed frames bought a `scoop_rupee` 66 on 0x7C
that had never run. Walk 9263 → 6384 frames, 6R → 12R, 21 → 26 kills, two
screens further. Same 7 hits / 3.496 hearts — it dies later, not softer.
0x77–0x7B are frame-identical to the committed walk; the change only fires
on 0x7C.

**Still dies, now on 0x7E** (215 frames in, 12R). Damage is
`fireball_or_statue_projectile` ×3, `leever` ×3, `octorok_blue` ×1. The
projectile is the 0x55 Zora spit and **the documented hole is still open**:
`dungeon/threat._FACING_AXIS` is `{0x08, 0x04: col, 0x01, 0x02: row}` and a
Zora reads `0x03` (Right|Left), so `firing_axis` returns None and
`in_firing_line` has still never returned True for one. `_FACING_SIGN` is
the second half — 0x03 has no sign, so adding the axis alone changes
nothing; an ambiguous facing has to mean *either side of the row*. Measure
the byte live before painting that.

Do not STATUS. M5 18909f not re-measured.

## Previous sitting (2026-09-16) — `bomb_topup` has live frames

`scratch/probe_topup.py`: assist ON, hunter off on the walk (the geometry
path that already reached 0x6F), then `RupeeTopUpController`. Glance after
t3: play **0x6F `(0,141)`** mode 5 TF `0x00` keys 0 bombs 0 rupees **2**
health **`0x22` 3/3** lo==hi. `progression_writes=0` / `capacity_writes=0`.
Not a Clean claim. Unassisted leftover is still `pre_l1_topup1` (died 0x7D);
this sitting did not re-run `--through pre-l1`.

| tag | 0x5F | 0x6E | kills | R | what it says |
|-----|------|------|-------|---|--------------|
| t1 | 87f, peak_live 0 | 106f, peak_live 0 | 6 (0x6F only) | 0 | back hop retraced the arrival edge |
| t2 | 1091f, 7 kills, 2R | 1197f, 2 kills, 214 occupancy_stand | 15 | 2 | hold works; 0x6E bush maze stands |
| t3 | 1091f, 7 kills, 2R | 1120f, 4 kills, 0 occupancy_stand | 17 | 2 | occupancy_stand now walks inward |

The retrace is structural. Hop 0 is UP to 0x5F; the first play frame on 0x5F
is y≈221, which is **not** the UP arrival edge (`y<70`), so hop_index
advances onto the DOWN home hop. Hunt then skips because y>200 **is** the
DOWN arrival edge, `recover_off_edge` allows DOWN, and the hop walks home
the same frame the wave would have spawned. 0x6E east is the same shape.
`RupeeTopUpController._extra_hop_action` holds (`topup_hold`) on a back hop
until `hunter.done`, and treats `occupancy_stand` as no claim so the inward
step is the sand corridor the neighbour probe already walked.

Both neighbours now get a 600f hunt. Both **retire** with bodies left
(`hunt_budget_5f` / `hunt_budget_6e`). 17 kills paid 2R (`streak_best` 5,
6 hurt_events). Two one-shot six-body waves cannot bank a 20R pack from a
0R arrival without the 10-kill 5-rupee. The top-up is the 19R-plus-one
gap, not a farm. The walk still has to survive to 0x6F with the corridor's
rupees.

Do not STATUS. M5 18909f not re-measured (this sitting did not touch
`path.py` / `tracking.py`).

## Previous sitting (2026-09-16) — the frames, not the tuning

`OverworldPathController.step` now censuses the *stem* of every
`FrameAction.reason` per screen (`report()["reason_by_screen"]`). That census
is what this sitting is: four stalls were found by reading it, none of them
by tuning a number.

| Screen | Before | What owned the frames | After |
|--------|--------|-----------------------|-------|
| `0x7B` | 1863f, 6 hits | `hop_ay` 394 — aligning to a row nothing needed | 480f, 1 hit |
| `0x7C` | 24877f timeout | `hunt_slash_recover` 19965 — one body holding the contact rung | crossed |
| `0x7C` | 24914f timeout | `hop` 12459 / `hop_lane` 12455 alternating one frame each | 4197f |
| `0x6F` | died f593 | `shop_p_hunt_settle` 593 — an idle *stand* in guard | finishes |

**The align was the big one.** `probe_coast_lane` already said 0x7B, 0x7C and
0x7D scroll east from **every** row; the hop table still carried
`align_y=131/130/130` across them. On a leever screen that is not a lane, it
is a vertical shuffle in a swarm. Those three hops now carry
`SCREEN_ANY_ROW_BAND = (77, 205)` — the sweep's own extent, so the push never
corrects — and the natural drift lands inside `SCREEN_7E_EAST_BAND` anyway.
`pre_l1_anyrow1` walked all nine hops and **arrived 0x6F for the first time**:
28 kills, 14R.

Four structural caps went in with it, each priced by the census above:

- `ScreenHunter._strike_budget` — the contact strike was the top of the ladder
  with **no budget at all** (not `TargetBook`, three rungs down; not the
  screen budget, also below it). One body that will not die owned 24877
  frames. Identity is `(slot, type)`; the frames also spend the screen budget.
- `OverworldPathController._grinding` — per `(hop_index, screen)` frame
  budget (`DEFAULT_HOP_SCREEN_MAX_FRAMES = 4000`). Past it the optional rungs
  (scoop, hunt, occupied-lane) are switched off and the push is all that is
  left. The local caps do not compose: a lane steer that picks the travel
  direction resets its own counter, and `track_stuck` sees a Link who is
  moving.
- `destination_hunted` returns True in guard — `_after_hops` answers a
  declining hunt with an **idle**, and the hunt declines for the whole guard
  branch, so on the destination screen the two met as a 2400-frame stand.
- `HUNT_PICKUP_RADIUS = 72` / `HUNT_HEAL_MAX_FRAMES = 120` — 0x7E is four
  `octorok_fast` plus a Zora and one pass spent 240 frames of `hunt_heal`
  and 179 of `hunt_scoop` crossing it for one heart and 2R, taking five of
  twelve hits doing it.

### Picking the rupees up

- `ScreenHunter.cleared` is now separate from `done`. `done` means "stop
  chasing here"; a budget **retire** lands in `done` with the wave still
  walking. Only `cleared` scoops money — that is the 5R-on-the-floor gap
  (`pre_l1_beam4`: 24 dropped, 19 banked): the kill that empties a screen
  drops on the frame the screen goes `done`, and `done` used to collect
  `heal_only`. `screen_table` stopped calling a retire "cleared" too.
- `combat.heal_wanted` replaces `filled_hearts < heart_containers` in
  `path._rupee_scoop`. That nibble is whole hearts *minus one*, so the old
  test was true at full health on every container count and the walk detoured
  for hearts it could not bank.
- A heal rung (`hunt_heal`) sits **above the beam**: one `$0670` chip is
  exactly what takes the beam away, so on every frame the heal can claim, the
  shot below it is already dead. `scoop_heal_radius` is 96 where rupees keep
  48 — a heart is worth crossing a screen for and a rupee is not.

### Arrive short → fight next door (`overworld/topup.py`)

`bomb_topup` is a new stage between the walk and the buy. It is a no-op on any
pass that is not short: `_at_stop` is `rupees >= price` **on the shop screen**,
so it finishes on frame 1 when the walk already banked the pack.

Not the corridor behind it. `overworld.respawn` is the ROM's rule and it is
decisive here: `RoomHistory` is six slots, so on arrival at 0x6F the five
screens behind it are all still in the ring and none of their waves come back.
Walking back gives nothing until the **sixth** screen (0x7A) — twelve screens
of leevers and Zoras for one respawn.

The fresh fights are the neighbours the walk never entered, and both are now
measured out **and back** (`scratch/probe_6f_neighbours.py`, tag `n4`: one
boot, the real hop table to 0x6F, state saved on arrival, every candidate
row/column restored, pushed, censused, pushed back):

| Exit | Lands | Lanes | Out | Home | Wave (peak 6) |
|------|-------|-------|-----|------|----------------|
| `0x6F` UP | `0x5F` | every column 72–232 (align clamps at x=128) | 193–240f | DOWN 85f | 3 octorok_fast, 2 octorok_blue, 1 zora |
| `0x6F` LEFT | `0x6E` | y 93–197; **77 / 85 / 205 dead** | 155–377f | RIGHT 106f | 3 moblin, 2 octorok_blue_fast, 1 moblin_blue |

0x5F is first in the table: octoroks are two wooden hits and the ROM's drop
table pays them, where 0x6E is four moblins. Both are six-body waves — the
same size as the corridor screens that cost the walk its hearts, so this is a
real fight, not a lap of an empty screen.

Live under assist, hunter-off walk: `scratch/probe_topup.py` t3, 17 kills,
2R, both neighbours hunted, still short. See the sitting above. The
unassisted walk has to survive to 0x6F for this to close the pack.

## Current walk (2026-09-15)

`--through pre-l1` is sword → `overworld/shop_p7.py` hunting walk → 0x6F
buy. Screens: `0x77 → 0x78 → 0x79 → 0x7A → 0x7B → 0x7C → 0x7D → 0x7E →
0x7F → 0x6F`. Overlay Map-1.png gives the screen sequence; it does **not**
give a row that walks. `0x4A` is the later arrows cave
(`overworld/bomb_shop.py`). Do not join via `0x68` / `0x5C` maze / `0x5E`
candle.

### The east lanes are measured, not painted

`scratch/probe_coast_lane.py` (tag `l1`, 2026-09-15): one boot, the emulator
state saved on arrival, then 17 candidate rows a screen — walk to the row,
hold EAST, did it scroll?

| Screen | Rows that scroll east |
|--------|-----------------------|
| `0x79` | **165 only** (the south beach) |
| `0x7A` | **133 and 141 only** — 125 and 149 are dead, and ≤117 / ≥157 are not even reachable from the west mouth |
| `0x7B` | every row (Link converges on 133/141 anyway) |
| `0x7C` | every row |
| `0x7D` | every row |
| `0x7E` | 117, 125, **141 and below**; **133 is dead** |
| `0x7F` | none — the hop out of `0x7F` is UP into `0x6F` |

Two of those rows are the walk timeout. `align_y` carries `y_tol=5`
(`overworld.common.align_and_push`), so the painted 131 on `0x7A` **accepts
y=126**, which does not scroll: the 2026-09-15 retest sat 27501f at y≈126
pressing RIGHT with 53 occupancy misses. `0x7E`'s painted 130 accepts the
dead 133 the same way. Both are now `y_band`s of the measured corridor
(`SCREEN_7A_EAST_BAND`, `SCREEN_7E_EAST_BAND`) — a band is the honest shape,
because a point plus a tolerance reaches outside what was measured.

Do not BFS the `OccupancyWalker` across these screens to find a lane. `0x78`
is a tree maze and the 1px learned grid walled in its own start cell after
1173 frames (probe `c1`): the sweep has to be a row sweep from a restored
state.

Unassisted leftover (`pre_l1_beam3` / `pre_l1_beam4`, reproduced 2/2): died
**0x7C `(192,85)`** at 7500f on hop 5, **19R**, 24 kills, streak best 17,
2.996 hearts over 6 hits. One rupee short of the 20R pack, with 5R of the
24R dropped still on the floor — the scoop is the gap, not the kill rate.
0x6F has not been arrived live. Stop is `ADDR_BOMBS >= 1`, not arrival on
the shop screen.

| | retest (before) | measured lanes | + the sword shot |
|---|---|---|---|
| leftover | timeout 0x7A | died 0x7D | died 0x7C |
| rupees | 2 | 5 | **19** |
| kills | 8 | 18 | **24** |
| streak best / resets | — | 6 / 6 | **17 / 3** |
| hearts spent | — | 3.996 (8 hits) | **2.996 (6 hits)** |
| beam fired / aimed / ready | — | 0 / 0 / 539 | **47 / 567 / 3016** |

Remaining damage is **4 of 6 hits from `0x55`** (the Zora spit), plus two
leevers. The bodies are no longer the bill.

### Do not (this sitting)

- ButtonsPressed is an edge. Hunt `_a_edge`: press A, then idle.
  Held A does not re-swing.
- Travelling frames offer `ScreenHunter.take_beam` **above** stall-escape
  (`path._do_hop`; 600f commits used to zero the weapon).
- Zora facing `0x03` is missing from `threat._FACING_AXIS`. The `0x55` spit
  is that hole, not a shop_p7 special case.
- Scoop vs restock: `scoop_rupees` vs `need_rupees`. Coast scoops; `need_rupees=0`
  so no 0x78 farm loop. `laps=0`. Do not poke Food/bombs/keys.

`SHOP_P7_HOPS` is measured (`SCREEN_79_BEACH_Y`, `SCREEN_7A_EAST_BAND`,
`SCREEN_7E_EAST_BAND`), not the overlay. `take_beam` then scoop sit above
`_stall_escape`. 5R on the floor is still the gap.

## Do not walk (measured traps)

ZD counts screens on the 16×8 grid. Two of those counts are not corridors.

| ZD text | Grid decode | Why it fails |
|---------|-------------|--------------|
| 0x79 overlay centre / inland joins | `0x79` y≈125; `0x68`/`0x5C`/`0x5E` | Overlay centre lane is a rocky bowl (no east cell `x>=232`, no north scroll). `0x6A`/`0x6B` south are tree walls; `0x7B` north is a bomb wall. `0x68` east from x=48 is a bush wall. `0x6C` east is dead. `0x5E` east is a tree wall. The live walk does not take these; it skirts 0x79 on the beach |
| After candle, down then left two, climb to White Sword | `0x0C` → `0x1C` → **`0x1B`** → `0x1A` → `0x0A` | `0x1B` is Lost Hills (wraps all four ways; 4th UP is L5 `0x0B`). Bypass: `0x0C` → `0x1C` → `0x2C` → west → `0x1A` → N `0x0A` |
| Heart-1 sidequest "up 2, right 2" from `0x7B` | `0x7B` → `0x5B` → **`0x5C`** maze | `0x5C` needs `LEVEL2_5C_MAZE_WAYPOINTS`. Do not treat as a free RIGHT |

`0x67` is a dead-end **from start** (`0x77` north). Coming onto `0x67` from the
east (`0x68` west) is the ZD 30R bomb wall, and is fine.

## Validated destinations

Catalog names are `overworld/locations.py`. "Walk" is what we would actually
drive. "Grid" is ZD's screen-count, even when the corridor is blocked.

| ZD § | Dest | Catalog | Open | Walk | Notes |
|------|------|---------|------|------|-------|
| 1.1 | `0x6F` `shop_p7` | `CAVE_SHOP_ARROWS` | open | **Map-1 south coast.** `overworld/shop_p7.py`: 0x79 beach east into 0x7A, then `0x7B`…`0x7F` UP `0x6F`. `0x4A` is the later arrow cave, same shop family, not this errand | Bombs 20R. Cave mouth `(48, 77)` |
| 1.2 | `0x7B` `heart_l8` | take-any | bomb N wall | from `0x6F` D1 L4 (`0x7F`→`0x7B`) | 4 HC. Approach from the east so we never enter 0x79 |
| 1.2 | `0x2C` `heart_m3` | take-any | bomb lower-right of center rock | after NE rupee stops | 5 HC. Same stand as IGN |
| 1.3 | `0x0F` `rupees_100_p1` | 100R | secret N wall of `0x1F` | `0x2C` → `0x2D` → `0x1D` → `0x1E` → `0x1F`, hug N wall | IGN's "NE corner past the gambling den" |
| 1.3 | `0x0E` `letter` | letter | open | `0x1F` → `0x1E` → stairs N | Needed for the potion shop. No pickup controller yet |
| 1.3 | `0x0C` `shop_m1` | `CAVE_SHOP_CANDLE` | **open** | `0x0E` → `0x1E` → `0x1D` → `0x0D` → `0x0C` | ZD candle shop. Better than IGN's bomb-open `0x66` and better than the long `0x5E` L8 corridor. 60R |
| 1.3 | `0x0A` `white_sword` | white sword | open, **5 HC** | **not through Lost Hills** | Blue Lynel on the 0x1A→0x0A climb |
| 1.4 | `0x48` `rupees_i5` | rupees | secret / burn | we already walk 0x48 on L1 | 30R burn, top-right bush |
| 1.4 | `0x47` `heart_h5` | take-any | burn 5th bush from the right | `0x48` LEFT y=141 | 6 HC. Pocket measured (`HEART_H5_*`) |
| 1.4 | `0x46` `shop_g5` | `CAVE_SHOP_ALT` | burn corner bush | `0x47` LEFT | Magical Shield **90R**. ZD: bait is also on this counter (buy later, not now) |
| 1.5 | `0x4A` `arrow_shop` | `CAVE_SHOP_ARROWS` | open | existing L2 prefix | Arrows 80R. Buy controller exists. Farm is still the gap |
| 1.5 | `0x6B` `rupees_100_l7` | 100R | secret / burn | `0x4A` east then south | Third-column lower bush |
| 1.5 | `0x67` `rupees_h7` | rupees | bomb N wall | `0x6B` L4 onto 0x67 from the east | 30R. Do not reach this by walking N from start |
| 1.5 | `0x64` `potion_e7` | potion | secret | `0x67` L3 | Show the Letter. 2nd Potion |
| 1.6 | `0x62` `rupees_100_c7` | 100R | burn, 3rd bush from top in the center | from the south-west 30R | IGN's brown-shrub 100R, ZD has the stand |
| 1.6 | `0x34` `special_shop_e4` | bait_or_blue_ring | Armos, **top-middle** | `0x51` R3 U2 | Blue Ring 250R. Pin slot order before a buy. Do not poke `ADDR_FOOD` |

Open-method mismatches (ZD vs catalog), live-pin before a hop:

- `0x3D` `rupees_n4`: ZD "right Armos 30R", catalog `OPEN_BURN`
- `0x56` `gamble_g6`: ZD burn 10R, catalog `OPEN_BOMB`
- `0x51` `gamble_b6`: ZD burn 10R, catalog gamble

## Order (ZD, not IGN)

Sword → farm while walking to `0x6F` bombs → `0x7B` heart → `0x2C` heart (5 HC)
→ `0x0F` 100R → Letter → candle `0x0C` → White Sword `0x0A` → `0x47` heart
(6 HC) + 90R shield `0x46` → arrows `0x4A` if 80R → potion `0x64` → Blue Ring
`0x34` → L1 `0x37`.

IGN bought the candle at `0x66` (bomb-open, next to start) and the White Sword
before the burn heart. ZD's NE-coast cluster (100R, Letter, candle `0x0C`,
White Sword) is one trip and skips `0x66`.

## Wiring

Dedicated `--through pre-l1`. Not spliced onto `level1_survival_tf_stages`.
`gathering.py` is the Composer row (sword + walk + buy). M5 18909f stays
the wooden 3HC oracle. Re-measure L1 after this prefix greens.
Hop dest and screen sequence come from Map-1.png (`overworld.zd_map`).
Live 0x79 east is the beach (`shop_p7.SCREEN_79_BEACH_Y`), not the overlay
centre lane and not the L8/candle corridor. `SHOP_P7_HOPS` is that
measured table (beach + the two east `y_band`s), not painted centre
rows. `take_beam` then scoop sit above `_stall_escape` on the hop
ladder. Coast `scoop_rupees=True` with `need_rupees=0`.

The 3HC L2 door suffix still dies on 0x4C even with evade-on (`rr-8t4.4-residual`
2026-09-14 census): last playable `(121,133)` hp `0x30` 0/4. Two whole hearts
were already gone. 6 HC would have had budget left; occupied-lane on hop 5
is still required so Link does not walk onto the body.

## ROM facts (any hunting walk)

These hold on any hunting walk. Corridor-specific numbers from the old
inland 0x4A prefix are labeled as such; they are not this dest.

**ButtonsPressed is an edge.** `Link_HandleInput` (`Z_05.asm`) wields on
`ButtonsPressed AND #$80`. `ButtonsPressed` is the edge (`Z_07.asm`:
`new EOR ButtonsDown AND new`), so a **held** A swings once and never
again. `ScreenHunter._strike` presses A one frame, then idles. The release
frame is an idle, never the direction. `_approach`'s blocked-align fallback
goes through the same edge.

**`$066F` lo nibble is whole hearts minus one.** `0x22` is 3/3.
`ram.whole_hearts` is the honest read. `filled_hearts` keeps the raw nibble
because the L1 chain is frame-perfect against it. `hits_taken` watches
`$066F` only; a wooden chip lands in `$0670` (`$80`) and never moves the
whole-heart byte. Read `hunt.report()["damage_taken"]`.

**`Link_BeHarmed` zeros `$50` / `$627`**, then subtracts damage and grants
`$04F0=24`. Survival assist writes `$0670` back to `$FF` the same frame, so
`hits_taken`, `damage_taken`, and `assist.damage_events` all read 0.
`--through pre-l1` forces assist off even if the caller passed one.

**20R pack is 36 unbroken kills.** `$0627` counts every kill; only
`Link_BeHarmed` zeroes it. `$0050` caps at 10 and any forced drop zeroes it.
`$0627 == 16` is tested **before** `$0050 >= 10`, so the 16-kill fairy
spends six kills of 5-rupee progress. A clean streak pays at 10, 26, 36, 46
— not 10, 20, 30. 20R is 36 unbroken kills in expectation, 46 if every
random roll cancels.

**Overworld waves are one-shot.** `ModifyObjCountByHistoryOW` clears a
screen's kill flags only when it is **absent** from the six-entry
`RoomHistory` (`$621`) and those flags read 7.
`RunCrossRoomTasksAndBeginUpdateMode` appends a room **only if it is not
already in the history**. An out-and-back evicts nothing at any depth. A
lap needs 7 distinct screens. (Old inland prefix, not this walk: the
`0x4A<->0x49` restock never was a farm for this reason.) `laps` stays 0
until a coast lap is measured better than one clean pass.

**Zora spit.** 195-frame surfacing clock on `$00AC`. The `0x55` shot sits
motionless on the muzzle for 17 frames, so `ObjectTracker` reads zero
velocity and `threat.assess` calls it safe for the whole dodge window.
Type `0x55` is not small-shield blockable. A zora's facing byte reads
`0x03`, which is missing from `_FACING_AXIS`, so `in_firing_line` has
never returned True for one.

**Scoop is the money gap.** Drops hit the floor. A kill census is not a
banked-rupee census. `CombatLedger.report` has `rupees_dropped` /
`rupees_left`.

**The flying sword is a real weapon on this corridor** (`zelda_i/beam.py`).
`Z_07.asm MakeSwordShot` puts the shot in object slot `$0E` when the blade
(slot `$0D`) reaches state 3, the low nibble of `$066F` equals the high
nibble, and `$0670 >= $80`. `Z_01.asm
CheckMonsterSwordShotOrMagicShotCollision` hands it the **blade's own** damage
points (`$10` wooden) and damage type (1), and its kill runs
`HandleMonsterDied` — so a beam kill ticks `$0627` / `$0050` and rolls the
same drop table as a melee kill. It is free reach, not a second economy.
Wooden `$10` one-shots a blue tektite, a red tektite and a red octorok (ROM
`ObjectTypeToHpPairs`: all `$10`); a blue octorok and a zora are `$20`, a
blue leever `$40`.

Measured live (`scratch/probe_beam.py`, 0x77, hp `0x22`/`0xFF`, all four
directions): **3.0 px/frame**, flying state `$10`, spreading state `$11` for
~22f, and the shot flies to the room bound — there is no distance decay, so
"range" is the screen. Muzzle offset from Link is ±19 px horizontally,
−16 UP and +30 DOWN. A live shot is its own cooldown: `MakeSwordShot`
returns early while slot `$0E` is non-zero.

**One chip takes the weapon away.** `Link_BeHarmed` subtracts damage from
`HeartPartial` *before* borrowing a whole heart, so a wooden octorok's `$80`
takes a full `$FF` to `$7F` — one below the gate. The beam is an *at full
health* weapon, which is why the hunt fires it early and why a heart on the
floor is now worth more than its buffer: it hands the weapon back.

**Reach is worth nothing to a layer that never spends the edge.**
`scratch/probe_beam.py --phase lane` (tag `b3`, the live coast walk): the
shot was up for **807** of 6218 frames and a body stood in a 9 px lane on
**361** of them — and `ScreenHunter.step` reached its own beam branch on
**262** and aimed on **0**. Those are disjoint sets. While Link is
*travelling*, the hop table owns the frame and `common.walk_or_swing` only
presses A at contact range; a body straight ahead in a lane is exactly the
geometry a hop produces. `ScreenHunter.take_beam` is the shot offered
to that layer from `_do_hop` before recovery.

**Where it sits in the hop ladder is the whole weapon.** Offering it *below*
the movement-recovery layers changed nothing at all (`pre_l1_beam2`: 539
ready frames, 0 aimed): probe `b4` caught a red octorok sitting 3 px off
Link's row 132 px ahead while one `hop1_escape` commit — `_stall_escape` is
600 frames — owned every frame of it. `take_beam` then scoop now run
above `_stall_escape` / occupancy / `unstick_wiggle` /
`recover_off_edge`, and below the reactive evader (`path._threat_action`,
which still runs first). It is self-limiting: `MakeSwordShot` refuses while
slot `$0E` is live, so a held lane costs one press per shot, not one per
frame.

**Stop is `ADDR_BOMBS >= 1`**, not arrival on a shop screen. Arrival cannot
tell a walk that banked 20R from one that banked one rupee.
