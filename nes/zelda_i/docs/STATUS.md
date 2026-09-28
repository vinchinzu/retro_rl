# Status — Zelda I

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M8 |
| Best verified result | Clean power-on to the **credits** in **312,940 frames** (`n9_credits`, 2026-09-28): triforce `0xFF`, **16 containers** (every heart container: 0x2C's take-any gives the heart now), one continuous session, 0 RAM writes, 0 deaths, no health refill. 21,823 frames under `clean_poweron_c12` (334,763) |
| Last verification | 2026-09-28 |
| Runtime class | Bronze |
| Intervention class | Clean (recording); the live policy uses ROM rollout lookahead, allowed for development (owner ruling 2026-09-28) |
| Evidence | [n9_credits.json](../recordings/n9_credits.json) (`run_survival_spine.py --clean --through level9-credits --save-points N9`): `ok=True`, `resumed_from` null, no assist, 0 inventory/progression writes, `set_state=31026` = `rollout_restores=31026`, 0 other loads. Recording: [n9_credits_replay.mp4](../recordings/n9_credits_replay.mp4) from [n9_credits_replay.json](../recordings/n9_credits_replay.json) (`replay_tape.py`): 312,940 frames, 0 desynced, 0 loads, 0 writes, 11.7x real time. Previous gate: [clean_poweron_c12.json](../recordings/clean_poweron_c12.json) (334,763f, 15 containers) and its replay MP4. |
| Not the gate | Survival tapes, resumed pins, offset evals, and any `--rollout` trial. |

## 2026-09-28 (later): the credits 21,823 frames sooner, every heart taken

`n9_credits` plays power-on to the credits in 312,940 frames (c12: 334,763)
with 16 containers. The gathering takes 0x2C's heart and buys the potion at
0x0D's shop, gets the White Sword straight off the candle through Lost
Hills, and skips pond trips at full health; the coast hunt, the NE tektite
screens and the White Sword leg run under `PolicyGuard`. Nine dungeon clears
whose exits are open or keyed and whose rooms hold nothing the route takes
are walked alive (L4 0x20 alone 7,496f -> 370f). Details and milestones:
[PRE_L1.md](PRE_L1.md). L3's 0x4D is the slowest room left (11,821f).

## 2026-09-28: Clean power-on to the credits, recorded from its own tape

`clean_poweron_c12` plays power-on → gathering → L1-L9 → Ganon → Zelda in
one session with no refill, no RAM write and no death. The Level 9 lanes
(shot guard, blade traps, 0x16 Patra bombs, Death Mountain walk, rollout-checked
Patra swings and `PolicyGuard`; details in [PRE_L1.md](PRE_L1.md)) closed the
gap from C10. Its 21,164 state loads are all rollout lookahead restores: the
controller saves the core, plays a few frames ahead, and restores the same
live frame. The owner ruled that allowed for development, so the gate
discloses them (`rollout_restores`) and fails only on other loads.

The published recording has none. Every spine run now writes a button tape.
`replay_tape.py` plays it into a fresh power-on with no controller and no
lookahead, and it checks the system RAM ($0010-$07FF) against the live run
on every frame. All 334,763 match. The replay encodes at 11.6x real time. The
live policy averages 5.7x, but three `PolicyGuard` stages in Level 9 run below
real time (the Patra join is 0.38x; `rr-yzb4`).

Under `docs/BENCHMARK_SPEC.md`, strict Clean allows no emulator-state
mutation during the attempt. The recording meets that; the live policy
meets it except for the disclosed lookahead.

The 18909f wooden-sword M5 (2026-09-14) is historical. Its standalone
recheck (`run_level1_complete.py --natural-entry`) is red at L1 0x33
(`0x33_needs_heart`, 2026-09-24); the gathered route above replaces it.

## 2026-09-25 (late): Clean power-on through L6 (Level 6 Triforce)

`clean_poweron83`: power-on → Level 6 Triforce (235,596f), no refill, no load,
no write, zero deaths, mode 18 in 0x0C with six shards (TF `0x3F`), Rod and
Magical Sword owned. What changed since the L5 gate:

1. **L6 reroute from the ROM door table** (`pin_probe.py --doors`): 0x78 and
   0x28 are walked (their exits are open), 0x28's east bomb wall reaches 0x29
   and the 0x19/0x09 key doors without the 0x18 Gleeok, 0x7a's key fight is
   skipped, and the post-Rod 0x29 walk goes straight down x=120 on the
   stepladder. 0x38, 0x09 and 0x3a blocks need every enemy dead (measured).
2. **Magical Sword before L6** (`overworld/magical_sword.py`): the ladder
   heart 0x5F and the raft heart 0x2F after L4's 0x67 rock (9 → 11
   containers), L5's heart makes 12, and the 0x21 grave (144,144) on the L6
   walk gives the sword. A what-if pin had shown the White Sword with 10
   containers cannot pay for L6, and the Magical Sword with 12 can.
3. **0x13's rupee rock only when short**: skipped at 153R (it cost 5.5h).

The L5 Triforce falls at 211,047f inside the same tape, so the Level 5 gate
holds on this code; a separate `--through level5` run was not made.
Worst Clean rooms left in L6: 0x3a (10h), 0x09 (5h), 0x19 (3.5h), 0x2c (3h).

## 2026-09-25: Clean power-on through L5 (Level 5 Triforce)

`clean_poweron76`: power-on → Level 5 Triforce (190,444f), no refill, no load, no
write, zero deaths, mode 18 in 0x14 with all 5 shards (TF `0x1F` / 31). Two fixes:

1. **Level 4 Room 0x40 west corridor:** `ROOM_40_SPEC.combat` occupancy bounds
   expanded from default `xmin=40` to `(32, 216, 77, 205)` with `occupancy_patrol=True`
   and `occupancy_from_tilemap=True`, preventing Link from pathfinding into a solid column
   when enemies spawn at x=32.
2. **Level 5 Room 0x26 stepladder latch:** When Link cleared Gibdos in 0x26 near
   the water moat, the ladder deployed at (200, 176) heading UP. `LatticeDoorWalker`
   in `dungeon/hop_controller.py` now latches the ladder release direction (`DOWN` onto
   the south corridor) until Link is completely off the ladder (`deployed_ladder` is None),
   preventing the 184<->185 oscillation.

Link traversed 0x26 west, cleared Pols Voice in 0x25, spent 1 key to enter 0x24,
whistle-shrank and killed Digdogger, took the heart container, and claimed the
Level 5 Triforce shard.

## 2026-09-25 (evening): Clean power-on through L4

`clean_poweron74`: power-on → L4 Triforce (152,416f), no refill, no load, no
write. Two changes: the L4 walk buys a blue potion at 0x64 keeping only the
bomb-pack price in reserve (it used to hold back the 80R arrows too, so it
never bought), and the Gleeok loop drinks at the last heart
(`dungeon.pause_select.drink_if_low`; it steps the env itself, so the stage
guard never ran). From the L3 Triforce, 12 RNG offsets: L4 Triforce 1/12
before, 6/12 with the potion alone, 12/12 with both.

The run then stops at 0x4A: 15R against the 80R arrows, 0 bombs. Manhandla
spent all six bombs, so the L4 walk bought a 20R pack after the 40R potion.
Money is the next wall (plan.md Next).

## 2026-09-25 (later): Clean power-on through L3

`clean_poweron69`: power-on → L3 Triforce (121,388f), no refill, no load, no
write; dies at the L4 Gleeok with 0.7 hearts (150,867f). L3 from its entry
pin now clears 9/9 RNG offsets (was 0/8): pond and a blue potion before L3,
potion drinks inside the boss suffix, Darknut flank strikes in every L3
Darknut room, 0x5A blade traps baited, raft-passage fixes, and a Manhandla
fight that dodges imminent fireballs. Details: [plan.md](plan.md).

## 2026-09-25: Clean power-on through L2 (gathering fixed)

`clean_poweron64` went power-on → gathering → L1 Triforce → L2 Triforce with
no health refill, no state load and no RAM write, then died in L3's 0x69
Darknut room (110,963f). The day before, Clean died at 40,801f on the White
Sword walk (`clean_poweron60`, same tape as CL45).

What changed (details in [plan.md](plan.md), "Done this sitting"):
- 0x2C's take-any gives the red potion; 0x47's container comes before the
  White Sword, so Link still has the five it needs.
- Overworld melee: a lattice-simulated peel (`common.body_escape`), swings
  only when the blade lands before contact (`ScreenHunter._swing_pays`), and
  the lattice shot escape for the hunter's Zora duck. Post-pond gathering
  over 12 RNG offsets: 0/12 → 11/12 reach the L1 mouth.
- Six stalls the new hit census exposed (0x48 cell flutter, 0x2D stairs
  lane, 0x28 stray cave, 0x7F help-drop bomb, potion guard B item, burn
  cells' B item).
- L3 0x59/0x69 Darknuts are struck from a non-shield side
  (`flank_shielded`): from the L3 entry pin, 7/8 offsets now reach the boss
  path (was 1/8); the boss suffix fails 8/8 (low hearts or `bombs=0`).

Last-heart death map on d1c42958 (`lasth_poweron60`): L1, L2 and L4 need no
refill; the gathering needed 2 (both fixed above); L3 needs 3.

The zero-poke Survival regression is red at L8 (`natural_credits_poweron67`,
HEAD edf97dd6: TF `0x7F`, 0x4C bomb wall with 0 bombs, rr-awh6). The L2
Dodongo failure of run 65 is fixed (bombs from a body length off the mouth:
66/66 saved pins).

## 2026-09-24 (latest): `--clean` was silently Survival; first real Clean runs

`level3/spine.py` removed `--clean` from `sys.argv` at import (since
81836486, 2026-09-10), so every `run_survival_spine.py --clean` run had the
health refill and pokes on. `clean_poweron40` "reached the credits" in the
same 289,154 frames and 341h of absorbed damage as the Survival tape. The
strip is deleted, and a test pins `--clean` through the imports.

With the refill really off, power-on dies in the gathering prefix:
`clean_poweron42` on the walk to the 0x39 pond (24,635f: seven half-heart
hits walking waypoints into octoroks and moblins). With the defend layer
below, `clean_poweron44/45` get past the pond, the 0x2C heart, the NE cluster,
the letter, the candle and 0x28, then die on the White Sword walk at 0x17
(40,564f). No heal comes between the pond and 0x0A, and Link arrives there
with 1 of 5 hearts. The separate M5 recheck above is red.

What changed for Clean (Survival runs use the same code):
- `ScreenHunter.defend`: strike, peel, shield and duck only. `OverworldPathController(defend=True)`
  consults it after the threat ladder, ahead of the hop ladder and any hand
  phase. The gather waypoints sat at the top of the hop ladder and walked
  Link into bodies with nothing able to veto.
- Gather walkers run `defend` + `evade` by default. No refill, 7 walks x 6
  offsets: stages survived 34/42 bare, 41/42 armed. Hand stages use `defend`
  per the same eval. `heart_7b`, `heart_47` and `white` stay bare.
- 0x34 (ring and bait): wake only the stairs Armos, from (80,125) facing
  LEFT, and wait above the statue row. The old hunt climbed x=64 and woke
  (64,160) on top of Link. Ring visit: 0/8 to 8/8. Bait visit: 5 hits to 0.1.
- A path in its hop phase that finds itself in a cave walks out
  (`stray_cave_exit`). A duck onto 0x56's open stairs held `ring` 17,716f.
- L7: `$066C` is the clock (`snap.clock`). 0x59 goriyas froze 12 px off
  Link's line: strike width 8 plus a lattice chase to `sword_stand`. 0x0D:
  wait out a clock drop, because a taken clock stops the Wallmaster ring.
- L2 Dodongo (`tf_spine.Level2DodongoController`): two swallowed bombs kill
  it, or one sword cut while it is stunned by a blast beside its head
  (ObjState 2). After a swallow (state 1) it stands still ~97 frames and no
  side takes a cut, so the fight now drops the next bomb at its mouth during
  that window, strikes a stun, and never reselects bombs or declares
  "out of bombs" on an empty bag while the Dodongo is stunned or hurt.
  14/14 saved pins pass; before, runs 46 and 48 spent all 7 bombs and timed out.
- 0x0A White Sword (rr-lkqf): the climb waits at the corridor foot until the
  Lynel is on the bottom band, stepping back to 0x1A to re-roll it when it
  comes near. The top-band walks hold while a Lynel sword shot (0x57) is about
  to cross the lane ahead: its beam flies across the lake. From the last-heart
  pins with no refill: `white` 1/6 to 4-5/6, `back_1a` 6/6 with 2 hearts
  lost to 0.
- L7 Recorder warp: a whirlwind landing on a door screen can carry Link
  into that dungeon (0x24 → L6 at 0x22). The warp now walks back out the south
  door and blows again, instead of failing the stage.
- Post-L8 walk, 0x38 north: ROM lattice to the x=112..136 cut first, as on
  0x17. A knock off x=120 after the bridge latch pressed UP into rock for
  10,387 frames.
- 0x47 burn: after a defend step the bomb cell re-walks to the exact stand
  (`_on_defended`). `heart_47` stays bare, because the Zora duck ended with
  the fireball landing on the flame press.

## 2026-09-24: continuous power-on to credits with ZERO inventory writes

`natural_credits_poweron39` (rr-ps7) achieved the first continuous power-on run
from boot through credits with zero inventory pokes or assist writes of any kind:
`ok=True`, `set_state_count=0`, 289,154 frames, all 8 Triforce pieces naturally
collected (`tf=255`), Ganon defeated, Zelda rescued, final credits mode 19.
- Wooden arrows (80R) bought naturally at 0x4A before L6 (`poke_wooden_arrows=False`).
- Bait (60R) bought naturally at 0x34 during post-ring gathering before L1 (`ADDR_FOOD` write retired).
- Bombs restocked naturally at 0x44 and 0x4A (`poke_bombs=False`).
- Blue Gohma 0x1E arrow fire gated on vulnerability and alignment, saving 19 rupees (4 shots vs 23 blind shots) to fully fund both post-L8 bomb packs for Level 9.
- `inventory_assist=None`. Survival health refill remains active (M5 Clean gate unchanged).

## 2026-09-24 (later): credits with no rupee write; final Patra aims

`blue_ring_full_poweron24` (92f1031d) went power-on to the credits in one
session, 0 state loads, 283,010 frames: the first credits run with no rupee
write. Its final Patra took 11,407 frames because the last eye's lap had
drifted below the room. `PatraAim` (`level9/patra.py`, rr-e59v) fits each
eye's lap and, once it drifts off the body, fires only on a predicted shot
hit; the lane stand now keeps its side and stands on walkable nodes. Run 31
(b8cc4ab6) replays run 24 frame for frame up to 0x52, then clears it in
3,362 frames: credits at 275,135. Survival, not Clean; bomb/key/arrow/Food
writes remain.

Last-heart (`--engage-hearts 1 --observed-damage-guard`, three pieces on
92f1031d): power-on to L6 0x09 with 6 + 1 safety refills, then L7 and L8
(0x1F included), then a timeout in L9 0x61 Patra: Link has no potion and
~11/14 hearts, so no shot (rr-6o39).

## 2026-09-24: no rupee writes; potions replace refills through L3

Survival still (the refill is on), but the wallet is never written now.
The gather chain opens the hidden rupee caves on and beside its walk
(`overworld/locations.py` `SECRET_RUPEE_CAVES`; payouts from the ROM cave
table at `$18610`: `$21`=30R, `$22`=100R, `$23`=10R) and pays the 250R Blue
Ring and the candle from play; the wallet caps at 255, so 0x62's 100R comes
after the ring and buys a red potion at 0x64. Continuous chain from the
power-on pre-L1 leave: 40,719 frames, 0 deaths, L1 mouth with Ring 1, a red
potion and 38R, no inventory write. Every rupee top-up is deleted (ring,
pre-L1 20R, L7 Bait 60R); only bomb/key counts and the L7 Food remain.

`PotionDrinkGuard` (every spine stage) drinks at the last heart and holds
the refill meanwhile; the walk from L3 to L4 restocks at 0x64. Last-heart
power-on (`--engage-hearts 1 --observed-damage-guard`, run 29, resumed from
save points after each fixed stall): power-on → L4 with **0 refills**, the
two drinks landing in L2 0x3E and L3's raft room; L4 stepladder → L6 heart
11 refills (5 at the last heart, 6 safety after a two-heart hit); L6 → L8
4 more. It stops in L8 0x1F, whose Darknut clear needs the full-heart beam.
Per-segment rows: [RUN_METRICS.md](RUN_METRICS.md).

## 2026-09-23: Survival power-on → credits, one continuous run

`run_survival_spine.py --through level9-credits --save-points Full --no-video`
went from power-on to the credits in one emulator session: `ok=True`,
`set_state_count=0`, 292,742 frames, 234 stages, TF `0xFF`, 12 containers,
Ganon and Zelda, final mode 19. Tape: `recordings/full_poweron11.json`.

This is **Survival, not Clean**: the health refill is on (768 hearts of
damage absorbed), bomb/key/rupee counts are topped up at declared gates
(`SPINE_*_RETOPUP`), and L7's Bait is the disclosed Food fixture. The M5
Clean gate above is unchanged. One run, not repeated yet.

How it got there: continuous runs exposed stalls that the resumed save-point
tapes hid (each fix moves every later frame). Almost every stall was a hand
walk disagreeing with the ROM, or two rules swapping 1-2 px each frame; the
fixes route through the ROM lattice (`dungeon/hop_controller.py`:
`stairs_step`, `block_push_step`, `door_nodes`, `ow_edge_band_step`,
`inland_lattice_step`) behind a stall gate. Heart containers are route
history, not a handoff gate.

## 2026-09-23 (later): lattice walkers, credits again — faster and steadier

`full_poweron27` (commit `b0a328e9`) went power-on → credits in one session:
`ok=True`, `set_state_count=0`, 277,687 frames (−15,055 vs `full_poweron12`),
TF `0xFF`, 12 containers, mode 19. Same Survival disclosure as below.

What changed: every walk plans on the ROM turn lattice (x%8==0, y%8==5).
Presses off it were the flutter source; flutter (1-2 px reversals, ledger in
`spine/ledger.py`) fell 17,942 → 8,178. Hand walks were replaced by shared
helpers in `dungeon/hop_controller.py` (`room_step`, `mouth_step`,
`ladder_release`/`release_action`, `exit_door`), cleared rooms sweep their
floor drops (bombs poked 82 → 67, rupees 9 → 0), and bomb walls retry a
dropped press. Runs 13–26 each stopped one stage later; every stall was
fixed from its save point. Metrics per run: [RUN_METRICS.md](RUN_METRICS.md).

Last-heart refill (`--engage-hearts 1`, refills = deaths prevented): power-on
→ L1 TF needed 2 refills; through L2, 4. The full run under it is bead rr-k3vj.

## What is open

Clean frontier: Level 4 (ladder rung C6 in [plan.md](plan.md)): the room
fights bleed ~6 hearts into the Gleeok. L6 is the wall after it (~10
last-heart refills).

## What is not written here

Per-hop Survival ledgers used to live in this file. The JSON files under `recordings/` still hold those runs. They are development tapes. They do not move the M5 gate. Lane notes under `docs/tasks/` are the same kind of record. Fixture pins with 15 hearts in too few containers are not Clean measurements. `scripts/audit_pins.py` is how a pin gets refused.
