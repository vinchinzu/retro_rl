# Status — Zelda I

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Clean power-on to Level 1 Triforce, triforce `0x01`, 18909f, 2/2 natural-entry |
| Last verification | 2026-09-14 |
| Runtime class | Bronze |
| Intervention class | Clean |
| Evidence | Historical `spine/clean_tip.py` row `l1_tf`; [level1_complete_isolated.json](../recordings/level1_complete_isolated.json). Current natural-entry recheck: [level1_complete_natural.json](../recordings/level1_complete_natural.json). |
| Not the gate | Gathering, Survival dungeon tapes, and any `--rollout` trial. `--through pre-l1` is the open prefix and has no Clean claim. |

The 18909f figure is the clean-tip oracle recorded on 2026-09-14. On
2026-09-24, the current checkout failed `run_level1_complete.py
--natural-entry --trials 2` twice at `clear33_key`, frame 9742: room 0x33
cleared, but the controller stopped with `0x33_needs_heart`. The current
natural JSON records this red recheck. The historical best is not a current
passing regression gate.

Re-measure with `scripts/run_level1_complete.py --natural-entry` and no health refill. Do not overwrite the oracle from a gathering or Survival run.

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

`spine/clean_tip.py` `next_open()` is `pre_l1`. Route and the one live tape are in [PRE_L1.md](PRE_L1.md). One flagged rollout trial bought bombs. The default walk is not accepted.

## What is not written here

Per-hop Survival ledgers used to live in this file. The JSON files under `recordings/` still hold those runs. They are development tapes. They do not move the M5 gate. Lane notes under `docs/tasks/` are the same kind of record. Fixture pins with 15 hearts in too few containers are not Clean measurements. `scripts/audit_pins.py` is how a pin gets refused.
