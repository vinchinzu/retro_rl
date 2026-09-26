# Plan — Zelda I

## Now: Clean credits as a health economy (2026-09-25)

Goal: power-on to credits, Clean (no RAM write, no state load), zero deaths.
Inventory is already natural (run 53: zero pokes); only the health refill is
left, so every remaining item below is a death point, not a resource.

### C8 green (2026-09-25, late)

`clean_poweron83` (c903cb4b) is the C8 gate: power-on → L6 Triforce, 235,596f,
TF `0x3F`, Rod and Magical Sword, 13 containers, 0 loads, 0 writes. The levers,
in payoff order: the ROM door table (0x78/0x28/Gleeok/0x7a/0x29 clears were
optional), the Magical Sword (coast hearts 0x5F/0x2F after L4 + the 0x21
grave; a what-if pin sized it first), and skipping 0x13's rock when the
wallet already pays. Next is C9 (L7). Re-run the zero-poke Survival credits
run first: L6 now spends one bomb at 0x28, skips 0x7a's key, and the L5/L6
walks carry the coast and grave detours.

| rung | stop | now |
|---|---|---|
| C1 | White Sword taken | **green** (`clean_poweron64`) |
| C2 | Blue Ring, L1 mouth 0x37 | **green** (64) |
| C3 | L1 Triforce (new M5 on the gathered route) | **green** (64) |
| C4 | L2 Triforce | **green** (64, TF `0x03`) |
| C5 | L3 Triforce | **green** (`clean_poweron69`, 121,388f) |
| C6 | L4 Triforce | **green** (`clean_poweron74`, 152,416f) |
| C7 | L5 Triforce | **green** (`clean_poweron76`, 190,444f, TF `0x1F`) |
| C8 | L6 Triforce | **green** (`clean_poweron83`, 235,596f, TF `0x3F`) |
| C9-C11 | L7, L8, L9 + credits | – |

A rung is green once one power-on `--clean` run reaches it (C5 also on
committed HEAD alone: `clean_poweron_h1`, L3 at 126,643f). Keep the zero-poke
Survival credits run green (`--no-pokes`) after each route change: **it is
red at L8** (`natural_credits_poweron67`, HEAD edf97dd6: TF `0x7F`, then
0x3E spends 4 bombs and 0x4C's wall has none, rr-awh6). The L2 Dodongo
(rr-pm7m) and the pre-L3 restock's hidden letter are fixed.

### Done this sitting (2026-09-25)

- G1: 0x2C take-any gives the red potion; 0x47's container comes before
  the White Sword (`walk_48` → `heart_47` → `walk_white` → `white` →
  `walk_back_48`). `white` is defended now.
- Overworld melee, scored on 12 RNG offsets from the pond pin
  (`scratch/eval_gather_clean.py`): post-pond gathering 0/12 → 11/12.
  * `common.body_escape`: the peel flies every input on the lattice
    against each near body (84 of 268 hits were the old "away" press).
  * `ScreenHunter._swing_pays`: swing only when the blade (frames 4-11 of
    the 13-frame pin) meets the body before any body touches Link.
  * The hunter's Zora duck uses `shot_escape` on the lattice.
- Stalls found by the census: 0x48 off-lattice burn cell flutter
  (`_nudge_dir`), 0x2D stairs lane (`OPENING_LANE_SLACK`), 0x28 stray-cave
  loop (`_on_stray_cave` restarts the corners), a help-drop bomb ending the
  coast walk on 0x7F (`CaveExitController` passes when not in a cave), the
  potion guard leaving B on the potion after a retry (burn cells now
  reselect their B item).
- Every stage report carries a hit census (`hits`: cause, action, 24-frame
  trail).
- L3: Darknut rooms 0x59/0x69 strike from a non-shield side
  (`CombatTuning.flank_shielded`, `_flank_strike`): 8 offsets from the L3
  entry pin, raft reached 1/8 → 7/8; L3 Triforce 0/8 → 0/8 (boss suffix).
- L3's cleared 0x5B now has a short natural five-rupee pickup stage. The
  continuous run collected it before leaving; this repairs the 75R→80R arrow
  budget without a wallet write. The post-L4 raft walk now forces a south
  dismount on 0x55 before turning east.
- L4 Gleeok uses the turn-node `GleeokStand` at `dy=30`; isolated replays pass,
  but the continuous run still reaches 0x13 at 0.71 hearts and dies before
  the fight.
- C6 (evening): the L4 walk's potion keeps only the bomb price in reserve,
  and the Gleeok loop drinks at the last heart (`drink_if_low`, moved to
  `dungeon/pause_select.py`). 12 offsets from `CL73_potion_restock_l3`
  (`scratch/offset_pins.py`; the L3 TF settle absorbs idle frames, so pin
  after it): L4 TF 1/12 → 12/12. `clean_poweron74` is the power-on proof.
- C7: Level 4 0x40 west corridor bounds expanded to `(32, 216, 77, 205)` under
  occupancy patrol, resolving x=32 node pruning hang. `LatticeDoorWalker` in
  `dungeon/hop_controller.py` latches the ladder release direction (`DOWN` off
  the vertical ladder) until Link is off the stepladder (`deployed_ladder` is None),
  eliminating the 184<->185 oscillation in room 0x26. Pols Voice in 0x25,
  Digdogger in 0x24 (Whistle-shrunk, sworded), heart container taken, and
  Level 5 Triforce piece collected. Continuous power-on run `clean_poweron76`
  reached Level 5 Triforce at 190,444f with 0 refills, 0 loads, 0 writes (TF `0x1F` / 31).

### L3 (same sitting, later)

From the L3 entry pin (9 RNG offsets, `L3o<n>_walk_pond_l3`) L3 went
0/8 → 9/9: the 0x39 pond on the L2→L3 walk, a blue potion at 0x64 when the
wallet keeps 40R, potion drinks inside the custom boss suffix
(`boss_combat.drink_if_low`), flank strikes on 0x5B/0x5C/0x69 (0x5C's hand
controller is gone), the 0x69 respawn cleared by the engine, 0x5A's west
blade traps sprung and waited out, keese cut in the raft passage, the
passage ladder fix, two bombs kept for Manhandla, and Manhandla dodging only
imminent fireballs (16 frames) and hands inside 14 px.

`clean_poweron83` leaves the Level 6 Triforce with 6.24 hearts. The
expensive rooms on that tape are 0x3a (10h), 0x09 (5h), 0x19 (3.5h), and
0x38 (2h). 0x78 and 0x28 are walked. 0x28 spends one bomb east into 0x29.
0x7a is skipped.

### Next, in order

Claim one spine bead. `bd ready -l zelda_i -l spine` is this list.

1. **C9, Level 7 (rr-rgum).** `--clean --through level7` from power-on.
   Food is already owned. Fix the first red stage from its `C9_` pin,
   then re-run the power-on.
2. **Zero-poke Survival credits (rr-k3vj).** `--no-pokes --through
   level9-credits`. Heart refill stays on. This does not block C9. It
   blocks the Level 8 bomb patch, because `natural_credits_poweron67`
   is from before the Level 6 reroute.
3. **L8 bombs (rr-awh6),** after that tape still dies at 0x4C with no
   bombs. `_3e_wall_bombed` clears on every screen change
   (`level8/north_column.py`).
4. **C10, Level 8 (rr-npv.4).** Blocked on C9 and on rr-awh6.
5. **C11, credits (rr-npv.5).** `--clean --through level9-credits`.
   That tape is the STATUS row.

Potion (rr-thlc), extra keys (rr-qb6w), and the 0x3a heart bleed
(rr-rgum.1) wait until a Clean tape actually dies there. They are not
on the spine filter.

## The Gathering (route order)

Zelda Dungeon calls this The Gathering. Order:

1. Wooden sword on `0x77`.
2. South-coast walk to bombs at `0x6F`. Stop when `ADDR_BOMBS >= 1`.
3. Heart at `0x7B`, the 0x39 pond, then **the red potion** at the `0x2C`
   take-any (left item, (88,149); 2026-09-25, was the container).
4. Northeast cluster: 30R `0x2D`, 100R `0x0F`, letter `0x0E`, candle `0x0C`,
   30R `0x28`.
5. Burn row before the sword: 30R `0x48`, heart `0x47` (5 containers), back
   up the x=120 cut to `0x28`, White Sword `0x0A`, back down to `0x48`.
6. Blue Ring at `0x34`, paid by the hidden rupee caves (`0x5B`, `0x6B`,
   `0x56`); 0x62's 100R after it buys Bait at `0x34`. Arrows are bought
   after L4, and 0x67 funds Level 9's second bomb pack.
7. The 0x39 pond again, then the Level 1 mouth at `0x37` with 5 containers
   and usually one potion charge.

The 18909f wooden-sword oracle is historical; the gathered Clean L1
Triforce (`clean_poweron64`) replaces it as the M5 evidence.

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --no-infinite-life --no-video --trials 1   # gather → L1 TF
uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --no-video --trials 1
```

That command forces the health assist off. `--rollout` stays opt-in.

## Blue Ring main spine: credits again (rr-c5az, 2026-09-23)

`blue_ring_full_poweron9` went power-on → credits in one session: zero state
loads, 260,248 frames, TF `0xFF`, 14 containers, Blue Ring held, mode 19.
Survival, not Clean: the refill absorbed 387.6 hearts (199 hits), and the
disclosed top-ups remain (250R at `ring`, bombs/keys at the declared gates,
L7 Food). Row and history: [RUN_METRICS.md](RUN_METRICS.md).

What got it there (each fixed from the stalled save point, then power-on):
L6 0x19 and 0x29 (latched `LadderEscape`, goal-aware `ladder_release`,
water waypoints moved to land, `reachable_only` clears), OW 0x15 (lattice
start beside a solid pose), L7 0x59 and L8 0x3E (lattice door first), the
post-L7 rupee floor (arrow budget), L8 Gleeok HC walk, L2 Dodongo HC (taken
on evidence, not "near the stand"), L9 0x10/0x05 Wizzrobes on the generic
engine (0x10 was 110 hearts and a 16,000f timeout; now 9.7h / 2,483f mean
over 12 offsets), cellar 0x4F ladder align, L9 0x04 north-aisle leg.

Next, in order, from the ledger of run 9:

1. Heart drains. Patra now stands outside the orbit (0x52 24h -> 6.8h mean,
   0x61 23h -> 2h); run 10 reached the credits again at 356.3h. Its worst
   rooms: L8 Gleeok 0x3C 16.7 (far stands never kill the heads; 22 px stays),
   L8 0x3E 14.0, L9 0x20 13.5, L6 0x28/0x38/0x3A 12.5-13.5 (offset means are
   4.9-8.0; the threat evader fails there), L5 0x05 11.0. Score any combat
   change with `stage_replay.py --idle` offsets from a pin cut at the room;
   `hits by cause` now works under the refill ($04F0 arming).
2. rr-iu0g: the remaining L9 `chase_sword_step` clears onto the engine.
3. rr-qb6w: engine clears now sweep the room's own key/bombs/rupees
   (`_room_item_goal`, world-flag gated): L2-L4 items are all taken on run
   14. Still left: L5 0x26/0x47 keys (not engine clears), L6 0x29/0x2D keys
   (tape-dependent), L7 bombs/rupees, L8 0x4C key, L9 0x61 key. Taking them
   is how the key top-ups retire (rr-doua).
4. Slow visits: L7 0x0D Wallmasters 8,448f, L2 0x6E 4,302f, L5 0x65 3,940f.
5. The ladder heart 0x5F and the raft heart 0x2F are taken after L4
   (`overworld/magical_sword.py`). The 0x21 grave is on the L6 walk.
   `clean_poweron83` ends at 13 containers and sword 3.

## This sitting's leftover

The verified gate is Clean power-on through the Level 6 Triforce.
The next Clean claim is rr-rgum. The 18909f wooden-sword oracle is
historical. PRE_L1.md stays the gathering note. It is not this sitting.

## Open on the bomb errand

`rr-ttyu.3` is still open. One flagged `--rollout` trial, tag `pre_l1_7e_band1` on 2026-09-20, reached cave `0x6F` with bombs 4. That trial is not the default arm and not a STATUS result. The next measurement is flag-off against flag-on damage on `0x7B`, `0x7C`, and the Zora shot `0x55`, with no new stall. Do not promote rollout. Do not retry a fixed y=133 coast row.

A short arrival is not the gate. The walk tries to bank more than 20 on the coast. If `0x6F` is still short, `bomb_topup` keeps hunting and comes back, and only then does the buy run. The gate is `ADDR_BOMBS >= 1`. The 2026-09-21 shortfall change was not a ROM re-run.

## Not this sitting

Level 2's east mouth, the dungeon lanes, and Clean continuous play have notes under `docs/tasks/`. `spine/clean_tip.py` still points at those files. None of them is the next open row. Survival power-on toward credits is not the Clean gate.

## Survival last-heart residual (rr-k3vj, 2026-09-23)

The continuous `lastheart_poweron28` run (`--engage-hearts 1`) cleared the
previous L3 Raft stall and the L4 Triforce, then died in the L5 0x64 blue
Darknut fight after 125,417 frames. It used no state loads. The assist made
12 refills; one death still occurred because blue Darknuts hit for two hearts
and Link entered the room with four. The L5 0x64 fight currently survives
under full refill by absorbing 28 hearts of damage, so a combat change is
needed before a whole last-heart baseline can finish.

The old `LastHeart28_level5_clear_0x77.state` is ringless and cannot serve
as a main-spine predecessor. The short replays `l5h_diag2`, `l5h_evade1`,
`l5h_south1`, `l5h_backstep1`, and `l5h_occupancy1` all died in 0x64.
The saved `LastHeart28_fight64_entry.state` is an isolated combat pin,
not natural-entry proof. The suffix now records its full fight report,
including fatal contact and enemy count. Next: develop a room-specific
0x64 survival policy from that pin, replay the L5 suffix from its predecessor,
then repeat power-on through credits. Keep the assist threshold at one heart.

Session loop: `.grok/skills/zelda-session/SKILL.md`. Claim one spine bead. Overwrite [PRE_L1.md](PRE_L1.md) with the leftover. Glance leave is room, mode, x/y, sword, bombs, rupees, and hearts. A pin is not leave proof.
