# Plan — Zelda I

## Now: Clean credits as a health economy (2026-09-25)

Goal: power-on to credits, Clean (no RAM write, no state load), zero deaths.
Inventory is already natural (run 53: zero pokes); only the health refill is
left, so every remaining item below is a death point, not a resource.

### What sets the plan (measured on d1c42958)

- `clean_poweron60` (`--clean`): dies in `white` on 0x18 at 40,801f, the same
  tape as CL45. Pond 0x39 (4/4) to 0x0A spends 5.5h and heals 1 (the 0x2C
  container). 7 of the 11 hits are 0x2B blue leevers and 0x1E tektites.
- `lasth_poweron60` (`--engage-hearts 1 --observed-damage-guard --no-pokes`,
  from a worktree snapshot): the death map. Each last-heart refill is one
  Clean death. Gathering: 2 (walk_28 before White Sword; the ring road, then
  0x34 on `rupees_62`). L1, L2 and L4: **0**. L3: enters at 4/8 after a 2.8h
  walk, then refills at 0x69 and the raft/boss rooms. Stopped at L5 0x66 on
  `bombs=0` (a bomb budget miss under a reshuffled tape, not health).
- Survival damage by level (run 53, beam always on): OW 61h, L1 3, L2 7,
  L3 12, L4 11, L5 40, L6 57, L7 19, L8 57, L9 71. L5+ are unmapped for
  last-heart; expect the next death points there.

### Re-think

1. **The unit is the segment between full refills**: pond fairy (0x39,
   0x43), potion drink, Triforce. A segment survives when the hearts it
   starts with plus what it heals inside beat its damage. Per-room combat
   tuning chased damage everywhere; most rooms are already inside budget.
2. **Heal before you fight.** For each death point, first try a heal on the
   route (take-any potion, pond, potion shop), then combat. A heal is one
   stage; a combat fix reshuffles every later room.
3. **Last-heart refills are the death map.** Run it from a worktree snapshot
   so the main tree stays editable (`PYTHONPATH=$W:$W/snes:$W/nes`).
4. **Not Clean:** `--rollout` (it `set_state`s the played emulator), any
   heart write. A game-over Continue writes nothing, but the target stays
   zero deaths.
5. The White Sword beam needs full hearts. Healing to full is worth more
   than the hearts: it turns the beam back on.

### Ladder (each rung is a `--clean` power-on, no resume)

| rung | stop | now |
|---|---|---|
| C1 | White Sword taken | **green** (`clean_poweron64`) |
| C2 | Blue Ring, L1 mouth 0x37 | **green** (64) |
| C3 | L1 Triforce (new M5 on the gathered route) | **green** (64) |
| C4 | L2 Triforce | **green** (64, TF `0x03`) |
| C5 | L3 Triforce | red: dies in L3 at 110,963f |
| C6 | L4 Triforce | – |
| C7 | L5 Triforce (bomb budget first) | – |
| C8-C11 | L6, L7, L8, L9 + credits | – |

A rung is green once one power-on `--clean` run reaches it. Keep
`natural_credits_poweron53` green (`--no-pokes`) after each route change:
**it is red now** (`natural_credits_poweron65`: the reshuffled tape reaches the
L2 Dodongo with 7 bombs and spends all of them, rr-pm7m).

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

### Next, in order

1. **L3 boss suffix (C5, rr-tff2).** From `L3o<n>_enter_level3` (8 offsets,
   `scratch/offset_pins.py` + a resume per offset) every run now reaches
   the boss path and fails there: low hearts after the raft (0.5-2.5 left)
   or `bombs=0` (0x5D's prep clear bombs Zols/Gels on a timer). Keep bombs
   for Manhandla; heal before L3 (0x39 pond from 0x59 on the entry walk).
2. **Dodongo robustness (rr-pm7m).** Survival regression red: 7 bombs spent at
   0x0E without a kill (pin `NC65_fight_dodongo`). The same fight passes in
   the Clean tapes; it is RNG-fragile.
3. **L3 small bleeds:** 0x5B north-chain Darknuts (1-4h, not flanked yet, rr-j47p),
   0x5A blade trap on the key-door push (0.75h every run), the 0x0F raft
   passage keese.
4. Then extend the last-heart death map past L5 (bomb budget first).

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
5. Overworld hearts still skipped: ladder heart 0x5F, raft heart 0x2F
   (rr-ps7.4.*), for 16 containers.

## This sitting's leftover

See PRE_L1.md, "Leftover (2026-09-22, second sitting)". The default spine is green to the L1 Triforce with the pond detour. Rung 2 (L1 health off) is red at L1 0x23, and rung 3 is red at `heart_7b`. The next measure is the 0x23 chase, without moving M5's 18909f.

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
