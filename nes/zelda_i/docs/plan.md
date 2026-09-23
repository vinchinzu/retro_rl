# Plan — Zelda I

## Now

Gather the Blue Ring before the next Level 1 measure. The clean tip is still `l1_tf`.
The default main spine buys it at `0x34` after the `0x47` heart, then returns
via the `0x39` pond to the L1 mouth. Survival tops the wallet to 250R at the
ring stage and the shop itself changes `$0662` from 0 to 1. Ringless L1 and
later checkpoints, including the old L5 pins, are obsolete for this main
spine. Rebuild them from power-on; `--resume` rejects a ringless downstream
state. A natural 250R farm remains future work.
The first open ladder row is `pre_l1` in `spine/clean_tip.py`: hypothesis,
inventory gap. The spec is [PRE_L1.md](PRE_L1.md).

Zelda Dungeon calls this The Gathering. Order:

1. Wooden sword on `0x77`.
2. South-coast walk to bombs at `0x6F`. Stop when `ADDR_BOMBS >= 1`.
3. Heart at `0x7B` (taken from a `BFS_7C` pin, `GatherHeartL8Leave`), then the heart at `0x2C` (taken from a `BFS_2C` pin, `GatherHeartM3Leave`).
4. Northeast cluster: 100 rupees `0x0F`, letter `0x0E`, candle `0x0C`, White Sword `0x0A`. Go around Lost Hills `0x1B`. The older 21609-frame chain ended ringless at L1 and is historical evidence only. Next: heal before `exit_6f` so the chain can run with no refill, then drop the key poke at `backtrack44`.
5. Burn heart `0x47` and the 90-rupee shield at `0x46`.
6. Blue Ring at `0x34` is mandatory before L1. Arrows at `0x4A` and potion at `0x64` remain separate route work.
7. Then the Level 1 mouth at `0x37`, and only then a new Clean Level 1 measure.

Do not overwrite the 18909f oracle while this prefix is open.

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

1. Heart drains (hearts lost under the refill): L9 Patra 0x52 24.0, L9
   0x61 23.0, L8 Gleeok 0x3C 16.7, L8 0x3E 14.0, L6 0x28 13.5, L6 0x38 12.5,
   L6 0x3A 12.5, L5 0x05 11.0, L9 0x42 10.0, L6 0x09 10.0. Score any combat
   change with `stage_replay.py --idle` offsets from a pin cut at the room.
2. rr-iu0g: the remaining L9 `chase_sword_step` clears onto the engine.
3. rr-qb6w: the dungeon keys the ledger reports untaken (L2 0x3E, L5 0x26,
   0x47, L6 0x2D, 0x58, L8 0x4C, L9 0x61); taking them is how the key
   top-ups retire (rr-doua).
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
