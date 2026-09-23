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

## Blue Ring main-spine audit (rr-c5az, 2026-09-23)

Read at every `BlueRingFull2_*` milestone pin. The ring stays accounted for:
Survival writes `$066D` rupees 73→250 at the `ring` stage (disclosed; natural
farm is rr-t49c), the shop contact sets `$0662` 0→1 and takes the 250, and
`ring=1` holds from `exit_ring` through L7. White Sword, blue candle, letter,
and 6 containers all reach the L1 mouth and stay held. Gaps found:

- L4 Gleeok heart container was never taken (HC 9 after L4, 11 at L7). The
  hunt walked fixed mid-room stands; the container is at (208,192), which is
  the item slot `$0083/$0097`. It now walks there: HC 9→10, and the TF comes
  about 800f sooner. Confirmed continuous in `blue_ring_full_poweron3`.
- L7 0x4A Red Candle cellar: the drop stopped at y=181 on the ladder, where
  RIGHT is dead and keese knock Link back up (36k-frame timeout). The floor
  is y=189; the focused replay takes the candle 1→2 in 4522f.
- Letter is carried but unused (potion, rr-sed5). Food at L7 is still the
  disclosed `$065D` fixture.

`blue_ring_full_poweron3` (L4 fix + 0x4A floor): power-on, zero state loads,
zero deaths, L1–L5 clear, then timed out on the L6 0x19 clear (one enemy
left; 9572 patrol + 3755 `ladder_back_off` frames). The L4 time change
reshuffles L5/L6 (chain is frame-perfect). Next: fix 0x19 from
`BlueRingFull3_level6_clear_0x19`, then rerun power-on; the L7 candle
fix is still unproven continuous.

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
