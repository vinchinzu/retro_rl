# Plan — Zelda I

## Now: what is left between Survival credits and Clean credits (2026-09-24)

Survival credits are routine now: run 31 (275,135f) had no rupee writes,
and last-heart run 37 (293,488f) had zero deaths and zero state loads.
natl8_3 went from power-on to the L8 leave with no bomb or key writes.
Clean is `run_survival_spine.py --clean`, meaning no refill and no pokes. It
has not been run past L1.

These assists are left:

| Assist | Where | Replaced by | Bead |
|---|---|---|---|
| Health refill | whole run. Run 37: 7 target + 16 safety refills | fewer hits, heart pickups, potions | rr-k3vj, rr-thlc |
| Bomb count | L9 chapter gates (`SPINE_L9_RETOPUP`) | a 20R pack on the post-L8 walk | rr-ps7.5 |
| Wooden arrows `$0659` | L6 Gohma `0x1C` | an 80R buy at a `CAVE_SHOP_ARROWS` cave after L1 | rr-ps7.7 |
| Food `$065D` | `level7_bait_purchase` | 60R bait at `0x34` | rr-8t4.4 / .5 |

The health refill is the largest gap, but it is not the only one. The other
three are shop buys, and so is the second potion. Together they cost
20 + 80 + 60 + 40 = 200R. natl8_3's wallet was 17R at L2 entry, 45R at L3,
34R at L4 and 38R at the L8 leave, so rupees are now the binding constraint.
Each buy also moves every later frame, because a timing change in one room
reshuffles every later room. So land the route buys first and tune combat
after, or the tuning has to be redone.

1. **rr-ps7.5, L9 bombs.** L8 leaves with 0 bombs and 38R. Add a want-gated
   `BombRestockController` stop on the `0x6D` walk, then empty
   `SPINE_L9_RETOPUP`. It moves only L9 frames. Watch for the every-frame
   `bombs<=0` guard that fails while the last bomb burns.
2. **rr-ps7.6, rupee budget.** Build a per-stage wallet table from natl8_3,
   then choose income that lands before each buy. Unused sources: caves in
   `SECRET_RUPEE_CAVES` that the walk skips (`0x67` 30R and `0x71` 30R are
   bomb rocks near the start, `0x13` 30R is past the river, `0x51` gives
   10R) plus the Armos caves `0x3D` 30R and `0x4E` 10R, for 140R in all.
   Another 35R is in the room rupee5 items natl8_3 left behind, and 103 of
   179 floor drops went unpicked. The wallet caps at 255.
3. **rr-8t4.5, bait at `0x34`.** `0x34` sells key 80 / Blue Ring 250 /
   bait 60, and the gather walk already goes there for the ring. Food keeps
   until L7-B, so the mountain-locked L6 → `0x34` walk (rr-8t4.4) may not
   be needed. Either stop at `0x34` a second time with 60R, or spend 0x62's
   post-ring 100R on bait instead of the red potion. Then delete the Food
   exception from ASSIST_CONTRACT.
4. **rr-ps7.7, wooden arrows.** The buy can go anywhere between the L1 bow
   and L6. `0x4A` is the cave where the pre-L2 bomb pack is bought. Set
   Gohma to `poke_arrows=False` on the default spine, and keep the 1R per
   shot arrow reserve.
5. **Inventory-clean gate.** Run `--through level9-credits --no-pokes`
   with the refill still on, one continuous power-on. Once this is green,
   health really is the only assist left.
6. **Re-baseline health.** Run `--engage-hearts 1 --observed-damage-guard
   --no-pokes` from power-on on HEAD. Run 37 was recorded before rr-doua,
   and every room after L1 has moved since. Read the damage census per room
   before tuning anything. Run 37's worst rooms were L5 0x05 (16h), L5 0x64
   (15h), L9 0x10 (14.5h), L6 0x3A (12.5h) and L9 0x20 (12.5h). Score
   combat changes on `stage_replay.py --idle` offsets, not on one tape.
7. **Health levers, cheapest first.**
   - Pick up heart and fairy drops when below full. natl8_3 missed 53.
   - Add a second potion stop once rupees exist (rr-thlc). A drink trigger
     earlier than the last heart stalled L3 0x69, so keep that trigger.
   - Room combat on whatever the re-baseline ranks worst.
   Under Clean the true death count is somewhere between the target
   refills (7) and target + safety (23). Only a `--clean` run gives the
   exact number.
8. **Clean attempts.** Run `--clean --through level9-credits` in one
   session. Each death becomes a room card, fixed from its `Full_<stage>`
   save point and then re-run from power-on. Promote in STATUS only when
   deaths = 0, `set_state` = 0 and every write count is 0 (rr-npv).

Bead housekeeping. rr-k3vj's baseline is met by run 37, so point it at the
re-baseline in step 6. rr-s9ep (L3 0x5C) did not come back in run 37;
close it if the re-baseline passes L3. rr-exhy (the legacy `--no-gather`
prefix) is not on the main path.

## The Gathering (route order)

Zelda Dungeon calls this The Gathering. Order:

1. Wooden sword on `0x77`.
2. South-coast walk to bombs at `0x6F`. Stop when `ADDR_BOMBS >= 1`.
3. Heart at `0x7B` (taken from a `BFS_7C` pin, `GatherHeartL8Leave`), then the heart at `0x2C` (taken from a `BFS_2C` pin, `GatherHeartM3Leave`).
4. Northeast cluster: 100 rupees `0x0F`, letter `0x0E`, candle `0x0C`, White Sword `0x0A`. Go around Lost Hills `0x1B`. The older 21609-frame chain ended ringless at L1 and is historical evidence only. Next: heal before `exit_6f` so the chain can run with no refill. The `backtrack44` key poke is gone: L1 takes 0x72's key (rr-doua).
5. Burn heart `0x47` and the 90-rupee shield at `0x46`.
6. Blue Ring at `0x34` is mandatory before L1, paid by the hidden rupee caves; 0x62's 100R after it buys a red potion at `0x64`. Arrows are rr-ps7.7 and bait is rr-8t4.5; both need the rupee budget (rr-ps7.6).
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
