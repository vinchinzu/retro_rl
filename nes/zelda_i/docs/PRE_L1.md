# Zelda I sitting residual — Level 9 lane integration (2026-09-28)

Lanes A (`a686cce2`) and C (`6c5eced9`) were on main at the handoff. On
2026-09-28 a single Claude thread landed D, B, E and F in the main tree (one
commit each), then made Level 9 survivable Clean from real entries (below).
Bead `rr-npv.5` stays in progress until a continuous power-on tape reaches
the credits. No STATUS change or push.

## Power-on tape reaches the credits; every audited load is a rollout

`run_survival_spine.py --clean --through level9-credits --save-points C11Q
--no-video --trials 1 --tag clean_poweron_c11p2` (and the identical
`clean_poweron_c11p1`): all 330 stages succeed from power-on in one
continuous emulator session, `resumed_from` null, 334,763 frames, no RAM
writes, no assist, ending on `level9_wait_credits` with the "hero of Hyrule"
text on screen (`recordings/clean_poweron_c11p2_final.png`, TF 0xFF, room
0x32). The run still reports `ok=False failed=mid_run_state_load`: the audit
counts `set_state=21164`, and all 21,164 are rollout lookahead restores
(`rollout_restores=21164`, new `zelda_i.rollout.restores()`), from
`PatraBlade`, `Rollout.walk` (0x04) and `PolicyGuard`. Each restore returns
the core to the live frame it saved on that same frame, so the tape is one
continuous play; but the lookahead does see the future, and the current
contract ("no state loads after power-on") does not exempt it.

Whether rollout lookahead is Clean is a policy call for the owner. If yes,
the gate should count loads that are not rollout restores (0 here) and
disclose `rollout_restores`; if no, the four rollout users must be replaced
by model-only policies before `rr-npv.5` can close. STATUS is unchanged.

## Clean Level 9 from real entries: 8 of 8 offsets reach the credits

Each run below starts from a real Clean Spectacle Rock state (`E2C11Evalo<n>
_s07`: the C10 predecessor, offset n, carried through the new entry chapter)
and plays to `level9_wait_credits` with no assist: `l9_probe.py
E2C11Evalo<n>_s07_level9_spectacle_rock_bomb --from s07`. Step pins are
`W1o<n>_*` (first pass) and `W2o<n>_*` (second).

First pass (lanes A-F): 4/8 credits. Deaths in the join (0x30), at Ganon (x2),
and 0x05's stairs hop failed on a knock back into 0x06. Per-step A/B from the
`W1` pins, `PolicyGuard` on vs off (8 offsets; total hearts lost):

| Step | Off | On |
|---|---|---|
| s11 0x55 stairs | 8/8, 13.0h | 8/8, 0.0h |
| s24 room 0x10 | 7/7, 23.8h | 7/7, 20.0h (two offsets worse) |
| s25 Patra join | 6/7, 36.8h | 6/7, 5.0h (o6 0x51 stall, fixed below) |
| s29 Ganon | 4/6, 14.9h | 6/6, 0.0h |

Second pass, with those four wrapped, `stairs_05` walking back from 0x06
(`reentry`, at most twice) and 0x51's walk lattice-first: 5/8 credits with
9.2-13.0 hearts left. The three failures were one bug: the lattice walk had
dropped the hand thread's A cadence, and a $17 Like Like swallowed Link at
(48,181) until the 24000f cap. With the cadence back, reruns from `W2o3` and
`W2o6` reach the credits with 12.0 and 10.2 hearts. `W2o1` then stalled in
0x04: all three one-shot crossing plans came back empty while a trap was still
charging, and the join fell to the trapped row-93 walk. 0x04 now also waits
before the bait and holds to replan every 30 frames (up to ten); the rerun
reaches the credits with 10.5 hearts. Every offset 0-7 has reached the
credits from its real Clean L9 entry (hearts left 9.2-13.0). Room 0x10 is now the
largest cost (0.25-8.25h; Wizzrobe magic $58/$59 and Bubble contacts).

## Red Ring detour on main, opt-in (F)

Lane F's `Level9RedRingController` (`level9/stairs.py`, probe alias `s14r`)
is on main but off the route: `NaturalSilverArrowsController(red_ring=True)`
or `l9_probe.py --red-ring` inserts it after the 0x16 kill, and
`SELECTED_NATURAL_ROUTE.red_ring_included` stays False. It follows
0x16->0x26->0x27->0x17->0x07->cellar 0x00 and back through three bomb walls
it opens. F measured the detour 4/4 at 10 hearts (4.9-8.4h, 5400-5600
frames) with 0x16's Patra still alive; that Patra now dies first. Wiring it
needs a ninth bomb: the join still spends one on 0x31 west (reroute via
0x51's west shutter -> 0x50 -> 0x40 -> 0x30), and the Clean runs below
reach the credits without the ring, so it waits on their failures.

## Overworld lane verified (E): Link reaches Level 9 with 12-15 hearts

The real C10 Clean resume (`CL9A`, copied from `CodexAllBase`) died in L9
0x55 after overworld 0x59 took 9 hearts: the hand walk overshot down 0x59's
x 112-128 corridor and held LEFT against its wall at (112,165..205) for 6500
frames under Zora fire. 0x59, 0x27, 0x07 and 0x06 now take the ROM lattice
to their exit band first (`ow_edge_band_step`), and the Spectacle Rock
controller's walk is four `room_step` legs instead of one-axis presses.

The rock bomb and the post-L8 walk to 0x05 run behind `rollout.PolicyGuard`:
each play frame a deep copy of the controller (0.25 ms) plays its own next
24 frames on the ROM; a hit there buys the first clean held-direction detour
(4/8/16 frames, never one that scrolls the screen). Detour frames do not step
the inner controller.

Clean, `l9_probe.py C11Evalo<n>_level9_post_l8_overworld --from s00 --to s07`
(2R, 4 bombs; offsets 5-7 bank the 0x5D drop), hearts at L9 entry:

| Offset | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Before this session | shop fail | shop fail | shop fail | shop fail | shop fail | 8.48 | 8.24 | 6.98 |
| Now | 12.0 | 15.0 | 12.5 | 13.75 | 12.25 | 15.0 | 14.75 | 14.5 |

The walk to the rock is 0 damage on 7 of 8 (1h on o2); the rock itself costs
0-3h in 800-2000 frames (was 3-4h in ~700, with two deaths in 16 pin runs).
Logs: `logs/lane_e/`. The pointless 0x4A detour on a 2R wallet is still
walked (skipped purchase); a `GatedLeg` to the direct hops would save ~1000
frames.

## Hazard lane verified (B) and every Patra on rollout-checked swings

Lane B's blade-trap model (`ShotGuard` steps each $49 trap as Z_01.asm
`UpdateTrap_Full` does, with a 64-frame horizon while traps are in the room)
landed with its two failures fixed. 0x04: the centre blocks leave only rows
85/93 and 181/189 as east-west crossings, both inside the corner traps'
14 px sensing band, so the old north-aisle walk paid 2 hearts and the guard
stood still at (168,107) for 15000 frames rather than walk it.
`room04_west_plan` (`level9/stairs.py`) now rolls the walk on the ROM
(`Rollout.walk`): step into the band at (144,93) and back to arm both top
traps, wait until they head home (>= 100 frames), then cross to the stand;
the first clean candidate is committed as `ROM_CHECKED` frames. 0x20:
`BombWallController._push_dir` takes `lattice_door_step` to the opened hole
once Link is more than 12 px across the face from the stand (a dodge left
him at (80,129)); nearer pushes are unchanged for all 28 call sites.

`PatraBlade` now also fights the final Patra (0x52, below full hearts; box
x <= 192 off the stairs) and 0x61's; `PatraMelee` is deleted. Plans that
leave the room or play mode are rejected.

| Step | Hearts | Before (main) | After |
|---|---:|---|---|
| s23 bomb north 0x20 | 15 | 3/4, 0.88h (o3 push_timeout) | 4/4, 1.12h |
| s25 Patra join | 15 | 3/4, 6.71h (o7 0x04 stall) | 4/4, 6.83h |
| s21 0x61 Patra | 10 | 4/4, 0h, 4335-7264f | 4/4, 0h, 882-1140f |
| s27 final Patra | 10 | 4/4, 3.75h, 1249-4701f | 4/4 (+o1/o5 6/6), 0h, ~375f |
| s10 / s14 / s17 | 10 | 0.5 / 0 / 0.5h | 0.5 / 0 / 0.5h |

Offsets 0/3/7/11 from the `L9S5_*` pins with hearts written at load; logs
`logs/lane_b/`, `logs/lane_g/`. The join is now L9's largest cost (0x2B,
0x17, 0x1B and Wizzrobe contacts across 0x20/0x41/0x31/0x30/0x04).

## Bomb budget lane verified (D)

0x16's Patra ($48, eyes $26) now dies for its BOMBS item (+4) in hop
`level9_patra_16` (probe alias `s14b`), after East15 and before North16.
Only the sword hurts a Patra: body and eyes carry ObjInvincibilityMask $FE
(`$04B2+slot`), so the Magical Rod's shot (magic, $10) is parried -- twelve
measured shots died on full-HP eyes. The fixed melee stand paid 5/3/10/10h
with two deaths at 10 hearts (offsets 0/3/7/11). `PatraBlade`
(`level9/patra.py`) instead searches short plans (step, turn, A, hold the
13-frame pin and 16 more) with a savestate rollout and commits the first whose
swing drops Patra HP with Link untouched; the emulator is deterministic, so
the committed plan plays out exactly. From `L9S5_s15` at 10 hearts, offsets
0/1/3/5/7/11 all kill it and take the item in 595-707 frames with **0
damage**, then North16 leaves in ~210. Its frames carry `ROM_CHECKED`, which
`ShotGuard.filter` passes through. The controller moved out of `prefix.py`
(1591 -> 1512 lines); the unused `FinalPatraFightController` was deleted.

`LEVEL9_BOMBS_WANTED` stays 8 as headroom: both 0x4A restocks opt into
`skip_unaffordable` (L9 only; other legs still fail closed on a short
wallet). On `C11Evalo0..4` Clean, which used to fail `bomb_restock_l8` with
4 bombs / 2R, both restocks now skip in one frame and all five reach 0x05.
Offset 0 then lost 8.78h on overworld 0x59 (26 Zora `0x55` hits) walking
back from the pointless 0x4A detour: skipping the detour itself when the
wallet is short (a `GatedLeg` pair with the direct `POST_L8_TO_LEVEL9_HOPS`)
belongs to the overworld lane E. Whole L9 from `L9S5_s08` under refill with
bombs written to 3 (the real count after the rock) reaches the credits; bomb
ledger 3 -> 2 (0x65N) -> 6 (0x16 item) -> 5 (0x06W) -> 4 (0x20N) -> join
ends with 6 (drops) -> credits 6. That run's damage is 17.5h (join 6.27h,
room 0x10 3.75h, final Patra 2h, Ganon 2h): refill evidence, not Clean.
Logs: `logs/lane_d/`.

## All six lanes landed; worktrees removed

| Lane | Status on main |
|---|---|
| A: shot guard | `a686cce2` |
| B: hazards | landed with the 0x04 bait crossing and 0x20 push recovery |
| C: stalls | `6c5eced9` (same tree as the agent's `b97d30c7`) |
| D: bombs | landed; the worktree's one-line Patra16 substitution was superseded |
| E: overworld | landed with `PolicyGuard` |
| F: Red Ring | landed opt-in (`red_ring=True`, `--red-ring`) |

The six `.claude/worktrees/agent-*` trees and their branches were removed on
2026-09-28 after their content was on main. Their final dirty patches and
untracked tests are in `logs/codex_takeover/final_worktree_patches/`, their
gitignored lane logs in `logs/codex_takeover/worktree_logs/`, and their 21
unique save states (`CodexLA_*`, `LF*`, `CodexB_room04`) were copied into
`custom_integrations/`. The original briefs stay in `logs/codex_takeover/`.

## Room-stall lane verified

Lane C fixes room 0x10's missing west-aisle combat node and exact stair
alignment. It replaces room 0x61's fixed melee stand with a reachable lane
that follows Patra's roaming body. Its 12 guarded ROM pin cases pass across
idle offsets 0/3/7/11: room 0x10 clears at both 15h and 10h, and room 0x61
clears in 4335–7264f with zero damage. These are Survival-origin loaded pins
with disclosed heart writes at load; they do not prove natural-entry Clean.
The full worktree Zelda gate passes 2124 tests, with 4 skips and 41 ROM tests
deselected. Main's focused integration gate passes 48 tests. The old
`patra_melee_action` API and Patra16 controller are preserved. `prefix.py`
shrinks to 1591 lines; no generic engine change or cap increase was needed.
Reproduction and complete before/after tables are in the lane C worktree's
`nes/zelda_i/logs/codex_lane_c/HANDOFF.md` (gitignored).

## Shot-guard lane verified

Production `level9/hops.py` wraps the Silver Arrows chapter (all 17 hops),
the Patra join, final Patra, and Ganon in `GuardedController`, once per stage.
Pause selection, the overworld, and the ending walk retain their controllers.
The wrapper delegates environment binding, budgets, terminal flags, notes,
and reports; each guarded stage adds `controller.shot_guard` telemetry.

Two guard bugs were exposed by tests and fixed: a 1.5 px simulated step could
overshoot the interior boundary from an off-grid pose, and the wrapper could
replace a completed or failed controller's terminal action with a dodge.
The debug flag now uses a normal `os` import. No hazard model was retuned.
`shot_guard.py` remains below 1000 lines; `prefix.py` and `natural_path.py`
were not edited.

The probe and offset evaluator default to the production guard scope.
`--guard` remains accepted; `--no-guard` selects the A/B baseline. The probe
shares one prefix guard across its hop breakdown and reports override deltas
per hop, avoiding duplicate wrapping or cumulative override double-counting.

### ROM A/B: assisted entry to credits

Pins `L9S5_s08_*` are assisted development data, with Blue Ring, 15 containers
and 7 bombs. Both sides restore health; these damage totals are not Clean
results. Offsets are idle frames after loading the same entry pin.

| Offset | Guard off: frames / damage | Guard on: frames / damage | Credits |
|---|---|---|---|
| 0 | 28546 / 38.28h | 31203 / 26.78h | both |
| 3 | 29741 / 45.30h | 29077 / 23.02h | both |
| 7 | 29519 / 36.78h | 33305 / 26.78h | both |
| 11 | 30360 / 45.03h | 27072 / 22.02h | both |
| Mean | 41.35h | 24.65h | 4/4 each |

### ROM A/B: partial-health hop pins

These trials load assisted pins and set filled hearts to 10/15 to disable
the beam, then play without refill. Inventory is preserved. They are
state-loaded, what-if development measurements, not natural-entry claims.

| Hop | Off: completions / mean damage | On: completions / mean damage |
|---|---|---|
| s10, 0x65 north bomb | 4/4 / 1.00h | 4/4 / 0.50h |
| s14, 0x15 east | 4/4 / 2.00h | 4/4 / 0.00h |
| s17, 0x05 stairs | 3/4 / 3.50h | 4/4 / 0.50h |
| s21, 0x61 Patra/stairs | 3/4 / 1.25h | 3/4 / 0.00h |
| s23, 0x20 north bomb | 4/4 / 6.00h | 4/4 / 3.12h |

Means include failed trials; they do not imply a full clear. The remaining
s21 offset-7 timeout is 20000 frames at (193,149), with no damage; successful
partial-health trials take about 15950 frames. Orange magic still hits s23.
These are the existing lane C and B frontiers.

### Actual spine replay

`codex_lane_a_spine` resumes from a worktree-local copy of
`C11Evalo5_level9_natural_silver_arrows`, renamed
`CodexLA_level9_natural_silver_arrows`. Initial RAM is L9 0x76, mode 5,
(120,205), 8.98/15h, sword 3, ring 1, bombs 7, keys 0, rupees 2, TF 0xFF.
This pin's bombs are assisted predecessor inventory, not the real C10's four.
The run uses the disclosed [Survival refill](ASSIST_CONTRACT.md), no inventory
pokes, and the actual spine chapter factories and ledger.

Result: credits in 31402 frames (including 199 boot frames), one state load,
zero deaths, zero progression/capacity writes. Ledger damage is 26.79h.
Guard overrides: Silver Arrows 941, join 183, final Patra 68, Ganon 0.
The final screenshot shows the credits. `screen_glance.grade_report` passes
with mode 19, room 0x32, (136,136), TF 0xFF, keys 1, bombs 7, full 15 hearts.
This is a resumed Survival result. It does not advance the Clean gate.

### Reproduce

From this worktree, use the shared venv without syncing another environment:

```bash
export PYTHONPATH="$PWD:$PWD/snes:$PWD/nes"
export UV_PROJECT_ENVIRONMENT=/home/v/01_projects/11_games/retro_rl/.venv
export QT_QPA_PLATFORM=offscreen
uv run --no-sync python -m pytest nes/zelda_i/tests -q
uv run --no-sync python -m pytest nes/zelda_i/tests/test_shot_guard.py -m rom -q
uv run --no-sync python nes/zelda_i/scratch/eval_l9_hop_offsets.py L9S5 s08 --to s33 --no-guard --hearts 15 --offsets 0 3 7 11 --jobs 4 --extra --assist
uv run --no-sync python nes/zelda_i/scratch/eval_l9_hop_offsets.py L9S5 s08 --to s33 --guard --hearts 15 --offsets 0 3 7 11 --jobs 4 --extra --assist
uv run --no-sync python nes/zelda_i/scratch/eval_l9_hop_offsets.py L9S5 s10 s14 s17 s21 s23 --no-guard --hearts 10 --offsets 0 3 7 11 --jobs 4
uv run --no-sync python nes/zelda_i/scratch/eval_l9_hop_offsets.py L9S5 s10 s14 s17 s21 s23 --guard --hearts 10 --offsets 0 3 7 11 --jobs 4
uv run --no-sync python nes/zelda_i/scripts/run_survival_spine.py --through level9-credits --save-points CodexLA --resume level9_natural_silver_arrows --no-video --no-pokes --trials 1 --tag codex_lane_a_spine
```

The baseline above was measured before wiring; the equivalent final command
explicitly uses `--no-guard`. Local logs are under `nes/zelda_i/logs/codex_lane_a/`;
the spine JSON and screenshot are under `nes/zelda_i/recordings/`.
The committed ROM test replays the actual Silver Arrows chapter from the
entry pin and asserts earned arrows, preserved inventory capacity, guard/hit
reports, and zero assist progression/capacity writes/deaths.

Validation: Zelda unit/offline gate 2119 passed, 4 skipped, 41 ROM tests
deselected; the new ROM regression passed (68 then-current non-ROM cases
deselected); docs gate 8 passed; `git diff --check` clean. The docs gate first
found eight absent gitignored published recordings; read-only symlinks to
the existing main-tree artifacts restored those inputs before the green run.

## Assumed

The Clean L9 numbers above come from resumed pins (each a real Clean C10
successor state plus one probe load), not a continuous power-on tape.
`rr-npv.5` closes only on `run_survival_spine.py --clean --through
level9-credits` from power-on with 0 loads.

## Plan

Get the owner's ruling on rollout lookahead (above), then either count only
non-rollout loads in the gate and write the STATUS row from
`clean_poweron_c11p2`, or replace the rollout users and rerun power-on. The next Clean
costs to cut are room 0x10 (Wizzrobe magic, up to 8h) and the Spectacle
Rock bomb (0-3h); the Red Ring and the 0x4A-detour skip stay optional.
