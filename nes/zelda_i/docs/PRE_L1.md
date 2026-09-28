# Zelda I sitting residual — Level 9 worktree integration (2026-09-27)

Lane A from Claude's interrupted session is complete and on main at
`a686cce2`, based on `ff66c93a`.
Bead `rr-npv.5` stays in progress: Clean power-on credits is still unproved.
The user authorized completing the five other worktrees with new sub-agents
and landing every lane on main. Lane C is now on main at `6c5eced9`.
The user then requested a graceful stop and handoff. The hazards and Red Ring
agents hit a usage limit before completing validation; their changes remain
uncommitted in their original worktrees. Bomb-budget and overworld agents
were not launched. Claude's partial edits were preserved. No STATUS change
or push. No takeover eval processes remain running at handoff.

The current-main real-C10 resume `codex_all_baseline` fails
`bomb_restock_l8` at 3008 frames with 4 bombs and 2R. It has one state load,
no heart assist, no inventory pokes, and confirms the shop shortfall.
The Red Ring's reciprocal ROM-door graph requires three extra bomb walls;
the bomb-budget lane must guarantee the 0x16 item and save another bomb,
for example by avoiding 0x31W. Existing lane-A verification follows.

## Resume the remaining lanes

All paths below are relative to `.claude/worktrees/`. The unfinished trees
are based on lane A (`a686cce2`), so bring them forward to current main while
preserving their dirty changes before continuing. Do not land either WIP
without fixing the failures and repeating its ROM matrix.

| Lane | Worktree | Checkpoint / next action |
|---|---|---|
| B: hazards | `agent-a936abbf5e5d1f561` | Dirty `dungeon/shot_guard.py` (staged Claude trap draft plus new unstaged changes), new `tests/test_shot_guard_hazards.py`. Fix 0x04 trap deadlock and 0x20 bomb-wall recovery before landing. |
| C: stalls | `agent-ac061e44b550e7c0a` | Clean agent commit `b97d30c7`; integrated on main as `6c5eced9`, including bead export and this living handoff. |
| D: bombs | `agent-a5e5e888637a7a237` | Original dirty one-line `natural_path.py` Patra16 substitution only; new agent never launched. Start this lane next. |
| E: overworld | `agent-ad65a8b9932b58953` | Clean lane-A base; new agent never launched. Reduce Death Mountain approach damage after hazard integration. |
| F: Red Ring | `agent-a6cbaad37898ea620` | Dirty route/controller/probe/tests, no commit. Detour passes 4/4 loaded offsets, whole L9 fails 0/4; finish D's guaranteed bomb budget before landing F. |

B's latest matrix (`logs/codex_lane_b/after15_v2.txt`) clears the Patra join
3/4 at 10927–14043f and 5.27–6.02h on the successful offsets. Offset 3
stalls at 24000f in 0x04, `(184,108)`. The 0x20 north bomb also clears 3/4,
mean 0.88h, but offset 3 ends `push_timeout` at `(80,129)` after a guard dodge.
The existing `BombWallController._push_dir` blind alignment walks into a
pillar; replace recovery with the existing lattice path to the mouth.
That scoped shared-helper fix was authorized but not implemented before
the usage limit. Verify another bomb-wall consumer as well. The last B unit
gate passed 2129 tests, 4 skipped, 41 deselected; it does not establish the
subsequent runtime changes or route regressions as safe.

F follows `0x16→0x26→0x27→0x17→0x07→cellar 0x00`, then returns to 0x16.
The three forward bomb walls consume exactly three bombs; reverse travel
uses the holes already opened. Its loaded, no-refill 10h measurements at
offsets 0/3/7/11 all naturally acquire ring 2 and return, taking
5582/5413/5430/5615f and 4.88/8.38/8.13/6.26h. These are heart writes at
load, not Clean proof. The tapes retain the live 0x16 Patra, so D's prior
kill will change timing. F's whole assisted L9 matrix fails all four offsets
(0x51 stall, final wall bomb failures, and the pre-C 0x10 stall).
Its full unit gate has one missing gitignored input failure:
`recordings/l9_room51_dump.json`; link the existing main input before rerun.
Logs and screenshots are under `logs/codex_lane_f/` in F's worktree.

The required chapter order is `East15 → Patra16 kill/item → RedRing →
North16`. D must retain skill ID `level9_patra_16` for stable probe alias
`s14b`; F uses `level9_red_ring` / `s14r`; the existing north walk remains
`s15`. The old Patra16 factory kills **and exits north**, so it cannot simply
be placed before the detour. Implement a kill-only/item controller and
move it out of `prefix.py` into an existing owning module without a circular
import. Keep `prefix.py` at or below its 1603-line base.

Natural C10 carries 4 bombs / 2R. Rock and 0x65 north cost two before 0x16;
the guaranteed Patra item adds four, and the ring detour spends three.
The remaining old route costs four, leaving a one-bomb deficit. Avoid 0x31W
via cleared 0x51's west shutter, then 0x50→0x40→0x30, or prove another
guaranteed natural source. Do not rely on random bomb drops. Make the L9
post-L8 shops skip an unaffordable purchase with an L9-only opt-in, preserving
other levels' fail-closed behavior. Verify the real C10 predecessor again.

Original lane briefs are preserved in main's
`nes/zelda_i/logs/codex_takeover/agent-*_brief.md`. Exact dirty tracked patches
and new tests are also snapshotted in `paused_lane_b`, `paused_lane_d`, and
`paused_lane_f` there. Artifacts are gitignored; the worktrees remain intact.
Use `PYTHONPATH="$PWD:$PWD/snes:$PWD/nes"`,
`UV_PROJECT_ENVIRONMENT=/home/v/01_projects/11_games/retro_rl/.venv`, and
`QT_QPA_PLATFORM=offscreen uv run --no-sync` from each worktree. Limit each
matrix to three jobs and never edit its runtime while its ROM eval runs.
Final main gate: `QT_QPA_PLATFORM=offscreen uv run --no-sync python -m pytest
nes/zelda_i/tests tests/test_docs.py -q` passes **2136 tests**, 41 ROM tests
deselected. Main's integrated guarded ROM checks reproduce s24 offset 0 at
3582f / 1.75h and s21 offset 0 at 7264f / 0h. Logs are
`logs/codex_takeover/main_handoff_checks.txt` and `main_lane_c_s{24,21}.txt`.
Next action: start a fresh D agent with the bomb budget and kill-only order
above, then resume B/F and launch E as slots open; integrate validated commits
on main with bead exports, finally run composed ROM and main test gates.

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

The full guard budget still exceeds the hearts available on the real Clean
predecessor. No continuous power-on run was performed. Trap/orange-shot,
0x10/0x61 stall, bomb-budget, overworld, and Red Ring work remain in their
separate lanes; none were incorporated here.

## Plan

Review/cherry-pick this lane into the integrator tree, then finish the next
selected lane. Keep `rr-npv.5` open and remeasure the combined route before
attempting the continuous Clean credits gate. Do not claim Clean from these
pins or promote STATUS from this replay.
