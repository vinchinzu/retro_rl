# Zelda I sitting residual — Level 9 worktree integration (2026-09-27)

Lane A from Claude's interrupted session is complete and on main at
`a686cce2`, based on `ff66c93a`.
Bead `rr-npv.5` stays in progress: Clean power-on credits is still unproved.
The user authorized completing the five other worktrees with new sub-agents
and landing every lane on main. The stall lane is now complete; hazards and
the Red Ring remain active. The bomb-budget lane is next, followed by the
overworld lane as a sub-agent slot opens.
Claude's partial edits were preserved. No STATUS change or push.

The current-main real-C10 resume `codex_all_baseline` fails
`bomb_restock_l8` at 3008 frames with 4 bombs and 2R. It has one state load,
no heart assist, no inventory pokes, and confirms the shop shortfall.
The Red Ring's reciprocal ROM-door graph requires three extra bomb walls;
the bomb-budget lane must guarantee the 0x16 item and save another bomb,
for example by avoiding 0x31W. Existing lane-A verification follows.

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
