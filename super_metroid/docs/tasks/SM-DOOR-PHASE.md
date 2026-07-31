# TASK SM-DOOR-PHASE: Phase-instrumented Kraid left-door probe + Y sweep (diagnostic)

## Recipe step
diagnostics (supports pure geometry for kraid_to_eye_return — **no pure-green claim**)

## Model
Luna

## Own files only
- `scripts/probe/kraid_door_phase_recon.py` (**create**)
- `docs/tasks/SM-DOOR-PHASE-report.md` (**create**)
- optional gitignored: `debug/kraid_door_phase_recon.json`

Do **not** edit `varia_return.py`, continuous, STATUS, progression, or other controllers.
This card is diagnostic only; a later geometry card may patch the controller.

## Context
Wave-2 SM-DOOR-RECON gold (from `post_varia_to_kraid_pure.state`):
- Start: x=463 y=395 pose 10, room `0xA59F`
- Pin: x=85 y=427 pose 138 by ~f170 — stuck through f600
- Never left `0xA59F`; `door_transition` always 0; enemy0 HP 1000; boss bits stable
- Hypotheses: wrong door height, pose-138 pin, closed blue door, enemy collision

Planner instruction: instrument phases separately + Y sweep — **not** more free spin.

Controller phases to mirror (read-only from `varia_return.play_kraid_to_eye_return`):
1. approach LEFT+B+A until x≤180
2. lip backoff RIGHT
3. unmorph
4. face LEFT + release
5. 4× standing door shots (LEFT+X / fuse)
6. spin-push LEFT+B+A with lip recovery

Source (required):
`custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state`

## Read first (all of these — multi-tool)
- `scripts/probe/kraid_left_door_recon.py` (session/sample pattern)
- `docs/tasks/SM-DOOR-RECON-report.md` (baseline trajectory)
- `routes/kpdr/varia_return.py` (`play_kraid_to_eye_return` full body)
- `routes/controller_common.py` (`unmorph`, `hold`)
- `scripts/probe/kpdr.py` (boot_from_state / pure session if useful)
- `ram.py` SuperMetroidState fields used in samples
- `docs/tasks/SM-K4-06.md` (geometry card acceptance shape — do not implement)

## Do (thorough — full list)
1. Create `scripts/probe/kraid_door_phase_recon.py` CLI with:
   - Boots the named source state; resource assist only (no progression/capacity/boss writes)
   - **Mode A `--mode phases` (default):** run the six controller phases above as separate
     labeled segments; sample **every frame** (or every 1–2f) inside each phase with fields:
     `frame, phase, room, pose, x, y, game_state, door_transition, transition_direction,
      enemy0_hp, boss_bits, selected_item, velocity_y` (if available)
   - Cap total budget ≤ ~2000 frames; exit cleanly with summary JSON
   - **Mode B `--mode y-sweep`:** from a fresh boot of the same source, for each attempt
     `i` in a small set (e.g. floor walk, short hop, medium hop — normal inputs only):
     approach left lip, attempt door shots + brief spin, record best x/y/pose and whether
     room/door_transition ever changes; **no teleports** for green claims (if a diagnostic
     placement is used, label it `dev_only` and never claim pure green)
   - Write JSON under `debug/kraid_door_phase_recon.json` (+ optional per-mode path)
   - Print a one-screen human summary (start/end per phase; any door_transition≠0; final pin)
2. Run both modes once from monorepo root (or package cwd with correct paths). Capture exit 0.
3. Write `docs/tasks/SM-DOOR-PHASE-report.md` with:
   - Phase table: frames spent, x/y/pose range, door_transition max, outcome
   - Y-sweep table: attempt → peak left x, final pose, room change? (Y/N)
   - Ranked hypotheses updated from Wave-2 recon (what this run supports / rejects)
   - **Recommended geometry patch recipes** for a future SM-K4-06b card
     (one change at a time: backoff duration, re-face, shot timing, jump approach Y)
   - Explicit: not pure green; not continuous; no STATUS promotion

## Residual required (super-clean — final message must include)
- Exact last pin (room, pose, x, y) per mode
- Whether any sample saw `door_transition != 0` or room ≠ `0xA59F`
- Which single controller change a planner should try next (one primitive only)
- Files changed + verify command paste

## Do not
- Claim pure green or edit the controller
- Forge door/boss RAM
- Free-explore multi-minute runs
- Touch continuous / STATUS / progression

## Acceptance
- [ ] Script exits 0 for `--mode phases` and `--mode y-sweep`
- [ ] Report with phase + Y-sweep tables and ranked hypotheses
- [ ] Diff limited to script + report (+ gitignored debug JSON ok)

## Verify commands
```bash
uv run python super_metroid/scripts/probe/kraid_door_phase_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state \
  --mode phases --frames 2000
uv run python super_metroid/scripts/probe/kraid_door_phase_recon.py \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state \
  --mode y-sweep --frames 2000
test -f super_metroid/docs/tasks/SM-DOOR-PHASE-report.md
```
