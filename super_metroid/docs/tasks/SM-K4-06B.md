# TASK SM-K4-06B: One-primitive short-hop Y approach on `kraid_to_eye_return`

## Recipe step
1 pure controller (geometry — **must green or super-clean residual; no force-pass**)

## Model
Luna

## Own files only
- `routes/kpdr/varia_return.py` (**edit** `play_kraid_to_eye_return` only)
- optional: `docs/tasks/SM-K4-06B-residual.md` (**create** if pure still red)

Do **not** edit continuous.py, STATUS, progression, tracker, other controllers,
or door-recon scripts.

## Context
Wave-3 SM-DOOR-PHASE gold (honest, not pure green):
- Source: `scratch/post_varia_to_kraid_pure.state` → room `0xA59F`
- Phases end pose 82 @ x≈37 y≈307; Y-sweep all end pose 82 @ x≈36 y≈374
- **Never** `door_transition != 0`; never left `0xA59F`
- Recommended **single** next change: fixed short-hop Y-approach **before**
  existing door shots — **do not** also change backoff / reface / shot / spin

## Read first (all)
- `docs/tasks/SM-DOOR-PHASE-report.md` (recipes + pin tables)
- `routes/kpdr/varia_return.py` full `play_kraid_to_eye_return`
- `routes/kpdr/varia_return.py` `play_varia_to_kraid` door pattern (style only)
- `routes/controller_common.py` (`hold`, `unmorph`, `wait_ordinary_room`)
- `docs/tasks/SM-K4-06.md` (acceptance shape)

## Do (aggressive but bounded)
1. Replace **only** the floor-level left approach (the `kraid_return_approach`
   spin-walk to x≤180) with **one** fixed short-hop Y-approach primitive:
   - Still end near left lip (x band similar to today, ~≤180)
   - Insert a single short hop (fixed A-hold + LEFT) so Samus passes a higher
     Y band during approach, then land/settle briefly before the existing
     lip backoff → unmorph → face → 4 door shots → spin-push sequence
   - Keep all post-approach timings **byte-identical** (backoff 10, face 8,
     release 6, 4× shot/fuse, spin-push + lip recovery)
2. Do **not** tune two parameters at once. If pure fails after this one change,
   stop — do not stack backoff or shot experiments.
3. Pure probe (required):
   ```bash
   uv run python super_metroid/scripts/probe/kpdr.py pure kraid-to-eye-return \
     --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
   ```
4. If green: optional save note path for planner; do **not** promote graph.
5. If red: residual must include exact last pin (room, pose, x, y), whether
   any frame saw `door_transition != 0`, and which **one** next primitive the
   planner should authorize (not free spin).

## Residual required (super-clean)
- Diff summary limited to approach primitive
- Pure exit code + last failure state
- Explicit non-claims: not continuous, not STATUS, not graph promotion

## Do not
- Combine multiple geometry knobs
- Touch continuous / STATUS / progression verification
- Forge door / boss / room RAM
- Claim pure green without exit 0

## Acceptance
- [ ] Only approach primitive changed (or residual proves attempt + stop)
- [ ] Pure green **or** honest residual with pin + one next primitive
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py super_metroid/tests/test_progression.py -q` green

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py super_metroid/tests/test_progression.py -q
uv run python super_metroid/scripts/probe/kpdr.py pure kraid-to-eye-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
```
