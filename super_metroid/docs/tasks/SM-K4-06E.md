# TASK SM-K4-06E: Jump-enter pure `kraid_to_eye_return` (Y-band residual)

## Recipe step
1 pure controller (one-knob geometry — pure probe **must** green)

## Model
Luna — bounded geometry; planner owns mechanism diagnosis.

## Wave type
implement

## Own files only
- `routes/kpdr/varia_return.py` (`play_kraid_to_eye_return` only)
- `progression.py` (edge verification promote after pure green — planner)
- `tests/test_progression.py` (lock controller_dev, not continuous)

## Context (minimal)
- Wave 6 stabilize closed; continuous kraid/varia integrity green
- 06B/06C/06D + door recon still RED @ door_transition=0, floor pin
  pose 82/138 x≈37 y≈307–427
- Graph: left Kraid exit is gray/`clear_local_lock` (not a free blue walk-in)
- Source: `scratch/post_varia_to_kraid_pure.state` → room `0xA59F`

## Root cause (planner diagnosis)
Standing floor band y≈400–427 never fires the left-door transition even after
beam shots. Dev placement sweep: y∈[140,380] @ x≈80 greens; y≥400 reds.
Prior knobs (hop-then-floor-settle, weapon type, PLM recon) never fixed the
**enter Y band**. One change: open with standing beams (X-only), then
**jump-enter** through the elevated band.

## Do
1. Replace floor spin-only exit with: stage mid-left → backoff → unmorph →
   face left → X-only beam shots → jump/spin enter with re-shot cycles
2. Keep require_state/require_room; no progression writes
3. Pure green from named source; optional `--output` capture eye state
4. Promote graph edge `kraid_to_eye_return` → `controller_dev` only after pure
5. Do **not** continuous-compose or STATUS-promote

## Acceptance
- [x] Pure probe green from `post_varia_to_kraid_pure.state`
- [x] Unit tests green (controller_common + progression)
- [x] Graph edge `controller_dev` (not continuous)
- [x] Eye source captured for SM-SRC-EYE / next hop

## Verify commands
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kraid-to-eye-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
uv run pytest super_metroid/tests/test_controller_common.py super_metroid/tests/test_progression.py -q
```
