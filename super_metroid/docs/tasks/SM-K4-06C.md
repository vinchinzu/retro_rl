# TASK SM-K4-06C: One-primitive short-hop **landing/settle** timing on kraid_to_eye_return

## Recipe step
1 pure controller (geometry — green or super-clean residual)

## Model
Luna

## Own files only
- `routes/kpdr/varia_return.py` (**edit** only the short-hop / approach-settle frames after 06B)
- optional: `docs/tasks/SM-K4-06C-residual.md`

Do **not** edit continuous, STATUS, progression, backoff/shot/spin blocks.

## Context
SM-K4-06B replaced floor approach with: stage LEFT walk → **18f** `LEFT+A` hop →
**12f** neutral settle → existing door shots. Pure still RED:
- pin pose `82` x=`37` y=`307`, `door_transition=0`, room `0xA59F`
- Authorized next: vary **only** hop duration and/or approach_settle frames

Source (required):
`custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state`

## Read first
- `docs/tasks/SM-K4-06B-residual.md`
- `docs/tasks/SM-DOOR-PHASE-report.md`
- `routes/kpdr/varia_return.py` `play_kraid_to_eye_return` (approach block only)
- DOOR-BLUE report (no open-state field yet)

## Do
1. Change **only** `kraid_return_short_hop` hold length and/or
   `kraid_return_approach_settle` hold length. Try at most **two** bounded
   pairs (e.g. hop 12/18/24 × settle 6/12/20) in separate pure runs — pick the
   best residual if all red; do **not** retune lip backoff / shots / spin.
2. Pure after each pair; stop early if green.
3. If red after ≤2 pairs: residual with pin table per attempt + one next
   primitive for planner (e.g. shot Y band / PLM field) — **not** free spin.

## Residual required
- Exact hop/settle values tried
- Best pin + door_transition max
- Non-claims

## Do not
- Touch post-approach door choreography timings
- continuous / STATUS / graph promote
- Forge door RAM

## Acceptance
- [ ] Only hop/settle approach knobs changed
- [ ] Pure green **or** residual with attempt table
- [ ] pytest controller_common + progression green

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py super_metroid/tests/test_progression.py -q
uv run python super_metroid/scripts/probe/kpdr.py pure kraid-to-eye-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
```
