# TASK SM-K4-R-03E: Zeela→Warehouse — anti floor-left during reverse-shot climb

## Recipe step
1 pure controller

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_zeela_to_warehouse_return` second-drop
  reverse-shot cadence only)
- optional residual: `docs/tasks/SM-K4-R-03E-residual.md`

## Context
- R-03D **RED**: forward-drop reverse-shot class applied; probe entered floor
  door lane: `room=0xA471 pose=82 x=20 y=396 door_transition=1`
  (same wrong-door family as R-03 / recon morph-left → `0xA4B1`).
- Recon best was `min_y=331` **in-room** without that door; R-03D over-lefted.
- **One change:** constrain leftward motion during second-drop reverse-shot so
  `x` cannot enter the floor-door band (e.g. x floor ~≤40 / y>250) before
  mid-band climb (`y<=325`). Prefer hard x floor (recenter RIGHT) over removing
  the reverse-shot class.
- Keep floor-door fail-loud guard. No `0xA4B1` success. No kihunter retouch.

## Do
1. One constraint on leftward portion of second-drop reverse-shot only.
2. Pure probe same source; residual if red with pin + next card.
3. If still same pin after one honest try, residual next = PLANNER redesign
   (do not invent R-03F spam).

## Acceptance
- [ ] Pure green → `0xA6A1` **or** residual with pin + next card
- [ ] No silent `0xA4B1` / floor-door success

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure zeela-to-warehouse-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_zeela_to_warehouse_return.state
```
