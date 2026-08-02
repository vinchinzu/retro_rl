# TASK SM-CLEAN-BT-ECONOMY: Bomb Torizo ammo/health economy (Clean only if RED)

## Recipe step
pure / one-knob stabilize (shared controller care)

## Model
Luna → planner if shared geometry

## Wave type
implement

## Own files only
- Prefer: `combat/bomb_torizo.py` (fight policy only)
- Or: one early-game policy JSON / segment settle if waste is pre-fight
- residual: `docs/tasks/SM-CLEAN-BT-ECONOMY-residual.md`

**Do not open this card until `SM-CLEAN-BOMBS` is RED with a named cause.**

## Context
- Clean BT must finish on natural missile capacity (no refill).
- Any shared-controller change requires **assisted** `--to bombs` re-verify
  before claiming either track green (`CLEAN_TRACK.md` stabilize rule).

## Read first
- `docs/tasks/SM-CLEAN-BOMBS-residual.md` (failure mode)
- `combat/bomb_torizo.py`
- assisted bombs report assist ammo write counts for comparison

## Do
1. Reproduce clean failure; log remaining missiles / HP / death frame.
2. One knob: fewer wasted shots, better activation, shorter fight, or pre-fight
   pack pickup if route-legal.
3. Re-run clean bombs + **assisted** bombs integrity.
4. Residual → re-open `SM-CLEAN-BOMBS` or next one-knob.

## Acceptance
- [ ] One named change only
- [ ] Clean and assisted bombs both integrity-checked after change
- [ ] No progression/capacity writes introduced

## Verify commands
```bash
uv run python super_metroid/scripts/record/continuous.py --to bombs --clean --no-video
uv run python super_metroid/scripts/record/continuous.py --to bombs --no-video \
  --report super_metroid/recordings/start_to_bomb_torizo_assisted_recheck.json
```
