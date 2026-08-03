# TASK SM-SPAZER-STAB: Dual continuous integrity for Spazer tip

## Recipe step
stabilize

## Model
Planner (or Luna under planner review)

## Wave type
stabilize

## Own files only
- `recordings/start_to_spazer.json` (+ second run) — gitignored OK
- residual: `docs/tasks/SM-SPAZER-STAB-residual.md`
- do **not** edit STATUS primary tip (STATUS card next)

Depends on: `SM-SPAZER-COMPOSE`.

## Context
- Epic: [`SPAZER_EARLY.md`](SPAZER_EARLY.md)
- Two matching integrity-green continuous runs: power-on → Spazer collected.
- Assist contract unchanged (energy + ammo only).

## Do
1. Record `--to spazer` twice (fresh process).
2. Assert 0 loads / 0 progression / capacity writes / 0 deaths; Spazer beam set.
3. Residual frames + next `SM-SPAZER-STATUS` and optional `SM-SPAZER-POLICY`.

## Do not
- Promote STATUS primary tip away from Frog / current K4 tip
- Fold default spine (FOLD)

## Acceptance
- [ ] Dual integrity green reports
- [ ] Spazer collected on both
- [ ] Residual next IDs named

## Verify commands
```bash
uv run python super_metroid/scripts/record/continuous.py --to spazer --no-video
# second run same; compare integrity blocks
```
