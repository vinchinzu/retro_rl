# TASK T4-ASSIST-HEALS: Drive emergency HP interventions below 65

## Recipe step
policy knob / route polish

## Model
Luna

## Wave type
implement

## Own files only
- `policy.py` — survival spacing / pizza scope knobs (**one** group)
- residual: `docs/tasks/T4-ASSIST-HEALS-residual.md`

Do **not** change emergency threshold/restore contract values without a
separate ASSIST_CONTRACT card.

## Context
- Baseline **65** e-heals (HP ≤ 16 → 80).
- Goal: fewer interventions via play (damage avoidance + pizza), not weaker
  assist contract.
- Keep **0 lives lost**. Clean work that removes heals for a stage is separate
  (`T4-CLEAN-S*`).

## Read first
- `docs/ASSIST_CONTRACT.md`
- `docs/BASELINE_METRICS.md`
- `docs/CLEAN_PLAYBOOK.md` (pizza scope rules)

## Do
1. Identify top heal stages from dry-run stage damage (Techno / Prehist /
   Starbase first).
2. One play knob; measure e-heal count on continuous-faithful probe if possible.
3. Residual → `T4-ASSIST-DRYRUN` for whole-run heal count proof.

## Acceptance
- [ ] Contract threshold/restore unchanged
- [ ] Residual with expected heal impact
- [ ] No STATUS self-apply

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run
# planner gate after knob — compare health_guard_interventions
```
