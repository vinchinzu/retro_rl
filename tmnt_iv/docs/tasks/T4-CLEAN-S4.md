# TASK T4-CLEAN-S4: Technodrome Clean multi-entry suite

## Recipe step
probe suite

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe_stage4_clean.py` (create from stage1 pattern if missing)
- `policy.py` — one Technodrome-local knob only if RED
- residual: `docs/tasks/T4-CLEAN-S4-residual.md`

## Context
- Stage byte **3**. Duo Tokka/Rahzar: left-flank + stall suppress; tank throws.
- Assisted bucket **1,022** damage — largest stage share; Clean will be hard.
- Prefer continuous-faithful `RaphFullHardStage4` / Duo entries.

## Read first
- `docs/CLEAN_PLAYBOOK.md`
- `scripts/probe_stage1_clean.py` (template)
- `scripts/run_stage4_segment.py`

## Do
1. Scaffold `probe_stage4_clean.py` heal=none multi-entry suite if absent.
2. Entries: fight-ready Stage4 / RaphFullHardStage4 + Duo; optional bridge.
3. 0 e-heals, 0 lives lost through stage_advance.
4. Do not global pizza seek.

## Acceptance
- [ ] Suite tool exists and runs
- [ ] Multi-entry 0 e-heals / 0 lives lost (or residual with one next knob)
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage4_clean --suite
```
