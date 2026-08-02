# TASK T4-CLEAN-S5: Prehistoric / Slash Clean multi-entry suite

## Recipe step
probe suite

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe_stage5_clean.py` (create if missing)
- `policy.py` — one Prehistoric/Slash-local knob only if RED
- residual: `docs/tasks/T4-CLEAN-S5-residual.md`

## Context
- Stage byte **4**. Dinos need B+Y; **no jump-slash on Slash shell**.
- Production Slash spin dodge **52** — spin 40 parked (continuous +807 dmg).
- Prefer `RaphFullHardBoss5` / RaphDiagStage5 entries.

## Read first
- `docs/CLEAN_PLAYBOOK.md`
- `docs/SLASH_PATTERN_LAB.md` (if present)
- `scripts/run_stage5_segment.py`

## Do
1. Scaffold heal=none multi-entry suite (stage entry + Slash boss).
2. 0 e-heals, 0 lives lost; do not port spin-40 without full dry-run.
3. Residual one knob if RED.

## Acceptance
- [ ] Multi-entry suite green or residual with next card
- [ ] Spin-40 ban held unless planner re-opens with dry-run proof
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage5_clean --suite
```
