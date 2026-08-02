# TASK T4-ASSIST-PREHIST: Cut Prehistoric / Slash damage (assisted)

## Recipe step
policy knob

## Model
Luna

## Wave type
implement

## Own files only
- `policy.py` — Prehistoric / Slash knobs only (**one** named group)
- residual: `docs/tasks/T4-ASSIST-PREHIST-residual.md`

## Context
- Baseline stage damage **861** (18.4%).
- Slash spin 40: probe win, continuous **+807** total damage — **parked**.
- Production spin stays **52** unless dry-run proves better trajectory.
- Prefer `RaphFullHardBoss5` probes; always full dry-run before promote.

## Read first
- `docs/BASELINE_METRICS.md`
- `docs/CLEAN_PLAYBOOK.md` (spin-40 ban)
- `docs/SLASH_PATTERN_LAB.md`

## Do
1. Probe Slash / Prehistoric under emergency assist.
2. One knob; never blind-port spin-40.
3. Residual → `T4-ASSIST-DRYRUN` if probe improves without soft-lock risk.

## Acceptance
- [ ] One-knob residual with probe metrics
- [ ] Spin-40 ban held or dry-run evidence attached
- [ ] No STATUS self-apply

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.run_stage5_segment --help
```
