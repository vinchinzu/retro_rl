# TASK T4-CLEAN-S3-STAB: Sewer stabilize (suite + assisted dry-run)

## Recipe step
stabilize

## Model
Flash / Gemini

## Wave type
stabilize

## Own files only
- residual verify paste only

## Context
- After sewer KEEP knobs: re-verify Clean suite + assisted dry-run. No knobs.
- Report assisted deltas vs baseline; planner decides revert.

## Do
1. Stage3 Clean suite.
2. Assisted dry-run.
3. Residual metrics; Next PLANNER-GATE.

## Do not
- Policy / BASELINE / STATUS edits

## Acceptance
- [ ] Suite + dry-run metrics pasted
- [ ] Zero policy diff

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run
```
