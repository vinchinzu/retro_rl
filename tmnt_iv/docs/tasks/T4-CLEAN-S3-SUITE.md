# TASK T4-CLEAN-S3-SUITE: Sewer multi-entry Clean verify (no knobs)

## Recipe step
probe suite

## Model
Flash / Gemini

## Wave type
stabilize

## Own files only
- residual Suite table only

## Context
- Gated until CKPT (+ BRIDGE if required by suite) green.
- Required entries = LiveHard multi-entry per probe; not last-life fade.

## Do
1. Run suite; copy JSON into residual.
2. GREEN only if required entries stage_advance.

## Do not
- Policy / STATUS edits

## Acceptance
- [ ] Residual matches JSON
- [ ] GREEN only on required LiveHard entries

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite
```
