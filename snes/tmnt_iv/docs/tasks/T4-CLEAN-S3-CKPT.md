# TASK T4-CLEAN-S3-CKPT: LiveHard Sewer full stage_advance (Clean)

## Recipe step
probe / policy knob

## Model
Luna

## Wave type
implement

## Own files only
- `policy.py` — one sewer-local knob only if RED
- residual: `docs/tasks/T4-CLEAN-S3-residual.md`

## Context
- GREEN: LiveHard (or preferred suite entry) pizza-only **stage_advance**,
  0 e-heals, 0 life_loss.
- Prefer REACH work first.

## Do
1. Probe LiveHard full stage.
2. stage_advance → Next BRIDGE; else residual one failure window + one card.

## Acceptance
- [ ] stage_advance **or** RED residual thin next card
- [ ] No last-life fade false fail

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite
```
