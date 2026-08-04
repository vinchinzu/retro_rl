# TASK T4-CLEAN-S3-BRIDGE: Stage2_Clear → Sewer Clean entry

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
- Continuous-faithful bridge from Alleycat clear. Path timing ≠ checkpoint.
- Use probe flags (`--from-stage2-clear` if available) per `probe_stage3_clean`.

## Do
1. Run bridge / continuous-faithful entry.
2. GREEN: stage_advance pizza-only; RED: residual mode + one next card.

## Acceptance
- [ ] stage_advance **or** RED residual
- [ ] JSON evidence

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite
# or --from-stage2-clear if supported by the probe CLI
```
