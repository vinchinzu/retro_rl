# TASK T4-CLEAN-S2-SUITE: Alleycat multi-entry Clean verify (no knobs)

## Recipe step
probe suite

## Model
Flash / Gemini

## Wave type
stabilize

## Own files only
- residual Suite table: `docs/tasks/T4-CLEAN-S2-residual.md`

## Context
- **Verify only.** Gated until CKPT + BRIDGE children are green.
- GREEN: `clean_suite.json` `all_passed=true` (or passed == suite_size).
- Never invent pass counts.

## Do
1. Run full `--suite`.
2. Copy results into residual.
3. If GREEN → Next `T4-CLEAN-S2-STAB`. If RED → Next the failing rung card.

## Do not
- Edit policy or STATUS

## Acceptance
- [ ] Residual matches `clean_suite.json`
- [ ] GREEN only if all required entries stage_advance

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --suite
```
