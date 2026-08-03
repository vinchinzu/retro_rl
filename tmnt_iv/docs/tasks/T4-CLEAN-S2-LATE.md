# TASK T4-CLEAN-S2-LATE: Pre-boss w17 pizza-only clear

## Recipe step
probe suite (single entry)

## Model
Flash / Gemini

## Wave type
implement

## Own files only
- residual note only if re-verify RED

## Context
- **Already GREEN** historically (`Stage2_Clear_w17_cam27882` → Metalhead clear).
- Re-run after Alleycat knobs via STAB.

## Do
1. Probe w17 state only (or suite row).
2. GREEN if stage_advance, pizza-only.

## Acceptance
- [ ] w17 entry stage_advance **or** residual one knob
- [ ] No STATUS edit

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Stage2_Clear_w17_cam27882
```

## Status
**done** (re-check via STAB).
