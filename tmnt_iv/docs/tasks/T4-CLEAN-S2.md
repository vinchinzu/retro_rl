# TASK T4-CLEAN-S2: Alleycat Blues Clean multi-entry suite

## Recipe step
probe suite

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe_stage2_clean.py` (if needed)
- `policy.py` — **one** Alleycat-local knob only if suite RED
- residual: `docs/tasks/T4-CLEAN-S2-residual.md`

## Context
- Stage byte **1**. Metalhead already Clean; early/mid Foot packs life_loss.
- Pizza: underfoot always; far seek **only between waves** (mid-wave chase
  burned continuous).
- Playbook: no mid-wave pizza chase, no pack jump-hop thrash, no elev≥44
  generic jump on Alleycat.
- Evidence dir: `recordings/stage2_clean_track/`.

## Read first
- `docs/CLEAN_PLAYBOOK.md`
- `docs/STATUS.md` (Stage 2 Clean section)
- `scripts/probe_stage2_clean.py`

## Do
1. Run `--suite` heal=none multi-entry (Stage2 checkpoint + Stage1_Clear /
   continuous-faithful bridge; power-on through Alleycat if available).
2. Target: all required entries stage_advance, **0 e-heals, 0 lives lost**.
3. If RED: residual names one failure window + one next knob (no thrash).
4. If shared policy edit: re-verify assisted dry-run or narrow Stage2 emergency
   probe after suite green (stabilize note).

## Acceptance
- [ ] Suite green 0 e-heals / 0 lives lost on required entries
- [ ] Playbook bans not re-opened
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --suite
```
