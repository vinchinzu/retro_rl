# TASK T4-CLEAN-S3: Sewer Surfin' Clean multi-entry suite

## Recipe step
probe suite

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe_stage3_clean.py` (if needed)
- `policy.py` — **one** Sewer-local knob only if suite RED
- residual: `docs/tasks/T4-CLEAN-S3-residual.md`

## Context
- Stage byte **2**. Prefer **`LiveHardStage3` (lives=2)** — last-life
  Stage3/Boss3 die on post-kill `event=0x0B` fade (checkpoint artifact).
- Residual 0x1C spikes; `SewerSpikeAvoid` jump-right landed; spike LEFT thrash
  rejected.
- Rat King: boss_active down to HP 1; long poke not jump-slash.
- Evidence dir: `recordings/stage3_clean_track/`.

## Read first
- `docs/CLEAN_PLAYBOOK.md`
- `docs/STATUS.md` (Stage 3 Clean section)
- `scripts/probe_stage3_clean.py`

## Do
1. Run `--suite` with LiveHard entries; do not gate on last-life Boss3 fade.
2. Target: stage_advance, **0 e-heals, 0 lives lost**, boss finish holds.
3. Cut residual 0x1C columns so boss entry HP stays high enough.
4. If RED: one spike/Rat King knob only; residual routes next.

## Acceptance
- [ ] Suite green on LiveHard multi-entry
- [ ] No dumpster/WalkProgress thrash re-open
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite
```
