# TASK T4-CLEAN-S3-REACH: Sewer LiveHard metric progress (Clean)

## Recipe step
policy knob **or** remeasure

## Model
Luna / Gemini

## Wave type
implement

## Own files only
- `policy.py` — one sewer-local knob (default: residual 0x1C spike column)
- residual: `docs/tasks/T4-CLEAN-S3-residual.md`

## Context
- Full LiveHard clear is hard. This card GREENS on REACH metrics only:
  fewer 0x1C hits, lower damage, higher min_hp, farther frames before death.
- Does **not** require full stage_advance.

## Read first
- `docs/tasks/CLEAN_LADDER.md`
- `docs/STATUS.md` (Stage 3 Clean notes)
- `docs/CLEAN_PLAYBOOK.md`

## Do
1. Baseline LiveHard probe if residual empty.
2. One knob aimed at residual 0x1C / spike lane.
3. Before/after JSON; KEEP or REJECT.
4. Next: CKPT if stage_advance else stay REACH.

## Do not
- Re-open dumpster thrash or spike LEFT thrash
- STATUS / second knob

## Acceptance
- [ ] Before/after metrics from JSON
- [ ] REACH improve **or** REJECT with numbers

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite
```
