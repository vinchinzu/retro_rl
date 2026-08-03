# TASK T4-CLEAN-S3-BOSS: Rat King pizza-only on LiveHard path

## Recipe step
probe / policy knob

## Model
Luna / Gemini

## Wave type
implement

## Own files only
- `policy.py` — one Rat King / sewer-boss knob only if RED
- residual: `docs/tasks/T4-CLEAN-S3-residual.md`

## Context
- Prefer LiveHard / non-last-life entries. Do not fail the card on last-life
  `0x0B` fade artifact after boss HP 0.
- GREEN: boss HP→0 and stage progression holds without life_loss on LiveHard.

## Do
1. Probe LiveHard or Boss path per `probe_stage3_clean`.
2. One knob only if RED (long poke / wall / stall — not jump-slash thrash).
3. Residual metrics + next card.

## Do not
- Spike LEFT thrash; dumpster WalkProgress thrash
- STATUS edit

## Acceptance
- [ ] LiveHard boss path holds **or** RED residual one knob
- [ ] Artifact fade not mislabeled as play failure

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite
```
