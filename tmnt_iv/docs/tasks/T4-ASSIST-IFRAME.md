# TASK T4-ASSIST-IFRAME: Shrink form-2 iframe hold (path to Clean)

## Recipe step
policy knob

## Model
Luna

## Wave type
implement

## Own files only
- `policy.py` — Super Shredder form-2 dodge only (**one** group)
- residual: `docs/tasks/T4-ASSIST-IFRAME-residual.md`

## Context
- Baseline form-2 iframe guard **4,635** frames (protection assist).
- Demutation bypasses ordinary HP; wall-aware dodge reduced isolated
  `Boss9_phase2` 3,825f → 2,631f but whole-run still holds iframe.
- Feeds Clean gate `T4-CLEAN-S9` / `T4-CLEAN-FULL`.
- Intermediate wins: fewer iframe frames on dry-run while 0 lives lost.

## Read first
- `docs/ASSIST_CONTRACT.md`
- `docs/STATUS.md` (form-2 notes)
- `scripts/record_full_hard_run.py` (iframe hold)

## Do
1. Probe `Boss9_phase2` / RaphFullHardBoss9 with assist still available.
2. Improve wall dodge / cycle so demutation is avoided by play.
3. Measure iframe frames; residual may open clean form-2 probe (`T4-CLEAN-S9`).
4. Do not remove iframe hold from continuous default in this card (CLI is
   `T4-CLEAN-CLI`).

## Acceptance
- [ ] One dodge knob with probe metrics
- [ ] Continuous default still protection-assisted unless planner says otherwise
- [ ] Residual routes DRYRUN and/or CLEAN-S9

## Verify commands
```bash
# phase2-focused probe if available; else stage9 segment / capture states
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.run_stage9_segment --help
```
