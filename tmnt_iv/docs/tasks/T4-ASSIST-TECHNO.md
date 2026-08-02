# TASK T4-ASSIST-TECHNO: Cut Technodrome damage (assisted continuous)

## Recipe step
policy knob

## Model
Luna

## Wave type
implement

## Own files only
- `policy.py` — Technodrome / duo / tank knobs only (**one** named group)
- residual: `docs/tasks/T4-ASSIST-TECHNO-residual.md`

## Context
- Baseline stage damage **1,022** (21.9% of route) — largest bucket.
- Continuous-faithful: `RaphFullHardStage4` clears ~30k f / 886 dmg / 13 heals.
- Duo left-flank + stall suppress already landed; tank + wall escape matter.
- Checkpoint gains often fail transfer — full dry-run before BASELINE claim.

## Read first
- `docs/BASELINE_METRICS.md`
- `docs/STATUS.md` (Tokka/Rahzar notes)
- `scripts/capture_raph_states.py` / RaphFullHard* states

## Do
1. Probe from `RaphFullHardStage4` / Duo under emergency assist.
2. One knob aimed at damage or heal count (not fight-length only).
3. If probe green: residual → `T4-ASSIST-DRYRUN` (planner dry-run).
4. Do not park Slash/other stage knobs in this card.

## Acceptance
- [ ] Named one-knob change with probe metrics
- [ ] Residual routes dry-run or next knob
- [ ] No STATUS/BASELINE self-apply

## Verify commands
```bash
# example probe via stage4 segment or local grind focus
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.run_stage4_segment --help
```
