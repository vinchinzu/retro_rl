# TASK T4-CLEAN-S2-REACH: Full Stage2 Clean — metric progress only

## Recipe step
policy knob **or** probe-only remeasure

## Model
Luna / Gemini

## Wave type
implement

## Own files only
- `policy.py` — **at most one** Alleycat-local named constant or one small branch
  named on residual “One change”
- residual: `docs/tasks/T4-CLEAN-S2-residual.md`
- unit test only if new branch needs a pin

## Context
- Full `Stage2` checkpoint still **life_loss** under pizza-only (Clean ≫ assisted).
- This card does **not** require stage_advance.
- GREEN = measurable REACH win vs residual baseline (pick **one**):
  - lower `max_hit` (target: no 24-dmg pack hit), **or**
  - lower `damage_taken`, **or**
  - higher `frames` before death, **or**
  - higher `min_hp` at comparable progress
- Baseline residual (min_range=0 era): Stage2 ~7,239f / 96 dmg / min 4 / max_hit 24.
  Re-PROBE first if suite JSON is newer.

## Read first
- `docs/tasks/CLEAN_LADDER.md` (REACH metrics)
- `docs/tasks/T4-CLEAN-S2-residual.md` (**Rejected knobs** — do not re-try)
- `docs/CLEAN_PLAYBOOK.md`
- `recordings/stage2_clean_track/clean_stage2.json` if present

## Do
1. If no fresh baseline: run Stage2-only probe; paste hits into residual.
2. Apply **one** residual-named change (default next: prefer
   [T4-CLEAN-S2-EDGE](T4-CLEAN-S2-EDGE.md) over Y-tol thrash).
3. Re-run Stage2-only probe; paste before/after.
4. KEEP only if a REACH metric improves **and** Boss2+w17 still clear
   (quick `--state Boss2` + w17, or full suite if time).
5. If worse or desync: revert knob; mark REJECT on residual table.

## Do not
- Widen `_ALLEY_Y_TOLERANCE` (6→12/8 already **rejected**; 18 unproven thrash)
- Mid-wave far pizza, pack jump-hop, elev≥44 generic jump
- Edit STATUS / claim SUITE or CKPT green
- Second knob in the same session

## Acceptance
- [ ] Before/after metrics from JSON
- [ ] REACH metric improved **or** knob REJECTED with numbers
- [ ] Playbook bans held
- [ ] Residual Next card: `T4-CLEAN-S2-CKPT` if stage_advance else stay REACH/EDGE

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Stage2
# if policy touched:
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Boss2
uv run pytest tmnt_iv/tests/test_policy.py -q
```
