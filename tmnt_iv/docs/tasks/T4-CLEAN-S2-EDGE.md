# TASK T4-CLEAN-S2-EDGE: Alleycat 0x5E pack left-flank edge-wait (one knob)

## Status
**DONE — REJECT (no-op).** Executed 2026-08-03 on the proven KEEP baseline
(`_ALLEY_MIN_RANGE = 0`). Stage2 output byte-identical to baseline:
**7,239f / 96 dmg / min 4 / max_hit 24**. The micro-pause window was
unreachable — the turtle is already at x=202 at the 24-dmg hit (prog≈20,995),
so it never passes x∈[160,180] with a `walk_right`/`approach_right` reason
during progress 20800–21500. Reverted the knob; logged in
[`T4-CLEAN-S2-residual.md`](T4-CLEAN-S2-residual.md) Rejected table. Next card:
**`T4-CLEAN-S2-REACH`**.

## Recipe step
policy knob

## Model
Luna / Gemini

## Wave type
implement

## Own files only
- `policy.py` — **only** Alleycat pack edge-wait / micro-pause as specified below
- residual: `docs/tasks/T4-CLEAN-S2-residual.md`
- optional: `tmnt_iv/tests/test_policy.py` one unit pin

## Context
- Residual theory: post-pizza 24-dmg hits at `player_x≈202–204`, progress
  ~20800–21500 while walking into right-edge spawns.
- **One change:** during that progress band on stage==1 non-boss, briefly delay
  `walk_right` / hold left-flank poke zone (`player_x` ~160–180) 2–4 frames so
  0x5E closes into left-flank range instead of trading slide damage.
- Do **not** also change Y-tol, min_range, attack lock, or pizza rules.

## Read first
- `docs/tasks/T4-CLEAN-S2-residual.md`
- `docs/tasks/CLEAN_LADDER.md`
- `docs/CLEAN_PLAYBOOK.md`

## Do
1. Implement the single edge-wait / micro-pause behavior (stage-local).
2. Probe `--state Stage2`; compare max_hit / damage / frames / hits at progress
   20800–21500 vs residual baseline.
3. KEEP only if REACH metric improves without Boss2/w17 regression.
4. Residual: KEEP or REJECT with numbers; Next = REACH/CKPT/STAB.

## Do not
- Touch sewer/other stage knobs
- STATUS / QUEUE promotion
- Rejected-knob re-tries (Y-tol widen, mid-wave pizza, pack jump-slash, …)

## Acceptance
- [ ] Exactly one new Alleycat behavior / constant group
- [ ] Stage2 before/after JSON metrics
- [ ] Boss2 still stage_advance (or residual notes regression → revert)
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Stage2
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Boss2
uv run pytest tmnt_iv/tests/test_policy.py -q
```
