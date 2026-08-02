# TASK T4-CLEAN-S9: Starbase + Super Shredder form-2 Clean (no iframe)

## Recipe step
probe suite + policy

## Model
Luna → planner for continuous claim

## Wave type
implement

## Own files only
- `scripts/probe_stage9_clean.py` (create if missing)
- `policy.py` — form-2 dodge / Starbase hover knobs (**one** named group)
- residual: `docs/tasks/T4-CLEAN-S9-residual.md`

## Context
- Stage bytes **8–9**. Hover Foot need jump-slash; form-2 demutation bypasses
  HP — currently Protection-assisted iframe hold at 1.
- **Hard Clean gate:** form-2 survival with iframe guard frames == 0.
- Prefer `RaphFullHardBoss9` / phase2 states for probes; continuous-faithful
  still required before whole-run claim.
- Pairs with `T4-ASSIST-IFRAME` (assisted shrink can land first).

## Read first
- `docs/CLEAN_PLAYBOOK.md`
- `docs/ASSIST_CONTRACT.md`
- `scripts/run_stage9_segment.py`
- `scripts/record_full_hard_run.py` (iframe hold site)

## Do
1. Scaffold heal=none + **no iframe** form-2 probes.
2. Prove form-2 HP→0 without `player_iframes` writes.
3. Starbase waves: 0 e-heals, 0 lives lost on multi-entry.
4. If RED: residual names demutation timing / wall dodge one knob.
5. Do not claim whole-run Clean here — route residual to `T4-CLEAN-FULL`.

## Acceptance
- [ ] Form-2 probe: iframe frames == 0 and kill holds (or residual)
- [ ] Starbase multi-entry Clean or residual
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage9_clean --suite
```
