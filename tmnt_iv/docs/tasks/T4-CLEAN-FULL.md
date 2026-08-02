# TASK T4-CLEAN-FULL: Continuous power-on → hard credits (Clean) ★

## Recipe step
continuous

## Model
Luna → planner review

## Wave type
implement

## Own files only
- residual: `docs/tasks/T4-CLEAN-FULL-residual.md`
- only if route fails: **one** policy knob (then assisted dry-run re-verify)

Depends: Clean infra green (`ARTIFACTS` / `CLI` / `INTEGRITY`); stage suites
S1–S9 Clean green preferred (S9 form-2 without iframe is the hard gate).

## Context
- Assisted continuous is already green (00:57:19 / 4,667 dmg / 65 e-heals /
  4,635 iframe / 0 lives).
- Clean disables emergency HP + form-2 iframe; pizza + play only.
- **Must not** overwrite `recordings/tmnt_iv_full_hard_credits.*` or
  `tmnt_iv_full_hard_dry_run.json`.
- Primary STATUS gate stays assisted until program decision.
- Super Metroid analog: `SM-CLEAN-BOMBS` (tip continuous Clean).

## Read first
- `docs/CLEAN_TRACK.md`
- `docs/CLEAN_PLAYBOOK.md`
- `docs/ASSIST_CONTRACT.md`
- `docs/BASELINE_METRICS.md`
- `scripts/record_full_hard_run.py`

## Do
1. Run clean continuous dry-run with clean artifact paths.
2. Require: hard credits event `0x1A`, 0 lives lost, 0 e-heals, 0 iframe frames,
   0 state loads, 0 stage writes, no A-special.
3. If RED: residual → named stage failure + one next card
   (`T4-CLEAN-S2`…`S9` or policy one-knob).
4. If GREEN: dual re-verify (`T4-CLEAN-STAB`) then STATUS secondary
   (`T4-CLEAN-STATUS`) — planner.

## Acceptance
- [ ] Clean report success + credits complete
- [ ] E-heals == 0, iframe frames == 0, life losses == 0
- [ ] Assisted baselines files unchanged
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --clean --dry-run
# re-verify once path stable:
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --clean --dry-run \
  --report tmnt_iv/recordings/tmnt_iv_full_hard_clean_reverify.json
```

## Residual routing
- GREEN → `T4-CLEAN-STAB` then `T4-CLEAN-STATUS`
- RED life_loss / e-heal would-have-fired → stage suite card for failing stage
- RED form-2 → `T4-CLEAN-S9` / `T4-ASSIST-IFRAME` play solution
