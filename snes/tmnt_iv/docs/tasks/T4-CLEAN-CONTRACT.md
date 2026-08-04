# TASK T4-CLEAN-CONTRACT: Clean-track docs + dual-path rules

## Recipe step
docs

## Model
Flash

## Wave type
implement

## Own files only
- `docs/CLEAN_TRACK.md` (exists; polish only if needed)
- `docs/ASSIST_CONTRACT.md` — short **Clean mode** pointer section only
- `docs/CLEAN_PLAYBOOK.md` — index pointer to CLEAN_TRACK process
- `AGENTS.md` — tasks queue pointer (if not already)
- optional residual: `docs/tasks/T4-CLEAN-CONTRACT-residual.md`

Do **not** edit `STATUS.md` primary gate, `scripts/record_full_hard_run.py`,
or assisted baselines.

## Context
- Primary path remains **Bronze / Resource-assisted + Protection-assisted**.
- Clean track: no emergency HP, no form-2 iframe — parallel privilege reduction.
- Contract: `docs/CLEAN_TRACK.md`.
- Super Metroid analog: `SM-CLEAN-CONTRACT`.

## Read first
- `docs/CLEAN_TRACK.md`
- `docs/ASSIST_CONTRACT.md`
- `docs/CLEAN_PLAYBOOK.md`
- `../../docs/BENCHMARK_SPEC.md` (Clean vs assisted classes)

## Do
1. Ensure `CLEAN_TRACK.md` is linked from ASSIST_CONTRACT and playbook.
2. Keep Allowed/Forbidden assisted writes unchanged.
3. State explicitly: default continuous remains assisted; Clean is secondary.

## Acceptance
- [ ] ASSIST_CONTRACT has Clean pointer without weakening assisted rules
- [ ] Playbook points to CLEAN_TRACK for tickets/process
- [ ] No STATUS primary tip rewrite

## Verify commands
```bash
rg -n "CLEAN_TRACK|Clean mode|Clean track" tmnt_iv/docs/ASSIST_CONTRACT.md \
  tmnt_iv/docs/CLEAN_PLAYBOOK.md tmnt_iv/docs/CLEAN_TRACK.md
```
