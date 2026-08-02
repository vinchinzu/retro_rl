# TASK SM-CLEAN-CONTRACT: Clean-track docs + dual-path rules

## Recipe step
docs

## Model
Flash

## Wave type
implement

## Own files only
- `docs/CLEAN_TRACK.md` (exists; polish only if needed)
- `docs/ASSIST_CONTRACT.md` — short **Clean mode** pointer section only
- `docs/README.md` — index row for Clean track
- optional residual: `docs/tasks/SM-CLEAN-CONTRACT-residual.md`

Do **not** edit `STATUS.md` primary tip, `routes/continuous.py`, or assisted
baselines.

## Context
- Primary path remains **Bronze / Resource-assisted** (Frog Save tip).
- Clean track: no energy, no ammo writes — parallel privilege reduction.
- Contract: `docs/CLEAN_TRACK.md`.

## Read first
- `docs/CLEAN_TRACK.md`
- `docs/ASSIST_CONTRACT.md`
- `../../docs/BENCHMARK_SPEC.md` (Clean vs Resource-assisted)

## Do
1. Ensure `CLEAN_TRACK.md` is linked from docs index.
2. Add a short section at the bottom of `ASSIST_CONTRACT.md`:
   - Clean = both energy and ammo assists off
   - Primary product path stays assisted
   - Point to `CLEAN_TRACK.md` for process / artifacts / tickets
3. Do not change Allowed/Forbidden assisted writes.

## Acceptance
- [ ] ASSIST_CONTRACT has Clean pointer without weakening assisted rules
- [ ] docs/README indexes Clean track
- [ ] No STATUS primary tip rewrite

## Verify commands
```bash
rg -n "CLEAN_TRACK|Clean mode|Clean track" super_metroid/docs/ASSIST_CONTRACT.md super_metroid/docs/README.md super_metroid/docs/CLEAN_TRACK.md
```
