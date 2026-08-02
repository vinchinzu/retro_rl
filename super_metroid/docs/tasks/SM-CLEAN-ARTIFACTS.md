# TASK SM-CLEAN-ARTIFACTS: Isolate clean report/video paths

## Recipe step
infra

## Model
Luna

## Wave type
implement

## Own files only
- `routes/continuous.py` — `default_tip_artifact_paths` / clean stem helper only
- `scripts/record/continuous.py` — path default when clean flags set
- `tests/` — unit tests for path helper (create or extend small test)
- `docs/tasks/SM-CLEAN-ARTIFACTS-residual.md`

Do **not** change default tip, default assist flags, or existing assisted
`artifact_stem` values. Do **not** edit `STATUS.md`.

## Context
- Risk: `continuous.py --to bombs --no-unlimited-ammo` writes the **same**
  default `start_to_bomb_torizo.json` as assisted and destroys bronze evidence.
- Clean stems: `{artifact_stem}_clean` under `recordings/`.
- Spec: `docs/CLEAN_TRACK.md` hard rule #2.

## Read first
- `docs/CLEAN_TRACK.md`
- `routes/catalog.py` (`ContinuousTip.artifact_stem`)
- `routes/continuous.py` (`default_tip_artifact_paths`)
- `scripts/record/continuous.py`

## Do
1. Add helper e.g. `default_tip_artifact_paths(tip, *, clean: bool = False)` that
   appends `_clean` to the stem when `clean=True`.
2. Wire CLI so any run with both resource assists disabled (or `--clean` once
   present) defaults report/video to the clean stem **unless** user passed
   explicit `--report` / `--video`.
3. Unit-test assisted default stems unchanged; clean stems differ.
4. Residual → `SM-CLEAN-CLI` if `--clean` alias not done here.

## Acceptance
- [ ] Assisted defaults still `start_to_bomb_torizo.{json,mp4}` etc.
- [ ] Clean defaults never equal assisted stems
- [ ] Explicit `--report` still wins
- [ ] Tests green; residual written

## Verify commands
```bash
uv run pytest super_metroid/tests/ -q -k "artifact or tip_path or clean"  # adjust to real name
uv run python super_metroid/scripts/record/continuous.py --list
```
