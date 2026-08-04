# TASK T4-CLEAN-ARTIFACTS: Isolate clean report/video paths

## Recipe step
infra

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/record_full_hard_run.py` — path default helper when clean flags set
- `tests/` — unit tests for path helper (create or extend)
- `docs/tasks/T4-CLEAN-ARTIFACTS-residual.md`

Do **not** change default assist behavior, default assisted stems, or
`STATUS.md`. Do **not** overwrite existing assisted recordings.

## Context
- Risk: a clean dry-run with default `--report` / `--output` could clobber
  `tmnt_iv_full_hard_credits.{mp4,json}` or `tmnt_iv_full_hard_dry_run.json`.
- Clean stems: `tmnt_iv_full_hard_clean.{json,mp4}` (and dry-run variant
  `tmnt_iv_full_hard_clean_dry_run.json` if desired).
- Spec: `docs/CLEAN_TRACK.md` hard rule #2.
- Super Metroid analog: `SM-CLEAN-ARTIFACTS`.

## Read first
- `docs/CLEAN_TRACK.md`
- `scripts/record_full_hard_run.py` (`_build_parser`, `main` dry-run rename)

## Do
1. Add helper e.g. `default_full_run_paths(*, clean: bool, dry_run: bool)` that
   returns non-overlapping paths for clean vs assisted.
2. Wire so clean mode defaults report/video to clean stems **unless** user
   passed explicit `--report` / `--output`.
3. Unit-test assisted defaults unchanged; clean stems differ.
4. Residual → `T4-CLEAN-CLI` if `--clean` alias not done here.

## Acceptance
- [ ] Assisted defaults still `tmnt_iv_full_hard_credits.*` / dry_run rename
- [ ] Clean defaults never equal assisted stems
- [ ] Explicit `--report` still wins
- [ ] Tests green; residual written

## Verify commands
```bash
uv run pytest tmnt_iv/tests/ -q -k "clean or artifact or path"
```
