# Residual — T4-CLEAN-ARTIFACTS

## Outcome
GREEN

## What landed
- `paths.clean_artifact_stem` + `paths.default_full_run_paths(clean=, dry_run=)`
- Assisted stems: `tmnt_iv_full_hard_credits.*` / `tmnt_iv_full_hard_dry_run.json`
- Clean stems: `tmnt_iv_full_hard_clean.{mp4,json}` / `*_clean_dry_run.json`
- `resolve_cli_paths` — explicit `--output` / `--report` always win
- Tests: `tests/test_clean_track.py`

## Verify
```bash
uv run pytest tmnt_iv/tests/test_clean_track.py -q
```

## Next card ID
T4-CLEAN-CLI (landed same wave)
