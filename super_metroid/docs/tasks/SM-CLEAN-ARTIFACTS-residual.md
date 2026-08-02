# Residual — SM-CLEAN-ARTIFACTS

## Outcome
GREEN

## What landed
- `default_artifacts(stem, *, clean=False)` + `clean_artifact_stem`
- `default_tip_artifact_paths(..., clean=)` / room-timing clean stem
- CLI uses clean stems whenever any resource assist is disabled

## Verify
`uv run pytest super_metroid/tests/test_clean_track.py -q`

## Next card ID
SM-CLEAN-MORPH (compose) — infra chain complete with CLI + INTEGRITY
