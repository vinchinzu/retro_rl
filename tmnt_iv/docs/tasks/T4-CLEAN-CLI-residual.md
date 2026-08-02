# Residual — T4-CLEAN-CLI

## Outcome
GREEN

## What landed
- `--clean` → both assists off + clean integrity + `*_clean` default stems
- Long forms: `--no-emergency-hp`, `--no-iframe-hold` (either alone is not full Clean)
- Any assist-off still uses clean artifact stems (assisted baselines safe)
- Defaults remain both assists **ON** when flags omitted
- Module docstring documents Clean CLI

## Verify
```bash
uv run python -m tmnt_iv.scripts.record_full_hard_run --help
uv run pytest tmnt_iv/tests/test_clean_track.py -q -k "parser or cli"
```

## Next card ID
T4-CLEAN-INTEGRITY (landed same wave) → stage suites `T4-CLEAN-S2`
