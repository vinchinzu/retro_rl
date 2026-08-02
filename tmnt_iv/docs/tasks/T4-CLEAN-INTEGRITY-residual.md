# Residual — T4-CLEAN-INTEGRITY

## Outcome
GREEN

## What landed
- `assist_integrity` / `evaluate_clean_integrity` on `RunMetrics`
- Clean mode fails closed if e-heals or iframe frames > 0 after success path
- Report fields: `intervention_class`, `clean_track`, `assists.*_enabled`,
  `require_clean_assists`, top-level `integrity` map
- Assisted mode keeps prior telemetry shape + still reports integrity zeros

## Verify
```bash
uv run pytest tmnt_iv/tests/test_clean_track.py -q -k "integrity or assist"
```

## Next card ID
T4-CLEAN-S2 (Alleycat multi-entry Clean suite)
