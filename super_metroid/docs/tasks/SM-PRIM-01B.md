# TASK SM-PRIM-01B: Migrate call sites to `settle_hold`

## Recipe step
primitive promote

## Model
Luna

## Wave type
implement

## Own files only
- One module per dispatch (serialize): prefer non-hot first
  - OK parallel with geometry if not same file
  - **Serialize** if editing `business_climb` / `hijump_return` / `varia_return`

## Context
- `settle_hold` already extracted + unit-tested (SM-PRIM-01)
- Migrate **in-file** repeated `hold(session, N, reason=…settle…)` to
  `settle_hold(session, N, reason=…)` where N matches the primitive default
  or pass explicit frames

## Do
1. Pick **one** controller file; replace obvious settle holds only
2. Keep reason strings informative
3. Pure re-check only if the file is on continuous spine climb
   (`business_climb` → pure `business-to-warehouse` from
   `continuous_like_business_climb_entry.state`)

## Do not
- Change settle frame lengths (that is a tighten card)
- Touch continuous / STATUS

## Acceptance
- [ ] Tests for controller_common still green
- [ ] If climb file: pure business still green
- [ ] Residual lists remaining files for 01C…

## Verify
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
```
