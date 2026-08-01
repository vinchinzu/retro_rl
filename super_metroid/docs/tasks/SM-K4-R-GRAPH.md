# TASK SM-K4-R-GRAPH: Promote reverse edges after pure chain (docs + locks)

## Recipe step
2 graph edge + tracker

## Model
Flash (docs) or Luna (tests)

## Wave type
implement

## Own files only
- `progression.py` (only edges pure-green this wave)
- `tests/test_progression.py`
- `docs/routes/KPDR_TRACKER.csv` + `KPDR_TRACKER.md`
- `docs/SOURCE_STATES.md` (if missing rows)

## Context
Promote **only** edges that already have pure green evidence:
- Already promoted this session: `kraid_to_eye_return`, `eye_to_baby_return`,
  `baby_to_kihunter_return` → `controller_dev`
- After R-02 green: `kihunter_to_zeela_return` → `controller_dev`
- After R-03 green: `zeela_to_warehouse_return` → `controller_dev`
- Never `continuous` here

## Do
1. Set verification strings to match pure evidence only
2. Update tracker K3.3–K3.7 rows
3. Keep path_verification locks: reverse path still not all_continuous
4. Residual lists any edge still `unverified`

## Do not
- STATUS.md continuous frame tables
- continuous.py compose

## Verify
```bash
uv run pytest super_metroid/tests/test_progression.py -q
```
