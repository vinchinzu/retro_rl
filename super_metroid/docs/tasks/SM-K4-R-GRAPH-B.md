# TASK SM-K4-R-GRAPH-B: Promote kihunter→zeela pure edge (controller_dev)

## Recipe step
2 graph edge + tracker

## Model
Flash

## Wave type
implement

## Own files only
- `progression.py` (`kihunter_to_zeela_return` verification string only)
- `tests/test_progression.py` (locks only if needed)
- `docs/routes/KPDR_TRACKER.csv` + `KPDR_TRACKER.md` (K3.6 pure notes)
- `docs/SOURCE_STATES.md` (gaps row for kihunter→zeela only if stale)

## Context
- SM-K4-R-CLIMB-REDESIGN pure **GREEN** ~1716f → `0xA471`
  source `scratch/post_kihunter_to_zeela_return.state`
- Edge `kihunter_to_zeela_return` is still `unverified` in progression
- R-03 / R-03B still open — do **not** promote `zeela_to_warehouse_return`
- Never mark `continuous` here

## Do
1. Set `kihunter_to_zeela_return` verification to `controller_dev` only
2. Tracker: note pure green frames ~1716 + source path; K3.6 pure open→dev
3. Keep reverse path_verification / continuous locks honest
4. Residual if any edge still wrongly claimed

## Do not
- STATUS.md frame tables / continuous tip claims
- Promote zeela→warehouse or warehouse→business
- continuous.py

## Acceptance
- [ ] `kihunter_to_zeela_return` is `controller_dev`
- [ ] `zeela_to_warehouse_return` remains `unverified`
- [ ] progression tests green
- [ ] Dual-track / pure-only non-claim in residual

## Verify
```bash
uv run pytest super_metroid/tests/test_progression.py -q
rg -n "kihunter_to_zeela_return|zeela_to_warehouse_return" super_metroid/progression.py
```
