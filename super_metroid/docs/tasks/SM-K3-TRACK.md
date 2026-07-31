# TASK SM-K3-TRACK: Tracker notes for reverse scaffolds K3.4–K3.7

## Recipe step
docs / tracker (chartable only — no STATUS promotion)

## Model
Flash

## Own files only
- `docs/routes/KPDR_TRACKER.csv` (**edit notes/status columns only for K3.4–K3.7**)
- regenerated `docs/routes/KPDR_TRACKER.md` via export script
- do **not** invent continuous green

## Context
Wave-2 SM-K4-R-SCAFFOLD registered reverse hops:
- `eye_to_baby_return`, `baby_to_kihunter_return`,
  `kihunter_to_zeela_return`, `zeela_to_warehouse_return`
  in `routes/kpdr/kraid_return.py` + registry
- Pure green still needs natural sources after `kraid_to_eye_return` works
- Tracker currently says “no controller scaffold yet” for K3.4–K3.7 — outdated

## Read first (all)
- `docs/routes/KPDR_TRACKER.csv` rows K3.2–K3.8
- `docs/routes/KPDR_TRACKER.md` matching section
- `scripts/export/kpdr_tracker.py`
- `routes/kpdr/kraid_return.py` (confirm segment ids)
- `docs/tasks/SM-K4-R-SCAFFOLD.md` / QUEUE residual notes
- Do **not** promote K3.3 pure-green (still open)

## Do (thorough)
1. Update CSV notes for **K3.4–K3.7** only:
   - Reflect controller scaffolds exist (`kraid_return.py` / registered segment ids)
   - Keep `status` / `layer` as **open** / **future** (or existing non-green values)
   - Add probe_command hints like `kpdr.py pure eye-to-baby-return` where the CLI
     kebab names exist (verify via `kpdr.py list` if needed)
   - notes must say scaffold only; pure needs natural source after K3.3 green
2. Leave K3.3 as pure-unverified scaffold (may lightly refresh note if inaccurate)
3. Do **not** mark any reverse hop continuous or controller_dev pure-green
4. Regenerate MD:
   ```bash
   uv run python super_metroid/scripts/export/kpdr_tracker.py
   ```
5. Spot-check regenerated MD rows match CSV

## Residual required (super-clean)
- Confirm K3.3 still not promoted
- List exact CSV cells changed
- Next pure source still blocked on door geometry

## Do not
- continuous.py / STATUS / progression verification promotion
- Claim pure green on reverse hops
- Reorder entire tracker

## Acceptance
- [ ] CSV notes updated for scaffolds
- [ ] MD regenerated
- [ ] No false continuous claims

## Verify commands
```bash
uv run python super_metroid/scripts/export/kpdr_tracker.py
rg -n "K3\.[3-7]|eye_to_baby|scaffold" super_metroid/docs/routes/KPDR_TRACKER.csv
rg -n "K3\.[3-7]|scaffold" super_metroid/docs/routes/KPDR_TRACKER.md | head -20
```
