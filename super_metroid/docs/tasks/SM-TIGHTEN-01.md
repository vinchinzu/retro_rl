# TASK SM-TIGHTEN-01: Deep analysis — business_to_warehouse dwell (report only)

## Recipe step
efficiency analysis (no controller patch in this card)

## Model
Flash

## Own files only
- `docs/tasks/SM-TIGHTEN-01-report.md` (**create**)

Read many files; write **only** the report.

## Context
- Dwell: `business_to_warehouse` ~2,257f on start_to_varia
- Controller: `routes/kpdr/business_climb.py` → `play_business_to_warehouse`
- Continuous uses this hop on `--to kraid` / `--to varia` prefix

## Read first (all of these)
- `routes/kpdr/business_climb.py` (full function + helpers)
- `docs/tasks/SM-K4-05-dwell-report.md`
- `docs/ROOM_TIMER.md`
- `docs/STATUS.md` optional-tighten bullets (read only)
- Grep action_reasons containing business/warehouse in
  `recordings/start_to_varia.json` if needed (jq / python one-liner)

## Do (thorough)
1. Run dwell tool focused if helpful:
   ```bash
   uv run python super_metroid/scripts/export/split_dwell.py \
     super_metroid/recordings/start_to_varia.json --top 20
   uv run python super_metroid/scripts/export/split_dwell.py \
     super_metroid/recordings/start_to_varia.json --reasons --top 30
   ```
2. Write `docs/tasks/SM-TIGHTEN-01-report.md` with:
   - Function structure map (phases / hold loops / timeouts)
   - Top suspected waste: idle settles, standing gates, retry paths, overshoot
   - Line-level references (function + approximate line / reason labels)
   - 2–3 concrete patch recipes for a **future** implement card
     (each: expected frames saved band = unknown; do not claim savings)
   - Acceptance command a future card should run
     (`continuous.py --to kraid --no-video` or pure probe path)
   - Explicit: no frame savings claimed without re-record

## Do not
- Edit controllers / continuous / STATUS
- Implement the tighten

## Acceptance
- [ ] Report file exists, ≥1 page, with line/reason references
- [ ] Diff is report-only

## Verify commands
```bash
test -f super_metroid/docs/tasks/SM-TIGHTEN-01-report.md
wc -l super_metroid/docs/tasks/SM-TIGHTEN-01-report.md
```
