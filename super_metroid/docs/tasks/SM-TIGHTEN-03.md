# TASK SM-TIGHTEN-03: Deep analysis — terminator_energy_tank dwell (report only)

## Recipe step
efficiency analysis (no controller patch)

## Model
Flash

## Own files only
- `docs/tasks/SM-TIGHTEN-03-report.md` (**create**)

Read many files; write **only** the report.

## Context
- Ranked candidate from `docs/tasks/SM-K4-05-dwell-report.md`
- Split: `terminator_energy_tank` ~4,693f on `start_to_varia`
- Likely controller path: terminator / E-Tank detour in early Brinstar routes
  (search `terminator`, `energy_tank`, related play_* in `routes/kpdr/` and
  continuous split labels)

## Read first (all)
- `docs/tasks/SM-K4-05-dwell-report.md`
- `docs/ROOM_TIMER.md`
- Grep / read controllers owning terminator E-Tank (e.g. under `routes/kpdr/`,
  `routes/`, continuous split registration)
- `scripts/export/split_dwell.py` usage
- Optional: action_reasons in `recordings/start_to_varia.json` via dwell tool

## Do (thorough)
1. Run:
   ```bash
   uv run python super_metroid/scripts/export/split_dwell.py \
     super_metroid/recordings/start_to_varia.json --top 20
   uv run python super_metroid/scripts/export/split_dwell.py \
     super_metroid/recordings/start_to_varia.json --reasons --top 40
   ```
2. Map the split window to controller function(s) and phase structure
3. Write `docs/tasks/SM-TIGHTEN-03-report.md` with:
   - Phase / reason map with line references where possible
   - Top waste candidates (idle, retry, overshoot, detour necessity)
   - 2–3 concrete **future** patch recipes (expected band speculative only)
   - Acceptance command for a future implement card
     (prefer shortest continuous tip that still includes the split, or pure probe)
   - Explicit: no frame savings claimed without re-record
4. If the split is mostly unavoidable boss/scripted time, say so and mark
   implement card as **low priority**

## Residual required (super-clean)
- Controller file path + primary function name
- Whether implement is worth a card (yes/no + why)
- Diff is report-only

## Do not
- Edit controllers / continuous / STATUS
- Implement the tighten

## Acceptance
- [ ] Report ≥1 page with line/reason refs
- [ ] Report-only diff

## Verify commands
```bash
test -f super_metroid/docs/tasks/SM-TIGHTEN-03-report.md
wc -l super_metroid/docs/tasks/SM-TIGHTEN-03-report.md
```
