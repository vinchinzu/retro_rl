# TASK SM-TIGHTEN-04: Deep analysis — green_brinstar_main_shaft dwell (report only)

## Recipe step
efficiency analysis (no controller patch)

## Model
Flash

## Own files only
- `docs/tasks/SM-TIGHTEN-04-report.md` (**create**)

Read many files; write **only** the report.

## Context
- Ranked candidate from `docs/tasks/SM-K4-05-dwell-report.md`
- Split: `green_brinstar_main_shaft` ~2,806f on `start_to_varia`
- Search controllers for green shaft / elevator / main shaft traversal
  (`play_green*`, green brinstar, shaft)

## Read first (all)
- `docs/tasks/SM-K4-05-dwell-report.md`
- Controllers under `routes/kpdr/` matching green shaft / elevator
- `docs/ROOM_TIMER.md`
- Dwell tool on `recordings/start_to_varia.json`

## Do (thorough)
1. Run dwell top splits + reasons (same as SM-TIGHTEN-03)
2. Map split → controller phases with line/reason labels
3. Write `docs/tasks/SM-TIGHTEN-04-report.md` with:
   - Structure map
   - Top waste (elevator waits, platform align, idle settles)
   - 2–3 future patch recipes (speculative bands only)
   - Acceptance command for implement card
   - No savings claimed without re-record
4. Flag if mostly elevator/scripted unavoidable time

## Residual required (super-clean)
- Primary controller path
- Implement worth it? (yes/no)
- Report-only diff

## Do not
- Edit controllers / continuous / STATUS
- Implement the tighten

## Acceptance
- [ ] Report with concrete refs
- [ ] Report-only diff

## Verify commands
```bash
test -f super_metroid/docs/tasks/SM-TIGHTEN-04-report.md
wc -l super_metroid/docs/tasks/SM-TIGHTEN-04-report.md
```
