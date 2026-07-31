# TASK SM-TIGHTEN-02: Deep analysis — hj_shaft_to_business dwell (report only)

## Recipe step
efficiency analysis (no controller patch)

## Model
Flash

## Own files only
- `docs/tasks/SM-TIGHTEN-02-report.md` (**create**)

## Context
- Dwell: `hj_shaft_to_business` ~1,885f
- Controller: `routes/kpdr/hijump_return.py` → `play_hj_shaft_to_business`
  (+ related `play_hj_room_to_shaft` if chained)

## Read first
- `routes/kpdr/hijump_return.py` (full)
- `routes/kpdr/hijump_collect.py` if referenced
- `docs/tasks/SM-K4-05-dwell-report.md`
- `docs/ROOM_TIMER.md`
- Dwell CLI on start_to_varia

## Do (thorough)
1. Run split_dwell top + reasons
2. Write report: phase map, waste candidates with reason labels, 2–3 future
   patch recipes, verify command for implement card, no savings claim

## Do not
- Edit controllers / continuous / STATUS

## Acceptance
- [ ] `docs/tasks/SM-TIGHTEN-02-report.md` exists with concrete references
- [ ] Report-only diff

## Verify commands
```bash
test -f super_metroid/docs/tasks/SM-TIGHTEN-02-report.md
```
