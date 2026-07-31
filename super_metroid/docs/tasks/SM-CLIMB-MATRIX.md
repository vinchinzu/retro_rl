# TASK SM-CLIMB-MATRIX: Pure A/B matrix for business climb knobs (report only)

## Recipe step
diagnostics / harness (pure matrix — **not** continuous; **do not leave red knobs**)

## Model
Luna

## Own files only
- `docs/tasks/SM-CLIMB-MATRIX-report.md` (**create**)
- temporary edits to `routes/kpdr/business_climb.py` **allowed only if fully
  restored** to the starting tree values before EXIT

Do **not** leave continuous-risk knobs red. Do not edit continuous/STATUS.

## Context
Known pure results (scratch Business entry source):
- 4 setup + 12f settle → pure **GREEN** (~3721f) [planner 2026-07-31]
- 3 setup `RIGHT,LEFT,LEFT` + 12f settle → pure **RED** @ 1339_ground y=1419
- 4 setup + 5f settle → continuous **RED** historically (Wave-3 01B)

Fill the matrix with pure probes only (fast):

| setup | settle | pure? | fail pin |
|-------|--------|-------|----------|
| 4     | 20     | ?     | |
| 4     | 12     | GREEN | |
| 4     | 8      | ?     | |
| 3 RLL | 12     | RED   | y=1419 |
| 3 LRL | 12     | ?     | |
| 3 LLR | 12     | ?     | |

## Read first
- `routes/kpdr/business_climb.py` settle + setup
- `docs/tasks/SM-PURE-ISO-note.md`
- QUEUE Wave-4 rollup

## Do
1. Note starting settle durations + setup tuple (git/read).
2. For each matrix cell: apply knobs, run pure business-to-warehouse with the
   named source, record success/frames or fail pin.
3. Cap total pure runs ≤ 8. Prefer cells that inform next tighten cards.
4. **Restore** business_climb.py to starting knobs (or to the known pure-green
   baseline 4-jump + 12f settle if that was starting — match start-of-card).
5. Write report table + ranked recommendations (what is pure-safe).
6. Explicit: pure ≠ continuous.

## Residual required
- Full matrix table
- Final file state matches start (diff empty on knobs or documented green baseline)
- Planner continuous still required for any savings claim

## Do not
- Leave pure-red knobs committed
- continuous.py / STATUS
- Claim STATUS savings

## Acceptance
- [ ] Report complete with ≥5 pure cells filled
- [ ] Controller knobs restored
- [ ] pytest controller_common green after restore

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
uv run python super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state
# must green on restored knobs
```
