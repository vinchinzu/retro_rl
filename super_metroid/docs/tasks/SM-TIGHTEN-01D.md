# TASK SM-TIGHTEN-01D: Document settle 12f pure-green baseline + continuous verify recipe

## Recipe step
docs / residual (Flash OK) — **no controller edit unless settles ≠ 12f**

## Model
Flash

## Own files only
- `docs/tasks/SM-TIGHTEN-01D-note.md` (**create**)
- `routes/kpdr/business_climb.py` **only if** the eight platform settles are not
  already 12f (then set them to 12f — known pure-green with 4-jump setup)

Do **not** change setup jump count, runup, continuous, STATUS.

## Context
Planner pure (2026-07-31): 4-jump setup + **12f** settles → pure business
GREEN (~3721f). 5f continuous-red historically; 3-jump pure-red. This card
locks the documentation for the pure-green settle baseline and the exact
planner continuous commands / dwell compare fields. Executors often skip
writing the continuous recipe clearly.

## Read first
- business_climb settles
- SM-TIGHTEN-01B-note, 01C-note
- SM-PURE-ISO-note
- QUEUE Wave-4 rollup

## Do
1. Confirm eight settles are 12f (fix if not, without touching setup).
2. Write note: pure command + expected green; continuous `--to kraid` recipe;
   dwell fields `business_to_warehouse`; rollback 12→20 if continuous fails;
   explicit non-claims.
3. Optional: run pure once and paste (good).

## Acceptance
- [ ] Note complete
- [ ] Settles 12f if card touched controller
- [ ] pytest controller_common if controller touched

## Verify commands
```bash
rg -n "business_.*_settle" super_metroid/routes/kpdr/business_climb.py
uv run python super_metroid/scripts/probe/kpdr.py pure business-to-warehouse \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/continuous_like_business_climb_entry.state
```
