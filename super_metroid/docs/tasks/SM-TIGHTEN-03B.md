# TASK SM-TIGHTEN-03B: Terminator exit idle trim (Recipe B only)

## Recipe step
efficiency implement (early spine — **not** STATUS)

## Model
Luna

## Own files only
- `routes/spore_spawn_controller.py` (**edit** terminator exit path only)
- `docs/tasks/SM-TIGHTEN-03B-note.md` (**create**)

Do **not** implement Recipe A (polling bomb tunnel — higher risk / needs RAM).
Do **not** edit continuous.py / STATUS / kpdr reverse.

## Context
- Report: `docs/tasks/SM-TIGHTEN-03-report.md`
- Split `terminator_energy_tank` ~4,693f; Recipe B: shorter exit hold / stop
  button spam after transition detected
- Aggressive: edits verified continuous pre-Spore path
- Continuous verify tip: `--to spore` (planner residual)

## Read first
- `docs/tasks/SM-TIGHTEN-03-report.md` Recipe B + phase table
- `routes/spore_spawn_controller.py` `play_parlor_to_main_shaft` terminator exit
- hold_until_room / similar helpers

## Do
1. Locate exit to Green Pirates (`0x99BD`) hold/timeout (report ~900f class).
2. Apply Recipe B only: reduce timeout ceiling **or** stop directional spam
   shortly after non-ordinary / room change detected — pick the safer of the
   two options in report; document choice.
3. Do not change bomb-tunnel 8-cycle timing (Recipe A out of scope).
4. Residual: speculative 200–500f not claimed; planner `--to spore --no-video`
   + split_dwell; rollback path.

## Residual required
- Exact before/after control flow
- Continuous verify command
- Non-claims

## Do not
- Recipe A bomb-tunnel rewrite
- continuous / STATUS
- Claim savings

## Acceptance
- [ ] Exit trim only
- [ ] pytest post_spore / controller_common green
- [ ] Residual complete

## Verify commands
```bash
uv run pytest super_metroid/tests/test_post_spore_controller.py super_metroid/tests/test_controller_common.py -q
# planner:
# uv run python super_metroid/scripts/record/continuous.py --to spore --no-video
```
