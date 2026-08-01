# TASK SM-PRIM-02: Extract `short_hop` hold primitive from kraid return approach

## Recipe step
1 pure controller (primitive promote) — **no continuous**

## Model
Luna

## Own files only
- `routes/controller_common.py`
- `tests/test_controller_common.py`
- optional: `docs/tasks/SM-PRIM-02-residual.md`

**Do not** edit `varia_return.py` in this card (call-site migration later).
Do not touch continuous / STATUS / door shot choreography.

## Context
- Wave-5 residual SM-K4-06C: best pin used short hop **24f** + settle **20f**
  (`kraid_return_short_hop` / `kraid_return_approach_settle`) still door RED.
- Even while door geometry is RED, the **named hop/settle pair** is a stable
  primitive worth extracting so the next one-knob card only changes shot/Y.
- Process: `docs/tasks/PROCESS.md` §2; source
  `scratch/post_varia_to_kraid_pure.state` (`docs/SOURCE_STATES.md`).

## Read first
- `routes/kpdr/varia_return.py` (`kraid_return_short_hop` / approach_settle)
- `routes/controller_common.py`
- `docs/tasks/SM-K4-06C-residual.md`
- `tests/test_controller_common.py`

## Do
1. Add helpers, e.g.:
   - `short_hop(session, direction, frames, *, buttons_extra=(), reason=...)`
   - optional `settle_hold` reuse if SM-PRIM-01 already landed; else local
     thin settle wrapper — **do not** invent new timing defaults that diverge
     from 24/20 without saying so in residual.
2. Unit tests on fake session (frame advance + buttons).
3. Residual: next card either pure door one-knob (shot band) **or**
   SM-PRIM-02B call-site migrate in `varia_return` (serialize geometry).

## Do not
- Change door shot / lip / fuse timings
- Free-spin or PLM forges
- Parallel with other `varia_return` geometry cards

## Acceptance
- [ ] Helper(s) exported + tested
- [ ] pytest controller_common green
- [ ] Residual: next card ID + one change + source path
- [ ] No varia_return / continuous / STATUS edits

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
```

## Done when
Luna returns residual; planner picks door geometry card vs call-site migrate.
