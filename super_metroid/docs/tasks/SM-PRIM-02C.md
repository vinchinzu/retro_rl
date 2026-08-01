# TASK SM-PRIM-02C: Vertical hop primitive (or document leave-raw)

## Recipe step
primitive promote / audit

## Model
Luna

## Wave type
implement

## Own files only
- `routes/controller_common.py` (**only if** adding a named vertical hop)
- `tests/test_controller_common.py`
- optional: migrate **one** call site in `routes/kpdr/green_hill.py`
- optional residual: `docs/tasks/SM-PRIM-02C-residual.md`

## Context
- SM-PRIM-02B PARTIAL: `green_hill.py` has two **vertical A-only** 24f jumps
  (`ghz_pillar_vertical_jump`, `noob_bridge_vertical_jump`) that do not match
  directional `short_hop`.
- Planner decision for this card: either (A) add
  `vertical_hop(frames=24)` (or similar) + tests + migrate both call sites
  **without** retuning frames, or (B) residual documenting leave-raw with
  next card `none`.

## Do
1. Prefer option A if it is a clean one-primitive extract with unit tests.
2. Do not change hop timings (24f). Do not touch business_climb / continuous /
   STATUS / kraid_return / varia_return.
3. Residual with PROCESS schema.

## Acceptance
- [ ] Primitive + tests green **or** explicit leave-raw residual
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py -q` green

## Verify
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
```
