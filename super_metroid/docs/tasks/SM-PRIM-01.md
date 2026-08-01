# TASK SM-PRIM-01: Extract `settle_hold` primitive from business climb 12f baseline

## Recipe step
1 pure controller (primitive promote) — **no continuous**

## Model
Luna

## Own files only
- `routes/controller_common.py`
- `tests/test_controller_common.py`
- optional: `docs/tasks/SM-PRIM-01-residual.md`

**Do not** edit `business_climb.py` in this card (call-site migration is
SM-PRIM-01B). Do not touch continuous / STATUS.

## Context
- Wave-5 climb baseline: platform settles **12f** pure-green with setup
  `LEFT,LEFT,RIGHT` (see QUEUE / SM-TIGHTEN-01D).
- Process: `docs/tasks/PROCESS.md` §2 — promote after pure + continuous
  survival. Climb pure+continuous already green once; extract the settle
  helper so future cards reuse it.
- Source catalog: `docs/SOURCE_STATES.md` (`business_climb_entry`).

## Read first
- `routes/controller_common.py` (existing `hold`, `hold_until`, `wait_ordinary_room`)
- `routes/kpdr/business_climb.py` (pattern: `_hold(session, 12, reason="…_settle")`)
- `tests/test_controller_common.py`
- `docs/tasks/PROCESS.md` §2

## Do
1. Add a small helper, e.g. `settle_hold(session, frames=12, reason="settle")`
   that is a thin named wrapper around `hold` (documents intent; optional
   standing/vy=0 poll later — **not** this card).
2. Unit-test: N frames advanced, reason string preserved (fake session).
3. Do **not** rewrite climb call sites here (one-knob: extract only).
4. Residual: propose **SM-PRIM-01B** to replace eight business 12f settles
   with the helper (serialize on `business_climb`).

## Do not
- Change climb setup jumps or run-up constants
- Add full-bank WRAM copies
- Claim continuous savings

## Acceptance
- [ ] Helper exported from `controller_common`
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py -q` green
- [ ] Residual names SM-PRIM-01B + one change
- [ ] No business_climb / continuous / STATUS edits

## Verify commands
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
```

## Done when
Luna returns residual with next card SM-PRIM-01B (call-site migrate only).
