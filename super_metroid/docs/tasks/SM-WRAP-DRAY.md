# TASK SM-WRAP-DRAY: Export wrap for Draygon strategy (Wave-3 residual)

## Recipe step
boss pipeline (registration only — continuous deferred)

## Model
Luna (or Flash if tiny)

## Own files only
- `combat/protocol.py` (**add `wrap_draygon_as_boss_strategy` only**)
- `combat/__init__.py` (**Draygon wrap export only**)
- `tests/test_draygon_combat.py` (**extend**)
- optional: `tests/test_boss_pipeline.py` if needed for registry list

Do **not** edit continuous, STATUS, kpdr, botwoon/phantoon beyond imports if required.

## Context
SM-DRAY-01 shipped module + unit tests **without** wrap by design. Residual
was wrap + `__init__` export. Mirror `wrap_botwoon_as_boss_strategy` /
`wrap_phantoon_as_boss_strategy`.

## Read first
- `combat/draygon.py`
- `combat/botwoon.py` + `protocol.wrap_botwoon_as_boss_strategy`
- `combat/protocol.py`
- `combat/__init__.py`
- `tests/test_draygon_combat.py` / `tests/test_botwoon_combat.py`

## Do
1. Add `wrap_draygon_as_boss_strategy` matching sibling wrap signature.
2. Export from `combat/__init__.py`.
3. Unit tests: wrap returns BossStrategy-compatible object; call is pure
   (no emu); catalog room matches Draygon.
4. Docstring: developmentOnly / not continuous until natural Maridia entry.

## Do not
- Claim Maridia natural entry
- continuous / STATUS
- Expand fight logic beyond wrap

## Acceptance
- [ ] Wrap + export + tests green
- [ ] No continuous claims

## Verify commands
```bash
uv run pytest super_metroid/tests/test_draygon_combat.py super_metroid/tests/test_boss_pipeline.py -q
```
