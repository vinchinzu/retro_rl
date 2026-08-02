# TASK SM-BOSS-PRIM-LANE: Combat primitive – lane-hold windows

## Recipe step
primitive promote

## Model
Flash

## Wave type
implement

## Own files only
- `combat/primitives.py`
- `tests/` unit covering the new helper (create or extend existing combat
  primitive tests)
- optional residual: `docs/tasks/SM-BOSS-PRIM-LANE-residual.md`

## Context (minimal)
- Shared before late bosses (Phantoon+)
- Kraid is living continuous template — **do not** touch Kraid continuous path
- `lane_hold_action` already exists (one-frame). This card adds a **window**
  helper: position band + duration + recovery settle
- Docs: `docs/BOSS_PIPELINE.md`, `docs/research/STRUCTURED_BOSS_RL.md`

## Read first
- `combat/primitives.py` (`lane_hold_action`, `settle_standing`, session helpers)
- `combat/kraid.py` (lane usage pattern — read only)
- existing `tests/test_*combat*` or primitive tests
- `docs/BOSS_PIPELINE.md` shared primitives section

## Do
1. Add a reusable **lane-hold window** helper (position + duration + recovery),
   e.g. session-level: hold lane for N frames then settle recovery M frames.
2. Unit-test against a synthetic or existing boss fixture (no full fight).
3. Document call signature in the helper docstring for future `BossStrategy` use.
4. Residual → next primitive ID (`SM-BOSS-PRIM-PHASE` or `SM-BOSS-PRIM-SPRAY`).

## Do not
- Implement any full boss fight
- Touch continuous spine, STATUS, or Kraid continuous controllers
- Edit `kraid.py` fight loop in this card

## Acceptance
- [ ] Unit green
- [ ] Helper exported / documented
- [ ] Residual with next primitive ID + one change

## Verify commands
```bash
uv run pytest super_metroid/tests/ -q -k "primitive or lane" --maxfail=5
# if a dedicated test module is created, run that path only
```

## Done when
Residual filed. No continuous claim.
