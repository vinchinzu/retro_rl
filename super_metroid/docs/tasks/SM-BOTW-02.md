# TASK SM-BOTW-02: Botwoon kite refine + evidence epic (dev-only)

## Recipe step
boss pipeline (strategy refine — continuous deferred)

## Model
Luna

## Wave type
implement

## Own files only
- `combat/botwoon.py`
- `tests/test_botwoon_combat.py` (**extend**)
- optional residual note

No protocol/__init__/continuous/STATUS/kpdr.

## Context
- SM-BOTW-01 scaffolded. Harden: better range-kite defaults, timeout labels,
  evidence completeness, multi-segment body awareness if features allow.
- Dev anchor: `dev_route_anchor_botwoon.state` optional smoke only.
- Maridia natural entry **not** on continuous chain.

## Read first
- `combat/botwoon.py`, `combat/primitives.py` (`range_kite_action`)
- `tests/test_botwoon_combat.py`
- `docs/BOSS_PIPELINE.md`

## Do
1. Tune or parameterize kite bands without claiming continuous green.
2. Ensure evidence captures min/max HP, action frames, outcome labels.
3. ≥3 new unit tests (kite prefers distance band; defeated empty; timeout path).
4. Optional: document one smoke command in residual (dev state).

## Acceptance
- [ ] Tests green
- [ ] Residual non-claims + next card (wrap if missing, or natural entry)

## Verify
```bash
uv run pytest super_metroid/tests/test_botwoon_combat.py -q
```
