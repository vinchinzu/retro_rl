# TASK SM-CROC-01: Crocomire BossStrategy scaffold (acid-push epic)

## Recipe step
boss pipeline (strategy shell + unit tests — continuous deferred)

## Model
Luna

## Wave type
implement

## Own files only
- `combat/crocomire.py` (**create**)
- `tests/test_crocomire_combat.py` (**create**)

No protocol/__init__ edits. No continuous / STATUS / kpdr.

## Context
- Catalog: `crocomire_catalog()` room `0xA98D`, **max_hp=0** — win is
  acid-push / boss bit, not HP zero alone. Notes say push into acid wall.
- KPDR default **skips** Croc; still need catalog completeness + practice
  harness for dual-track / side path.
- continuous_status already `"side"`.

## Read first
- `combat/features.py` (`crocomire_catalog`)
- `combat/phantoon.py` (shape)
- `combat/primitives.py`
- `tests/test_botwoon_combat.py`
- `docs/BOSS_PIPELINE.md`

## Do
1. `combat/crocomire.py`:
   - `ROOM_CROCOMIRE = 0xA98D`
   - Strategy: push-oriented (hold direction toward acid + periodic fire)
   - Evidence: track boss bit / outcome labels (`pushed`, `timeout`, …)
   - Action pure function: prefer RIGHT/LEFT push + fire; empty when boss bit set
   - Docstring: developmentOnly / side path; not KPDR continuous spine
2. Unit tests: catalog HP0 note, push action non-empty while active,
   boss-bit defeated → empty, evidence keys.
3. Residual: natural Croc entry = open / not continuous gate.

## Acceptance
- [ ] Tests green
- [ ] Residual non-claims

## Verify
```bash
uv run pytest super_metroid/tests/test_crocomire_combat.py -q
```
