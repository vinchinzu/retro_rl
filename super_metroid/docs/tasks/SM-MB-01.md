# TASK SM-MB-01: Mother Brain BossStrategy scaffold (multi-phase epic)

## Recipe step
boss pipeline (strategy shell + unit tests — continuous deferred)

## Model
Luna

## Wave type
implement

## Own files only
- `combat/mother_brain.py` (**create**)
- `tests/test_mother_brain_combat.py` (**create**)

**Do not edit** `combat/protocol.py` / `combat/__init__.py`.
Do **not** edit `scripts/probe/mother_brain.py` or `dev/mother_brain_dev.py`
unless a tiny pure helper import is required (prefer zero edits outside Own).

## Context
- Catalog: `mother_brain_catalog()` room `0xDD58`, multi-phase, continuous deferred.
- Existing probe surface: `scripts/probe/mother_brain.py` +
  `dev_mother_brain_entry.state` / `dev_route_anchor_mother_brain.state`
  (optional smoke only).
- Harder than single-phase bosses: design **phase labels** in strategy/evidence
  even if fight_action is still spray-first.
- Escape closeout is **out of scope** (see SM-ESCAPE-01).

## Read first
- `combat/features.py` (`mother_brain_catalog`, phases)
- `combat/phantoon.py`, `combat/kraid.py` (evidence patterns)
- `combat/primitives.py`
- `tests/test_phantoon_combat.py`, `tests/test_kraid_combat.py`
- `docs/BOSS_PIPELINE.md`

## Do
1. Create `combat/mother_brain.py`:
   - `ROOM_MOTHER_BRAIN = 0xDD58`
   - `MotherBrainStrategy` (phase-aware tunables: fire period, max frames,
     weapon missiles/supers; optional phase thresholds as constants)
   - `MotherBrainEvidence` + `to_dict` (phase timeline fields ok as None stubs)
   - `fight_mother_brain_action(...)` pure; empty when defeated / event set
   - `play_mother_brain_fight(session)` bounded loop
   - developmentOnly docstring; no rainbow-beam / hyper special-case required
     for first scaffold (document as residual if omitted)
2. ≥5 unit tests (no emu required): catalog, active spray, defeated empty,
   strategy period, evidence keys.
3. Residual: natural entry + escape = PLANNER-GATE / SM-ESCAPE-01.

## Do not
- Forge boss bits / escape timer for green
- continuous / STATUS
- Claim Tourian continuous

## Acceptance
- [ ] Tests green
- [ ] Residual PROCESS schema + next card
- [ ] Non-claims explicit

## Verify
```bash
uv run pytest super_metroid/tests/test_mother_brain_combat.py -q
```
