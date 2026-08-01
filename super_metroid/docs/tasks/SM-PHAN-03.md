# TASK SM-PHAN-03: Phantoon phase windows + evidence epic (dev-only)

## Recipe step
boss pipeline (strategy refine — continuous deferred)

## Model
Luna

## Wave type
implement

## Own files only
- `combat/phantoon.py`
- `tests/test_phantoon_combat.py` (**extend**)
- optional: `docs/tasks/SM-PHAN-03-note.md`

No continuous / STATUS / kpdr / protocol wrap changes.

## Context
- SM-PHAN-01/02 scaffolded spray + units. Next hardness: **phase-aware**
  windows (invisible / eye open / flame) using existing features if present;
  otherwise stub phase labels from HP/spritemap heuristics with tests.
- Dev entry: `dev_phantoon_entry.state` optional smoke only — not continuous.
- Natural ship entry still missing on KPDR continuous.

## Read first
- `combat/phantoon.py`, `combat/features.py` (phantoon catalog / phases)
- `combat/primitives.py`
- `tests/test_phantoon_combat.py`
- `docs/BOSS_PIPELINE.md`

## Do
1. Add phase label helper or strategy fields for ≥2 phases (e.g. `invisible`,
   `vulnerable`) driven only by RAM features already available.
2. `fight_phantoon_action` may hold fire when labeled invisible (if feature
   supports it); otherwise document residual “needs spritemap probe”.
3. Extend evidence with phase transition timestamps (nullable ok).
4. ≥4 new unit tests covering phase helper + action gating.
5. Optional emu smoke against `dev_phantoon_entry` — report only; never claim
   continuous.

## Do not
- Ship route continuous compose
- Boss RAM forge

## Acceptance
- [ ] Extended tests green
- [ ] Residual: natural WS entry still PLANNER-GATE

## Verify
```bash
uv run pytest super_metroid/tests/test_phantoon_combat.py super_metroid/tests/test_boss_pipeline.py -q
```
