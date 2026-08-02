## Residual — SM-BOSS-UNIT-MATRIX

### Result
GREEN

### Files changed
- `super_metroid/tests/test_boss_catalog_matrix.py` — parametrized catalog matrix: room_id + boss_id lock, optional strategy-module import, soft wrap_* room match
- `super_metroid/docs/tasks/SM-BOSS-UNIT-MATRIX-residual.md` — this residual

### Verify paste
```text
$ uv run pytest super_metroid/tests/test_boss_catalog_matrix.py super_metroid/tests/test_boss_pipeline.py -q
...................s..............                                       [100%]
33 passed, 1 skipped in 0.18s
```
(exit=0; single skip = `spore_spawn` has no `wrap_*` yet)

### Acceptance
- [x] Parametrized tests over every catalog id
- [x] Every catalog entry has positive `room_id` (and key/`boss_id` match)
- [x] Existing `combat/<boss_id>.py` modules import cleanly
- [x] Soft wrap_* check (skip when absent; room match when present)
- [x] Residual lists bosses without strategy modules + next card + one change
- [x] No continuous claim / no combat geometry rewrites

### Residual risks
- **Catalog bosses without `combat/<id>.py` strategy module:** `spore_spawn` only (live fight lives in `routes/spore_spawn_controller.py`, not under `combat/`)
- **Catalog bosses without `wrap_*`:** `spore_spawn` only (matrix soft-skips)
- Escape scaffold (`combat/escape.py`) is not a `BOSS_CATALOG` entry — out of this matrix by design
- Matrix is import/unit only — not fight, pure, or continuous evidence for any deferred boss

### Next action (required)
- **Next card ID:** SM-SPORE-STRATEGY-01
- **One change:** Add thin `combat/spore_spawn.py` facade + `wrap_spore_spawn_as_boss_strategy` delegating to the existing continuous controller (no geometry rewrite) so the catalog matrix has full module + wrap coverage
- **Source state:** N/A — unit/matrix only (Spore continuous already elsewhere)

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence
- Did not edit combat strategy geometry, `continuous.py`, or STATUS
- Did not claim Spore or deferred bosses as newly continuous

### Probe pin
N/A — unit matrix only (no pure geometry probe)
