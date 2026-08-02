# Residual — SM-BOSS-PRIM-LANE

### Result
GREEN

### Files changed
- `super_metroid/combat/primitives.py` — added session-level `lane_hold_window` (position band + hold duration + optional `settle_standing` recovery); exported in `__all__` with BossStrategy call-signature docstring
- `super_metroid/tests/test_combat_primitives.py` — pure unit tests (FakeSession) for hold advance, in-band face, skip recovery, mid-air recovery settle
- `super_metroid/docs/tasks/SM-BOSS-PRIM-LANE-residual.md` — this residual

### Verify paste
```text
$ uv run pytest super_metroid/tests/ -q -k "primitive or lane" --maxfail=5
.........                                                                [100%]
9 passed, 306 deselected in 2.57s
```

### Acceptance
- [x] Unit green
- [x] Helper exported / documented (`lane_hold_window` in `primitives.__all__` + docstring)
- [x] Residual with next primitive ID + one change
- [x] No continuous claim

### Residual risks
- No boss strategy wired to `lane_hold_window` yet (Kraid continuous path untouched by design)
- `combat/__init__.py` re-export not updated (out of own-files scope; import from `combat.primitives`)
- Not continuous or emulator-fight evidence

### Next action (required)
- **Next card ID:** SM-BOSS-PRIM-SPRAY
- **One change:** Add session-level `spray_window` (periodic fire+jump for N frames then optional settle recovery), mirroring `lane_hold_window` on top of one-frame `spray_action`
- **Source state:** N/A — unit/primitive only (no pure geometry source)

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence
- Did not edit `continuous.py`, Kraid fight loop, `kraid.py`, or KPDR spine controllers

### Probe pin
N/A — combat primitive extract only (no pure geometry probe)
