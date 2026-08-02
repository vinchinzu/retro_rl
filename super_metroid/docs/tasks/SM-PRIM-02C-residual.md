## Residual — SM-PRIM-02C

### Result
GREEN

### Files changed
- `routes/controller_common.py` — **already present:** `vertical_hop(session, frames, *, reason="vertical_hop")` holds `A` only; exported in `__all__` (no churn this session)
- `tests/test_controller_common.py` — **already present:** `test_vertical_hop_advances_frames_and_forwards_reason` + export assert; 24f A-only hold (no churn this session)
- `routes/kpdr/green_hill.py` — **already present:** both call sites use `_vertical_hop(..., 24, ...)` for `ghz_pillar_vertical_jump` and `noob_bridge_vertical_jump` (no churn this session; timings unchanged)
- `docs/tasks/SM-PRIM-02C-residual.md` — this residual (created this session)

### Verify paste
```text
$ uv run pytest super_metroid/tests/test_controller_common.py -q
..............                                                           [100%]
14 passed in 0.27s
exit 0
```

Evidence (pre-existing, inspected — no retune):
- `vertical_hop` → `hold(session, frames, "A", reason=reason)`
- GHZ: `_vertical_hop(session, 24, reason="ghz_pillar_vertical_jump")`
- Noob: `_vertical_hop(session, 24, reason="noob_bridge_vertical_jump")`

### Acceptance
- [x] Primitive + tests green **or** explicit leave-raw residual
- [x] `uv run pytest super_metroid/tests/test_controller_common.py -q` green

### Residual risks
- Unit-test extract only; not pure-green / continuous evidence for GHZ or Noob Bridge.
- Other routes may still inline A-only holds; out of card scope (only two green_hill sites named).
- Continuous re-record / STATUS remain planner gates.

### Next action (required)
- **Next card ID:** none
- **One change:** Card complete — both named 24f vertical A-only sites migrated; no leave-raw debt; planner may open next primitive promote elsewhere.
- **Source state:** n/a (unit primitive extract; no pure/geometry probe)

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence
- Did not retune hop frames (24f preserved)
- Did not edit continuous.py, progression.py, catalog.py, business_climb, kraid_return, varia_return, or unrelated cards
- Did not manufacture code churn on already-fulfilled implementation paths

### Probe pin (if pure/geometry) — **mandatory metrics**
room=n/a pose=n/a x=n/a y=n/a door_transition=n/a
frames=n/a dwell=n/a last_pin=n/a
# unit-test primitive promote; no pure probe on this card
