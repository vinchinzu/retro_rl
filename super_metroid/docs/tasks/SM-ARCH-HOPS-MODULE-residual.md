## Residual — SM-ARCH-HOPS-MODULE

### Result
GREEN

### Files changed
- `super_metroid/routes/kpdr/hops.py` — **create**; `RouteHop`, hop tuples (`_*_HOPS` + public aliases), `PostSupersTipSpec`, `POST_SUPERS_TIP_SPECS` / `POST_SUPERS_TIP_BY_ID`
- `super_metroid/routes/continuous.py` — import rewire only: re-export hop/tip tables from `kpdr.hops`; keep `play_hops` + post-Supers runners
- `super_metroid/docs/tasks/SM-ARCH-HOPS-MODULE-residual.md` — this residual

### Verify paste
```bash
uv run python -c "from super_metroid.routes import continuous; from super_metroid.routes.kpdr import hops; print('ok', len(getattr(continuous,'POST_SUPERS_TIP_SPECS',())))"
# ok 9
# exit 0

uv run python -c "
from super_metroid.routes.kpdr import hops
from super_metroid.routes import continuous
assert continuous.POST_SUPERS_TIP_SPECS is hops.POST_SUPERS_TIP_SPECS
assert continuous._WAREHOUSE_HOPS is hops._WAREHOUSE_HOPS
assert continuous.RouteHop is hops.RouteHop
print('equality smoke ok')
"
# equality smoke ok
# exit 0

uv run pytest super_metroid/tests/ -q -k "continuous or hop or tip_spec" --maxfail=5
# ...........................                                              [100%]
# 27 passed, 296 deselected in 2.55s
# exit 0
```

### Acceptance
- [x] `hops.py` exists; continuous imports from it
- [x] No intentional behavior change (import / unit smoke green; same tip ids, hop order, shared objects)
- [x] Residual next card ID + one change
- [x] Non-claims: not a tip promotion

### Residual risks
- Docs still say “append RouteHop in continuous.py” (`scaffold_tip.py`, ARCHITECTURE tip recipe, segment module docstring) — stale location only; runtime is hops.py.
- `SM-ARCH-TIP-SPEC` backlog row still open until planner marks hop-extract closed / multi-registry tip wire addressed.
- `RouteHop` / `PostSupersTipSpec` types live in hops.py; continuous re-exports for API stability — further typing polish is `SM-ARCH-GRAPH-API` / tip-spec docs, not geometry.
- Private hop names (`_WAREHOUSE_HOPS`, …) re-exported on continuous for existing tests; prefer importing from `routes.kpdr.hops` in new code.

### Next action (required)
- **Next card ID:** SM-ARCH-TIP-SPEC
- **One change:** Update ARCHITECTURE / scaffold checklist strings so new tips append rows in `routes/kpdr/hops.py` (not continuous hop tables); mark hop-extract debt closed after docs sync.
- **Source state:** n/a (structure extract; no pure/geometry)

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence
- Did not change tip order, hop content, frame semantics, geometry, combat, or probe paths
- Did not re-record continuous

### Probe pin (if pure/geometry) — **mandatory metrics**
room=n/a pose=n/a x=n/a y=n/a door_transition=n/a
frames=n/a
dwell=n/a
last_pin=n/a (arch extract card; no pure run)
