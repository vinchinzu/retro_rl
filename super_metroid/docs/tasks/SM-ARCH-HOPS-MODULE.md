# TASK SM-ARCH-HOPS-MODULE: Extract hop tables to routes/kpdr/hops.py

## Recipe step
docs | efficiency (structure)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/hops.py` (**create**)
- `routes/continuous.py` — **import rewire only** (no tip order / hop content edits)
- `routes/kpdr/__init__.py` — export if package surface needs it
- tests: import / equality smoke only if an existing arch test file fits;
  otherwise residual notes manual import check
- optional residual: `docs/tasks/SM-ARCH-HOPS-MODULE-residual.md`

## Context (minimal)
- Hop / tip tables currently live in `routes/continuous.py` (`_*_HOPS`,
  `POST_SUPERS_TIP_SPECS`)
- Debt row: `SM-ARCH-HOPS-MODULE` / related `SM-ARCH-TIP-SPEC`
- Planner-serial arch work — **no tip claims**, no STATUS
- Wave: `docs/tasks/WAVE-11.md`

## Read first
- `routes/continuous.py` hop table section (read; extract only)
- `docs/ARCHITECTURE.md` structure debt
- `routes/catalog.py` / `routes/segment.py` hop types (read)

## Do
1. Move hop / tip table **data** into a clean `routes/kpdr/hops.py` module
   (or thin re-export module if types must stay next to continuous for now —
   prefer real extract).
2. Keep `continuous.py` importing the new module with **no behavior change**
   (same tip names, same hop order, same frame semantics).
3. Residual with any follow-up typing work (`SM-ARCH-GRAPH-API` / `SM-ARCH-TIP-SPEC`).

## Do not
- Change any tip ordering or continuous behavior
- Edit `STATUS.md` / progression graph verification
- Stack geometry knobs in the same session

## Acceptance
- [ ] `hops.py` exists; continuous imports from it
- [ ] No intentional behavior change (import / unit smoke green)
- [ ] Residual next card ID + one change
- [ ] Non-claims: not a tip promotion

## Verify commands
```bash
uv run python -c "from super_metroid.routes import continuous; from super_metroid.routes.kpdr import hops; print('ok', len(getattr(continuous,'POST_SUPERS_TIP_SPECS',())))"
# prefer any existing continuous/arch tests if present:
uv run pytest super_metroid/tests/ -q -k "continuous or hop or tip_spec" --maxfail=5
```

## Done when
Residual filed. Planner re-records continuous only if a behavior diff appears
(should not).
