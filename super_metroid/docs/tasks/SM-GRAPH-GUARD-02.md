# TASK SM-GRAPH-GUARD-02: Reverse pure edge unit locks (anti-inflation)

## Recipe step
docs / unit locks

## Model
Luna

## Wave type
implement

## Own files only
- `tests/test_progression.py` (**extend**)
- optional: `tests/test_graph_guard_reverse.py` (**create** if cleaner)

No `progression.py` verification **promotions**. No STATUS. No continuous.

## Context
- Wave 7: K3.3–K3.5 `controller_dev`; K3.6 kihunter→zeela still RED/unverified.
- Force-pass ban: pure scaffolds / units must not mark reverse path continuous.
- Prior SM-GRAPH-GUARD locked kraid_to_eye unverified patterns — extend to full
  reverse chain honesty.

## Read first
- `tests/test_progression.py`
- `progression.py` (read edge verification strings only)
- `docs/tasks/PROCESS.md` force-pass ban
- `docs/routes/KPDR_TRACKER.md` K3.3–K3.7 rows

## Do
1. Add unit tests that assert:
   - `kihunter_to_zeela_return` is **not** `continuous` / not falsely green
   - `zeela_to_warehouse_return` not continuous
   - reverse path aggregate not `all_continuous` while any reverse edge open
   - existing controller_dev edges (kraid_to_eye / eye_to_baby / baby_to_kihunter)
     remain `controller_dev` (not continuous)
2. Do not change production verification strings unless a test proves a **bug**
   (false continuous) — then residual for planner.

## Acceptance
- [ ] New locks green
- [ ] Residual lists still-unverified reverse edges

## Verify
```bash
uv run pytest super_metroid/tests/test_progression.py -q
```
