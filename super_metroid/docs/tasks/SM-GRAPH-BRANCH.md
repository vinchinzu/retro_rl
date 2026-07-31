# TASK SM-GRAPH-BRANCH: Lock Wave + Ice branch edge contracts (new test file)

## Recipe step
2 graph edge (tests only)

## Model
Luna

## Own files only
- `tests/test_k4_speed_branches.py` (**create**)

Do **not** edit `progression.py` or `tests/test_progression.py`.

## Context
- Graph: `START_TO_SPEED_GRAPH` in `progression.py`
- Existing: Varia→Speed + return chain tests live in `test_progression.py`
- Need separate lock for **Bubble→Wave** and **Business→Ice** branches

## Read first
- `progression.py` — `_K4_SPEED_EDGES` from Bubble/Wave/Ice section
- `tests/test_progression.py` — `test_k4_graph_varia_to_speed_scaffold` for style
- Caps: varia + hi_jump + morph + bombs + missiles + supers as used there

## Do (thorough)
Create `tests/test_k4_speed_branches.py` with tests that:
1. Shortest path Business (0xA7DE) → Bubble (0xACB3) edge_id sequence
2. Bubble → Wave (0xADDE) edge ids + all verification currently `unverified`
   (or whatever live graph has — assert actual, don't invent continuous)
3. Business → Ice (0xA890) edge ids + verification matrix
4. path_verification Business→Wave / Business→Ice: reachable, not all_continuous,
   blocking is first non-continuous edge
5. Optional: farm/speed hall hops present for Speed path from Bubble

Use real START_TO_SPEED_GRAPH; no production mutation.

## Do not
- Promote edges
- Touch continuous / STATUS

## Acceptance
- [ ] `uv run pytest super_metroid/tests/test_k4_speed_branches.py -q` green
- [ ] ≥4 tests
- [ ] Diff summary

## Verify commands
```bash
uv run pytest super_metroid/tests/test_k4_speed_branches.py -q
```
