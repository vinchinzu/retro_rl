# Residual — rr-a3s hygiene (probe scripts + one z3-json-data tree)

Planner owns `docs/STATUS.md`. This sitting did not STATUS-promote.

## What existed

- **Probe scripts:** `scripts/probe_courtyard_main_door.py` (~981) and
  `scripts/probe_secret_entrance_exit.py` (~721) were exploration debris
  after constants landed in `opening_route/anchors.py` +
  `maps/screen_1b_courtyard.json`. Already deleted in `9c49a870`
  (`hygiene: drop spent one-off probe scripts`). No leftover 700+ LOC
  probe `.py` under `snes/alttp/` (largest remaining `opening_route/escape_graph.py`
  is graph, not a probe). Scratch `scratch/probe_zelda_pins.py` (166) belongs
  to the Zelda-cell bead; left alone. Recordings JSON/PNG under
  `recordings/probe_courtyard_door/` and `recordings/probe_secret_exit/` kept.
- **Dual trees:** committed wholesale copy `snes/alttp/z3-json-data/` (44 JSON
  files, no nested git) vs gitlink `snes/alttp/refs/z3-json-data` at pin
  `1eb7a785bda0d671136316c24f223c7ce12257e6`. Six dungeon JSON files had
  diverged (workspace newer: extra desert/eastern/mire/pod/swamp/thieves
  nodes/strats). Root `.gitignore` still used pre-`snes/` paths
  (`alttp/z3-json-data/`, `alttp/refs/z3-json-data/`), so the workspace
  tree was trackable.

## What this sitting did

- Authority: **`snes/alttp/refs/z3-json-data/`** (setup-script pin / gitlink).
  Did not touch the gitlink, `.gitmodules` (absent), or the three deleted
  PNGs inside the nested repo.
- Removed the committed duplicate: `git rm -r snes/alttp/z3-json-data`
  (44 files). Loader fallback in `z3_json_data.default_data_root` still
  accepts a local workspace copy if someone clones one; it is no longer
  on disk and must not be re-committed.
- `.gitignore`: added `snes/alttp/z3-json-data/` and
  `snes/alttp/refs/z3-json-data/` next to the stale `alttp/` lines.
  The refs gitlink stays tracked (gitignore does not untrack it).
- `docs/Z3_JSON_DATA.md`: path table now `snes/alttp/…`; workspace is
  gitignored optional fallback, not a second committed tree.
- Tests: no path tweak (`test_z3_json_data.py` uses synthetic fixtures).

## Exact next action

Commit the staged `z3-json-data/` deletions + gitignore/docs with
`.beads/issues.jsonl` when closing `rr-a3s`. Do not restore the workspace
vendor tree. Do not `git submodule` surgery on refs (no `.gitmodules`).
Optional local fallback: `uv run python alttp/scripts/setup_z3_json_data.py`
(already targets refs) or copy into `snes/alttp/z3-json-data/` (ignored).

## Non-claims

Did not edit opening-route maps or `STATUS.md`. Did not git commit.
Did not restore tektite/courtyard probe scripts. Did not vendor upstream.
