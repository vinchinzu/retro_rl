# rr-npv.6 review — Clean OW stitch + heart farm

Mode: local exclusive-tree. Source not edited. Not committed.

Checked: `overworld/{rupee_farm,heart_farm,stitch,common,cave_shop}.py`,
`tests/{test_heart_farm,test_overworld_stitch,test_overworld}.py`,
`docs/OVERWORLD_DOORS.md`, `docs/tasks/rr-npv.6-residual.md`, plus the
flagged lock files `overworld/{path,graph,nav}.py`.

Independent ROM settle (`make_env` + idle `PostL*TriforceSettleController`,
no pokes): Level4Complete → OW `0x45` `(128,125)` mode 5 TF `0x0C` whistle 0
in 633f; Level5Complete → OW `0x0B` `(112,125)` mode 5 TF `0x1C` whistle 1
from pin start (not granted) in 1116f. Packed stitch TF `0x0F`/`0x1F` is the
natural cumulative, not the pin skip — that match is honest.

34 unit tests in the three named files: green.

## Summary

Do not land as-is.

Packed L4→L5 / L5→L6 leave xy bands are correct, `route_eligible` stays
false, and Whistle is not granted. The occupancy claim is not met: a true
no-move miss still probes (OccupancyWalker forgets inferred blocks, then
walks again). Exclusive-tree is broken: `overworld/path.py` carries the
rr-ps7.3 L2 `0x4C` east-mouth occupancy walk the residual said was not
edited. `graph.py` / `nav.py` are clean. Live Clean `farm_below_hearts`
remains the leftover the residual already named (unit analog only).

## Issues

### Issue 1 -- Severity: bug
- File: nes/zelda_i/overworld/path.py:562
- Description: Exclusive-tree lock is `overworld/{rupee_farm,heart_farm,stitch}.py`; `path.py` is rr-ps7.3 (L2 `0x4C` east mouth) and the residual says it was not edited. Working tree has `_occupancy_align_action` (UP/DOWN + `align_x`, leftover on the east/west edge, never RIGHT at `x≥232` because that scrolls to `0x4D`, comment names the `0x4C` corridor) plus the `_do_hop` call at path.py:641. Same sitting also dirties `tests/test_ow_path.py` (`test_east_mouth_4c_never_pushes_up_off_column`). `graph.py` and `nav.py` have no diff.
- Suggestion: Revert `path.py` / `test_ow_path.py` from this lane. Leave the L2 door occupancy on rr-ps7.3. Correct the residual line that claims no `path.py` edits.
- Status: open

### Issue 2 -- Severity: bug
- File: nes/zelda_i/overworld/heart_farm.py:197
- Description: Contract is occupancy miss → block cell → replan; no path → stand. `_grade_occupancy` does mark a true no-move (heart_farm.py:180-183) via `mark_blocked_ahead` (inferred). `_walk_to` then calls `OccupancyWalker.next_dir`, which forgets every inferred block when BFS is empty and returns a cardinal anyway. Frozen chase (prey north of Link, xy held still) never emits `occupancy_stand`: it cycles UP/LEFT/RIGHT/DOWN, `forgets` increments, blocked set resets to empty, and the same cells are probed again. `test_occupancy_no_path_stands` (test_heart_farm.py:224-230) only stands because it writes spec blocks and `inferred.clear()` — a real occupancy miss never takes that path. Docstring at heart_farm.py:190 overclaims stand.
- Suggestion: After a miss, replan with the new block still in the grid. If `shortest_path` is None, return `occupancy_stand` without going through `next_dir`'s inferred-forget. Add a frozen-xy chase test that expects stand (or a peel that is not the missed cell) and does not clear `inferred`.
- Status: open

### Issue 3 -- Severity: suggestion
- File: nes/zelda_i/overworld/heart_farm.py:199
- Description: Occupancy records `last_dir` from `next_dir`, then `walk_or_swing` may face a contact threat or dodge a projectile instead. A no-move frame then marks the occupancy cell blocked even though that cardinal was not the button this frame. False fences, then Issue 2 forgets them and probes.
- Suggestion: Grade the action actually emitted, or skip occupancy grade on a frame where `walk_or_swing` overrode the occupancy cardinal.
- Status: open

### Issue 4 -- Severity: nit
- File: nes/zelda_i/overworld/stitch.py:355
- Description: L5→L6 `MouthStitch.status` is still `"verified"` while the leave packet is `evidence="live"`, `verified=False`, `route_eligible=False`. L4→L5 uses `"live"`. Notes now say fixture-live, so the row status is the leftover that will be misread as a verified leave.
- Suggestion: Set status to `"live"` (mouth verified belongs in notes, which already say L6 mouth `0x22` is live).
- Status: open

## Checked clean

- `route_eligible`: `_pose` / `handoff_from_ram` force false; new stitch test asserts both L4 and L5 packets ineligible.
- Whistle: L5 packet copies `whistle=1` as a pose fact; ROM settle reads `ADDR_WHISTLE=1` on pin entry and 1 on leave. No write in the reviewed modules. L4 leave whistle stays 0.
- Packed xy vs pin glance: L4 `(0x45, 128, 125)±4`, L5 `(0x0B, 112, 125)±4`. Matches residual, ROM settle, and existing L4 island / L5 door geometry (`POST_L4_TO_LEVEL5_HOPS[0].align_x == 128`, L5 door x≈112).
- `rupee_farm.py`, `common.py`, `cave_shop.py`, `tests/test_overworld.py`: no diff this sitting.
- `min_filled<=0` skip is the `farm_below_hearts=0` analog and is unit-covered. Live Clean farm on `filled_hearts < 3` is still leftover (no low-heart OW pin); that is honest in the residual, not a code defect in the exclusive files.
