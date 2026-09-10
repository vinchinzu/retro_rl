# rr-npv.5 review — Clean L9 Patra/Ganon dodge (fixture-live)

## Summary

Lane sitting is **fixture-live leftover, not the enter-pin acceptance gate**. Core combat change is real: Patra/Ganon no longer `idle` through attack cooldown; brown Ganon holds the silver-arrow axis once `_arrow_direction` matches; door holds in `path.py` / `natural_path.py` call shared `door_band_goal` instead of a frozen spawn x. `prefix.py` is untouched (1541 LOC, no diff). `door_graph/level9_exits.py` is untouched.

Unit tests: 11 new cases in `tests/test_level9_dodge.py`; 95 `test_level9*` tests collected, 46 sampled (dodge + `test_level9.py` + join) passed. Residual ROM labs are suffix pins (`Level9FinalPatraReconFixture`, `Level9BeforeGanonReconFixture`), deaths 0, `route_eligible=false`. Residual leftover correctly says entry→Patra from the enter pin is later.

Do **not** STATUS. Do **not** close the bead. Acceptance (“from L9 enter pin … Patra north, Ganon silver-arrow kill, credits, deaths 0”) is still open.

What is landable as fixture-live: dodge + axis-commit + leftover-relative door column, with the issues below. What is not: enter-pin gate, spine-green, or treating 95 unit tests as ROM eval.

Checked flags: prefix.py not grown; `idle(n)` is gone from the combat cooldown path but replaced by a walking “face” hop (Issue 1); enter-pin is not claimed as green, though “e2e enter→credits” is easy to misread (Issue 6); `natural_path.py` 1243 LOC with no merge/delete (Issue 3); clone CLIs reintroduced outside the exclusive tree (Issue 4).

## Issues

### Issue 1 -- Severity: suggestion
- File: nes/zelda_i/level9/patra.py:107
- Description: Cooldown with no nearby eye is `nes_action("UP")` `"face_up"` for the whole cooldown, not a one-frame face. Zelda I has no face-without-move except idle, so this walks north off the south-stand (`stand_dy=30`), then `align_south_y` walks back. Gohma (the stated gold standard) faces one frame when ready to fire and **idles** the rest of cooldown (`level6/gohma.py:317-318`). The reactive-hop rule is “no path → stand”; “no `idle(n)` as a hop” is dest-nav, not “must press a button on cooldown.” Patra lab 3716f vs historical ~2189f south-stand is consistent with the extra align churn. Same pattern on Ganon sword cooldown: `ganon.py:259-262` holds `sword_dir` for 12f (walks into the boss when in sword range 24 but outside contact 12). Tests lock the walk in (`test_level9_dodge.py:110-123`).
- Suggestion: Dodge when a hazard is inside `DODGE_DIST`. Otherwise stand (idle) to keep facing. Face one frame only when `cooldown==0` and about to fire, like Gohma. Drop `test_patra_cooldown_without_eye_faces_up_not_idle` or change it to “not idle only when a hazard is in range.”
- Status: open

### Issue 2 -- Severity: suggestion
- File: nes/zelda_i/level9/prefix.py:1442
- Description: `patra_action` is also the live policy for room `0x61` (`Level9Stairs61Controller`). That room already needed a 90-frame still-position escape because 0x61 geometry can pin the south-stand (`prefix.py:1416-1441`, `_stuck_xy` exact equality). Cooldown dodge / walk-UP changes Link’s xy most frames, so `_stuck_frames` resets and the escape may never fire. Prefix.py was not grown, but its 0x61 hop behavior changed with no 0x61 ROM trial this sitting. `test_stairs_61_policy_and_factory` only checks dest/fail-closed, not combat.
- Suggestion: Re-run `L9Room61EntryReal` / silver-arrows chapter after this dodge. If still-position escape is the 0x61 safety net, key it on best-distance-to-stand (same lesson as `_wp_step`) or keep cooldown stand so the existing 90f detector still works.
- Status: open

### Issue 3 -- Severity: suggestion
- File: nes/zelda_i/level9/natural_path.py:1
- Description: File is 1243 LOC (was 1221). Soft max ~1000; crossing 1k means merge into the Composer or delete, not a sibling extract. Residual admits “no sibling extract.” The +22 lines are leftover fields + `leftover_door_step` calls (the right helper lives in `path.py`). Growing an already-over-bar file is still the size rule. `prefix.py` 1541 and `overworld.py` 1031 are pre-existing and out of this sitting’s writes.
- Suggestion: Next L9 sitting that touches this file must fold a chapter (join or ending adapters) into the hop table / existing owner until this file is under ~1000, or delete a dead controller. Do not add `natural_path_2.py`.
- Status: open

### Issue 4 -- Severity: suggestion
- File: nes/zelda_i/scripts/run_level9_patra.py:1
- Description: Exclusive writes for `rr-npv.5` are `level9/**` and `door_graph/level9_exits.py`. New untracked clone runners `scripts/run_level9_patra.py` (152 LOC) and `scripts/run_level9_ganon.py` (re-exports `main`) sit outside that tree. `docs/HYGIENE.md:62-64` and `docs/LEVEL9_ROUTE.md:549` already record that those recon CLIs were pruned; Composer binds the dests. Coding standards: delete clone runners on sight (`run_stageN_segment.py` class). `run_level9_ganon.py:5` `from run_level9_patra import main` only works because the script’s directory is `sys.path[0]`.
- Suggestion: Delete both scripts. Keep lab invocation on the existing Composer (`level9_credits_chapter` / `run_survival_spine.py`) or a note in the residual. If a lab CLI is required, it belongs under `level9/` and must not revive the pruned names. `docs/LEVEL9_ROUTE.md` is L9-specific; tests/`rr-npv.5-residual.md` are expected lane leftovers — not the same class of violation.
- Status: open

### Issue 5 -- Severity: nit
- File: nes/zelda_i/level9/ganon.py:194
- Description: `_arrow_direction` axis slop went 8→4. Comment at `ganon.py:235-236` cites “Gohma FIRE_TOL”; Gohma’s `FIRE_TOL` is **8** (`level6/gohma.py:89`). Axis-commit only runs when this predicate hits, so at `|dx|` 5–8 brown Ganon still takes the cooldown dodge and can walk off the column — the miss class the residual recorded (352 silver-arrow pulses). Trial 2 green does not make the comment or the tighter window self-explanatory.
- Suggestion: Either keep 8 to match Gohma and commit as soon as the arrow can land, or keep 4 and say why silver arrows need a tighter column than Gohma. Add a unit case: brown, `|dx|==6`, cooldown>0, fireball in range → still on-axis (or explicitly dodge, if 4 is the rule).
- Status: open

### Issue 6 -- Severity: nit
- File: nes/zelda_i/docs/tasks/rr-npv.5-residual.md:15
- Description: Acceptance is “from L9 enter pin.” Residual leftover and LEVEL9_ROUTE both say suffix pins and that entry→Patra is later — that is honest. Line 15 still says “3167f e2e enter→credits” from `Level9BeforeGanonReconFixture`; LEVEL9_ROUTE.md:29 table “credits | same | 3167 e2e” can be read as the enter-pin gate. “Enter” here is the `level9_enter_ganon` stage, not `Level9EntranceReconFixture`.
- Suggestion: Write “BeforeGanon → wait_credits 3167f” (or `--start level9_enter_ganon --through credits`). Keep the enter-pin glance in leftover. Do not let “e2e enter→credits” drift into a STATUS sentence.
- Status: open

### Issue 7 -- Severity: nit
- File: nes/zelda_i/level9/ganon.py:53
- Description: `DODGE_DIST=14` and `CONTACT_CHEBYSHEV=12` duplicate `combat.CONTACT_MANHATTAN` / `combat.CONTACT_CHEBYSHEV` (`combat.py:19-20`). Ganon fireball types are a local frozenset, which is fine; the numeric pair is not a new measured L9 constant.
- Suggestion: Import the combat contact constants (or alias) so dodge radius cannot drift from the rest of dungeon combat.
- Status: open

### Issue 8 -- Severity: nit
- File: nes/zelda_i/level9/patra.py:87
- Description: Production docstrings restate the ticket (“instead of idle(n); dest is RAM”, `ganon.py:222-223` “Never idle(n) through cooldown”). Comments should be WHY (Gohma face-then-fire; dodge-off-column missed the silver arrow — that WHY at `ganon.py:235-236` is the good one).
- Suggestion: Keep the axis-commit WHY. Drop the bead-contract slogans from `patra_action` / `ganon_action` docstrings.
- Status: open

## Not issues (checked)

- **prefix.py grown:** no. `git diff nes/zelda_i/level9/prefix.py` is empty. Behavior of `patra_action` still leaks into 0x61 (Issue 2).
- **Enter-pin gate claimed as green:** not in the residual’s leftover section. Labs are named suffix fixtures; bead stays open.
- **idle(n) as dest-hop:** `leftover_door_step` is align-then-push, not idle. Combat cooldown idle is gone (Issue 1 is the replacement, not a leftover idle hop). `wait_north_door` / `wait_ganon` idle when the object is missing — RAM wait, not a hop.
- **door_graph/level9_exits.py:** no writes.
- **STATUS / beads / survival.py:** not touched.
- **Unit tests for leftover column:** `test_leftover_door_band_off_column_uses_door_x` and `test_north_41_uses_leftover_column_not_frozen_spawn` match `door_band_goal` (in-band keeps leftover x; off-column uses door x). `SOUTH_10` still runs `room10_lane_step` before the door hold, so the statue-band trap is not reintroduced.
- **Lab path pokes:** `run_level9_patra.py` drives `level9_credits_chapter` (pause-select, no `ADDR_SELECTED_ITEM` assign). `GanonFightController`’s disclosed B-slot write is pre-existing and not on this lab path.
