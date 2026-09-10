# rr-npv.2 review — Clean L5 whistle path (fixture-live)

Reviewer sitting. Does not edit source. Does not claim STATUS.

Scope: git diff HEAD + untracked under `nes/zelda_i/level5/**`,
`tests/test_level5_*.py`, `scripts/run_level5_*.py`, `docs/LEVEL5_ROUTE.md`,
`docs/tasks/rr-npv.2-residual.md`.

## Summary

The sitting does the contracted hop conversion: `whistle_path.py` no longer
calls `_ops().idle(n)` / `push_dir(frames=N)`; waits go through
`level5.path.RamWaitHop` / `wait_ram` (RAM dest, `max_frames` budget) and
leftover-relative door columns use Wave 0 `door_band_goal`. Exclusive-tree
hygiene holds: no `STATUS.md`, no pokes, no writes outside L5-owned files,
`whistle_path.py` is 808 LOC. Unit gate is real (`22 passed`). Residual is
honest that ROM stopped at Clean Pols Voice death in `0x77` (whistle RAM
waits never ran).

Acceptance (L5 enter pin → TF `0x10`, deaths 0, no assist) is **not** met.
That is the residual's own first red, not a hidden STATUS claim.

One correctness defect on the converted whistle path: `bomb_wall` dropped the
fuse stand and immediately holds face back into the live bomb. Shared
`dungeon.bomb_wall` still waits `BOMB_N_WAIT_BLAST=100` after a step-back.
This is also a Survival L5 suffix risk (same `bomb_west_from_66/65` helpers).

L8's report that `RamWaitHop.pred` breaks `test_level8_spine_wiring.py`
collection is **not reproduced**. That file collects 13 tests; `RamWaitHop`
constructs. Both dataclasses are `kw_only=True`.

## Issues

### Issue 1 -- Severity: bug
- File: nes/zelda_i/level5/whistle_path.py:235
- Description: After `B`, the converted hop backs ~12px (`_away`, max 40f)
  then immediately `wait_ram(_play_room(dest), hold=spec.face, max_frames=600)`.
  Dest-room is false until the wall opens, so Link walks back onto a live
  bomb for the rest of the fuse. The previous chain idled 100f after the
  step-back; `dungeon.bomb_wall` still does `step_back` then
  `wait_blast=100` idle (`BOMB_N_WAIT_BLAST`) before push. Clean bomb
  damage is a death; this hop is on the contracted whistle path
  (`bomb_west_from_66` / `_65`) even though the first ROM trial died
  earlier in `0x77`.
- Suggestion: Dest is RAM, not a frame count: stand (`hold=None`) until the
  bomb object is gone or the wall tile/census changes, then hold face into
  the hole. Do not use dest-room as the fuse predicate.
- Status: open

### Issue 2 -- Severity: suggestion
- File: nes/zelda_i/level5/spine.py:226
- Description: `run_level5_whistle_suffix` (Survival `--through level5` and
  the new Clean pin loader) calls the same `bomb_wall` helpers. Survival
  credits is already green through this suffix. Infinite-life may eat the
  self-bomb, but knockback can miss the hole and timeout. The sitting did
  not ROM-eval Survival L5 after the idle(n) fold.
- Suggestion: After the fuse dest is RAM, re-run Survival `--through
  level5-whistle` (or the existing L5 TF tape compare) before treating the
  fold as spine-safe.
- Status: open

### Issue 3 -- Severity: suggestion
- File: nes/zelda_i/scripts/run_level5_whistle_tf.py:90
- Description: `run_level5_from_entrance(..., assist=None)` always. That
  matches the Clean pin, but `add_common_args` still defaults
  `--infinite-life` to True, and `track` is `"assisted"` unless `--clean`
  / `--no-infinite-life` is passed (`:154`, `:109`). Default invocation
  writes a JSON that looks assisted and a VideoTap intervention that says
  Clean. `--clean`'s help text admits the runner never assists.
- Suggestion: Drop `add_common_args`'s life flag, or actually attach
  `make_assist` when `--infinite-life` is on. Keep `track` / intervention
  in sync with whether an assist object was bound.
- Status: open

### Issue 4 -- Severity: suggestion
- File: nes/zelda_i/level5/path.py:367
- Description: L8 residual blamed `RamWaitHop.pred` (required field after
  `HopController` defaults) for `test_level8_spine_wiring.py` failing to
  collect. Against this tree that is false: `QT_QPA_PLATFORM=offscreen uv
  run pytest nes/zelda_i/tests/test_level8_spine_wiring.py --collect-only`
  collects 13 tests; `RamWaitHop(pred=lambda s: True)` constructs. Parent
  and subclass are both `@dataclass(kw_only=True)`, so the required `pred`
  is legal. It would TypeError only if `kw_only` were dropped on either
  class.
- Suggestion: No L8 fix needed from this lane. Optional hardening: default
  `pred` (`lambda snap: False`) so a future non-kw_only edit cannot break
  Survival/L8 imports of `level5.spine`.
- Status: open

### Issue 5 -- Severity: suggestion
- File: nes/zelda_i/tests/test_level5_path.py:208
- Description: `test_whistle_path_has_no_idle_n_on_clean_path` is an
  `inspect.getsource` substring check (`idle(env`, `push_dir(`). It does
  not step `bomb_wall` through fuse dest, timeout-vs-arrived on the last
  budget frame, or death-fail of `wait_ram`.
  `test_ram_wait_hop_arrives_on_dest_room_not_frame_count` only covers
  hold-LEFT room change. `test_l5_west_door_band_is_leftover_relative`
  tests Wave 0 `door_band_goal` in isolation, not that
  `key_west_to` / `cellar_other_mouth` / `exit_whistle_04` consume `gx`.
- Suggestion: Unit-step a bomb-wall fuse pred (object gone / wall open,
  `hold is None`) and assert `policy` is idle until that bit, then face
  into the dest room. Keep the source grep as a belt.
- Status: open

### Issue 6 -- Severity: nit
- File: nes/zelda_i/level5/path.py:388
- Description: `drive_hop` / `wait_ram` / the 0x04 thaw comment restate the
  campaign slogan (`Dest is RAM; no idle(n)` / `Budget, not a hop`) instead
  of a non-obvious constraint. `wait_ram`'s "not a hop" also contradicts
  `RamWaitHop`.
- Suggestion: Drop the slogans. Keep one WHY where it is real (recorder
  overhead freeze; bomb fuse is object/wall RAM, not dest-room).
- Status: open

### Issue 7 -- Severity: nit
- File: nes/zelda_i/level5/whistle_path.py:356
- Description: `lambda snap, d=direction: done(snap) or snap.screen !=
  ROOM_L5_BLUE_64` binds `d` and never uses it. Direction only comes from
  `hold=`. `take_whistle_04` also double-reads RAM for leftover
  (`_rs(...).link_x, _rs(...).link_y` at `:634`).
- Suggestion: Pred = `done or screen != 0x64`; leftover = one snapshot.
- Status: open

### Issue 8 -- Severity: nit
- File: nes/zelda_i/docs/LEVEL5_ROUTE.md:232
- Description: New Clean section pins **`Level5EntranceFromL4`**. `## Next`
  still says isolated `Level5WhistleFrom77` remains the pin. The Clean
  section also omits the sitting's ROM leftover (play `0x77` mode 17
  `(48,157)`, deaths=1).
- Suggestion: One pin name. One line that fixture-live ROM is red at Clean
  Pols Voice, whistle waits unrun.
- Status: open

## Contract check (no extra issues)

| Check | Result |
|-------|--------|
| Edits outside exclusive tree | None in this lane. Dirty L3/L7/L8/L9/OW/Wave 0 files are other sitters. |
| STATUS claims | None. Residual + LEVEL5_ROUTE say `route_eligible=false`, not Clean STATUS. |
| Pokes | None added. Pin runner sets `allow_pokes=False`, `assist=None`. |
| `idle(n)` on Clean whistle path | Gone from `whistle_path.py` (comment only). `west_path` / `cellar_path` / `boss_path` still have `idle(n)` as residual states. |
| `whistle_path.py` ~1000 LOC | 808. |
| L8 `RamWaitHop` collection break | Not reproduced (Issue 4). |
