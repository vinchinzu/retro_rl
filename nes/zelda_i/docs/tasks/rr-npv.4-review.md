# rr-npv.4 review — Clean L8 leftover-relative RoomHop

Mode: local exclusive-tree. Scope: `nes/zelda_i/level8/**`,
`tests/test_level8_*.py`, `docs/LEVEL8_ROUTE.md`,
`scripts/level8_clear_lab.py`, `docs/tasks/rr-npv.4-residual.md`.
Source not edited. Not committed.

## Summary

Partial accept. Fixture-live leftover-relative wrapping landed in
`level8/**` only; `level8/overworld.py` was not grown; integrator files
(`spine/survival.py`, `dungeon/hop_controller.py`) were not touched by
this lane. `--clean` / `allow_pokes=False` now passes an empty retopup
set from `continue_level8_spine` (Wave 0 `_run_stages` already no-ops
the poke). Gleeok heart dest *can* read slot 19 (`$83/$97`) when
`bind_env` ran. Acceptance (TF `0x80`, MK earned, deaths 0) is honestly
red: first ROM trial died in 0x5E darknuts.

The leftover-relative claim is overstated for most `RoomHopSpec`
cardinals. Gold-standard `DoorHopController` binds
`door_band_goal(leftover)` **once**. L8 one-frame `step` functions
recompute from the **current** pose every frame. For west/south/east and
north-column UP, dest_x policy was already "align to the door mouth when
`abs(x-door)>4`"; wrapping `door_band_goal` does not change off-column
knockback there. The one real door-policy change is `north_3c_step`,
which dropped UP-inland and now x-aligns along leftover y (south band
181, open bomb hole to 0x4C). Heart dest still falls back to frozen
`(32,192)`.

## Issues

### Issue 1 -- Severity: suggestion
- File: nes/zelda_i/level8/gleeok.py:60
- Description: `heart_xy` still returns frozen `HEART_XY = (32, 192)`
  when `_env` is unset or slot 19 reads `(0, 0)`. The docstring says
  "Never a frozen spawn." Pre-change tests
  `test_body_gone_walks_sw_heart_not_stand` and
  `test_f3_stand_walks_south_onto_heart` still green by walking toward
  `(32, 192)` without `bind_env`. The new RAM test only covers the bound
  path. ROM `run_controller_stage` does call `bind_env`, so the spine
  path should hit slot 19 — but the freeze is still the default dest.
- Suggestion: Fail closed (or idle) when ram/slot 19 is missing instead
  of returning `HEART_XY`. Drop "Never a frozen spawn." Add a unit test
  that unbound / `(0,0)` slot 19 does not walk the freeze.
- Status: open

### Issue 2 -- Severity: suggestion
- File: nes/zelda_i/level8/triforce.py:91
- Description: `north_3c_step` is the only door hop whose knockback
  path actually changed. HEAD UPd inland while `y > NORTH_BAND_Y` (141)
  then x-aligned; now `door_band_goal("UP", current, NORTH_DOOR)`
  x-aligns first. From the measured leftover `(32, 181)` that is RIGHT
  along south-band y=181 toward x=120, with the south bomb hole to 0x4C
  already open (`doors=12`). `RAM_CLAIM` (line 83) still says "UP inland
  (do not exit south), x-align 120". T2/T3 2/2 was inland-first. Fail
  closed on 0x4C exists, but the hop is unit-green only and not
  ROM-proven past 0x5E.
- Suggestion: Either restore inland-first until x-align altitude is off
  the south band (e.g. still UP while `y > NORTH_BAND_Y` and x is
  off-column only if y is already inland), or rewrite `RAM_CLAIM` and
  ROM-prove `(32, 181)` → 0x2C with the new x-first path.
- Status: open

### Issue 3 -- Severity: suggestion
- File: nes/zelda_i/level8/path.py:108
- Description: Knockback leftover is not leftover-relative in the Gohma
  / `DoorHopController._bind_goal` sense. `west_1f_step`,
  `_south_door_step`, `east_3e_step`, and `_north_door` call
  `door_band_goal` on the **current** `(link_x, link_y)` every frame.
  Dest is never latched from hop-start leftover. For LEFT/RIGHT,
  `door_band_goal` dest_y is a no-op vs aligning to the door mouth when
  off-band (same `abs(y-door_y)>4` branch). For UP/DOWN, dest_x is the
  same no-op vs `NORTH_DOOR`/`SOUTH_DOOR` x=120: off-column already
  aligned to 120 on HEAD; on-column already kept leftover x when
  `abs(x-120)<=4`. The west knockback test at `(208, 157)` hits
  `x > STAIRS_WEST_X` and LEFTs for stairs-clear — `door_band_goal` is
  not consulted. `test_on_column_leftover_keeps_leftover_x` (x=118)
  would pass on HEAD south. Wrapping the helper is documentation, not a
  dest-policy change, except the y halt (`NORTH_HALT_Y=109` /
  `SOUTH_BAND_Y=181` vs door plane 93/205).
- Suggestion: Bind `door_band_goal` once from the first play leftover
  (mirror `_bind_goal`) if the contract is leftover dest; or stop
  claiming leftover-relative for hops whose dest_x already matched HEAD.
  Add a west test that would fail without `door_band_goal` (on-band
  leftover y after `x <= STAIRS_WEST_X`).
- Status: open

### Issue 4 -- Severity: nit
- File: nes/zelda_i/docs/LEVEL8_ROUTE.md:12
- Description: Status paragraph still leads with Survival spine-green
  power-on, then inserts leftover-relative / `--clean` empty retopup as
  if those were part of that green. Scaffold table still lists
  `level8/gleeok.py` heart `(32,192)`. The page also still says "Nothing
  on this page is a Clean claim." Residual is honest (0x5E death); the
  route doc is not.
- Suggestion: Keep leftover-relative + Clean retopup as a fixture-live
  note, not a spine-green bullet. Point at `rr-npv.4-residual.md`. Update
  the gleeok scaffold row to slot 19 + fallback.
- Status: open

### Issue 5 -- Severity: nit
- File: nes/zelda_i/tests/test_level8_spine_wiring.py:303
- Description: `test_clean_clears_l8_bomb_key_retopup` was added to a
  file the residual says does not collect (L5 `RamWaitHop.pred` required
  field via `zelda_i.spine.survival` import). The live pin is the copy
  in `test_level8_suffix.py:263`. Dead duplicate.
- Suggestion: Keep the suffix copy; do not add more assertions to the
  uncollectable wiring file until L5 collection is fixed (other lane).
- Status: open

### Issue 6 -- Severity: nit
- File: nes/zelda_i/level8/triforce.py:60
- Description: `NORTH_BAND_Y = 141` is unused by `north_3c_step` after
  the wrap (`door_band_goal` halt is 109). Comment above it still
  describes the old "align on the door-row, then UP the center aisle"
  path. Test `test_north_band_aligns_x_to_door` only pins the constant.
- Suggestion: Use `NORTH_BAND_Y` as the inland x-align altitude, or
  delete the dead constant and the comment that describes the old path.
- Status: open

## Flag checklist (review prompt)

| Flag | Result |
|------|--------|
| Frozen spawn x still used | Yes — `HEART_XY` fallback (Issue 1). Door mouths `(120, 93)` etc. are geometry, not spawn freeze. |
| Retopup still on Clean | No — `continue_level8_spine` passes `frozenset()` when `allow_pokes=False`; `_run_stages` also gates. Constant `SPINE_L8_RETOPUP` stays non-empty for Survival. |
| Grew `overworld.py` | No — `level8/overworld.py` unmodified. Dirty `overworld/{heart_farm,path,stitch}.py` is other-lane (rr-npv.6). |
| Exclusive-tree violations | No for this lane. Writes are `level8/{gleeok,north_column,path,spine,triforce}.py` plus L8 tests/docs/lab. |
| Knockback leftover not leftover-relative | Yes for west/south/east/north-column (Issue 3). Real policy change only on `north_3c_step` (Issue 2). |

## What is fine

- Exclusive tree held; no `survival.py` / `door_hop.py` / OW-farm edits
  from this worker.
- `SPINE_L8_RETOPUP` off Clean is unit-pinned in `test_level8_suffix.py`.
- Gleeok slot 19 addressing is correct (`ADDR_LINK_X+19=$83`,
  `ADDR_LINK_Y+19=$97`).
- Lab `--from-enter --clean` disables infinite life, does not attach
  retopup, counts mode-17 as a death. Residual first-red 0x5E is
  consistent with `hc=3` and no refill.
- `route_eligible=False` / no STATUS / bead left open.
- North-column overshoot still holds UP past y=93 (y=87 push test).
- Dest room is still RAM (`$EB`); fail rooms 0x0F / 0x3C / 0x4C unchanged.

## Verdict

Do not promote. Do not close `rr-npv.4`. Leftover-relative is a helper
wrap plus one real `0x3C` north-shutter path change and a Gleeok heart
RAM read with a freeze fallback. Clean Entrance→TF is red at 0x5E as
disclosed. Next sitting: heart-safe 0x5E combat (or Survival-assist
combat only, not this Clean STATUS), and either bind leftover dest or
stop claiming leftover-relative on hops whose dest_x already matched
HEAD.
