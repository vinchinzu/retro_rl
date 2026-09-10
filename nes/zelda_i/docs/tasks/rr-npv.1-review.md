# rr-npv.1 review — Clean L3 Entrance→TF dest hops

Reviewer sitting. Scope: `nes/zelda_i/level3/**`, `tests/test_level3_*.py`,
`scripts/run_level3_*.py`, `docs/LEVEL3_ROUTE.md`, `docs/tasks/rr-npv.1-residual.md`.
No source edits. No commit. ROM eval not re-run (fixture-live claims taken
from the residual, checked against the code).

## Summary

**Do not close the bead. Do not STATUS.** Acceptance (L3 enter pin,
`--no-infinite-life`, no bomb/key pokes, TF `0x04`, deaths 0, hearts never 0,
glance leftover, jitter 0/30/90/223f) is not met. Residual is honest: green
through `bomb_5b` → play `0x5c`; blocked on wooden-sword `0x5c` Darknut clear
(three serial Clean deaths).

What landed and holds:

- Clean composer is `level3_entrance_tf_stages` → dest_6b +
  `level3_boss_suffix_stages`. `route_eligible=False` on the runner.
- `poke_bombs` is **not** on the Clean path. `make_l3_bomb_5b()` pause-selects
  `B_SLOT_BOMBS` and fails closed at bombs=0 (`test_bomb_5b_zero_bombs_fails_without_poke`).
- Door dest hops are leftover-relative `DoorHopSpec` rows via
  `L3DoorHopController` (rod gate skipped in-tree; Wave 0 `door_hop.py` not
  edited by this lane).
- `path_to_5d` no longer contains `push_dir(` or the 60/40/110 `idle(...)`
  holds. Clean suffix does not use `idle(env, assist, total, n)` as a hop.
- `boss_path.py` is **839 LOC** (HEAD 677; diff +568/−406 = 974 lines). Under
  the ~1000 soft max. Packed into the existing Composer; no sibling extract.
- 36 unit tests in `tests/test_level3_*.py` — **36 passed** (0.24s).
- `STATUS.md` untouched. Residual says do not STATUS / do not close.

What does not hold:

- `ROOM_5C_SPEC` combat is dest-unsafe. `occupancy_patrol=False` plus
  `engage_distance=40` still chases north of the waist onto diamonds — the
  trial-3 death at `(153,101)`.
- Jitter 0/30/90/223f is a RAM scroll-hold unit test, not fixture-live leftover
  idle.
- Survival `path_to_5d` still calls `exit_raft_passage` (`idle(..., 120)`) and
  `_PlayWait` (idle until play). Same Composer as the Clean suffix.

Exclusive tree: this lane’s implementation writes are in `level3/**`. Supporting
tests/docs/scripts sit outside that exclusive list but do not touch
integrator-only files (`spine/survival.py`, `assist.py`, `ram.py`, `STATUS.md`,
`.beads/issues.jsonl`) or Wave 0 `dungeon/door_hop.py`.

## Issues

### Issue 1 -- Severity: bug
- File: nes/zelda_i/level3/dungeon.py:385
- Description: Clean suffix `clear_5c` is not dest-safe. `ROOM_5C_SPEC` sets
  `occupancy_patrol=False` after occupancy boxed at `(120,125)`, with a
  waist/south patrol and `contact_backstep=8`, but `_engage` still chases any
  Darknut inside `engage_distance=40` with no y floor. Residual trial 3 died
  at `(153,101)` (mode 17, hearts lo=0, bombs=7 unused) after engage walked
  north of the waist onto diamonds. `0x5b` already clips dest with
  `occupancy_bounds` ymin=109; `0x5c` dropped occupancy and did not replace it
  with diamond seed / ymin. Lane acceptance (TF `0x04`, deaths 0) cannot pass
  while this stage is on the Clean tape.
- Suggestion: Occupancy-seed `0x5c` diamond cells (miss → block → replan; no
  path → stand) with ymin≥109, or a bomb-from-waist dest policy that never
  commands UP at y≤109. Do not bump `max_frames`. Keep the bead open until a
  fixture-live `clear_5c` leftover is play `0x5c` doors R|L, deaths 0.
- Status: open

### Issue 2 -- Severity: suggestion
- File: nes/zelda_i/docs/LEVEL3_ROUTE.md:262
- Description: New “Clean fixture-live Entrance→TF” section lists the full
  dest-hop chain through Manhandla TF `0x04` and a runner command, with only
  “Not spine-green. Not Clean STATUS.” It does not say fixture-live is red
  at `clear_5c`. A later sitting can read this as a wired green route. Header
  Status line is still assisted-entry; residual is the honest record.
- Suggestion: One sentence that `rr-npv.1` is blocked on `0x5c` Darknut clear
  (see `rr-npv.1-residual.md`). Keep the runner command.
- Status: open

### Issue 3 -- Severity: suggestion
- File: nes/zelda_i/tests/test_level3_boss_path.py:155
- Description: Acceptance “jitter idle 0/30/90/223f still arrives” is covered
  only by a RAM loop: 0/30/90/223 steps of **mode 6** on a frozen snap, during
  which `DoorHopController.guard` holds RIGHT (`scroll_action`), then one play
  origin and one dest snap. That is not leftover-pose jitter, not `nes_idle`,
  and not fixture-live. `test_right_5c_leftover_relative_goal` does bind
  `door_band_goal` on-band vs south mouth — that part is real.
- Suggestion: Keep the unit bind tests. Fixture-live jitter is idle 0/30/90/223f
  on a play leftover from `bomb_5b`/`clear_5c` then dest RAM. Do not treat this
  parametrize as that evidence.
- Status: open

### Issue 4 -- Severity: suggestion
- File: nes/zelda_i/level3/boss_path.py:478
- Description: Reactive hop contract is “no `idle(n)` as a hop.” Clean suffix
  stages do not call `idle(env, assist, total, n)`. Survival `path_to_5d` still
  does, indirectly: `_PlayWait.policy` is idle-until-play (max 120f) after
  `0x5d`, and `exit_raft_passage` still `idle(..., 120)` at
  `level3/boss_combat.py:144`. `test_path_to_5d_has_no_5b_return_fight_clear`
  only asserts the three old constants `60`/`40`/`110` are absent from
  `path_to_5d` source, so it cannot see those remaining idles. Integrator-owned
  spine still drives this method.
- Suggestion: Drop `_PlayWait` in favor of DoorHop `wait_modes`. Convert
  `exit_raft_passage` leftover to dest RAM (play `0x69`) without a 120f idle.
  Assert `idle(` is absent from `path_to_5d` and its callees, or stop claiming
  the conversion is complete.
- Status: open

### Issue 5 -- Severity: suggestion
- File: nes/zelda_i/level3/spine.py:148
- Description: Acceptance “hearts never 0” is an end-state check
  (`filled_hearts == 0` or mode 17). A mid-run lo-nibble 0 that recovers (or a
  death that has already left mode 17) would not trip `ok`. Trial-3 leftover
  `health 0x70` / mode 17 is caught; a green TF tape would not prove hearts
  stayed non-zero the whole way.
- Suggestion: Sample `filled_hearts` each stage (or in `on_frame`) and fail
  closed on lo==0. Residual PNG glance remains the leave proof.
- Status: open

### Issue 6 -- Severity: suggestion
- File: nes/zelda_i/level3/boss_path.py:516
- Description: `poke_bombs` is not invoked by `level3_boss_suffix_stages` or
  `run_level3_entrance_tf`. It remains recon opt-in on
  `Level3BossPathController` and is still called from `path_to_5d` at
  `boss_path.py:679`, `700`, `713` when set. Clean and Survival share this
  file. Tests (`getattr(ctl, "poke_bombs", None) in (None, False)`) pass for
  any controller that simply lacks the attribute (BombWall, DoorHop).
- Suggestion: Leave Survival recon on the mixin; do not thread `poke_bombs`
  through dest-hop objects. Tighten Clean tests to `select_item == B_SLOT_BOMBS`
  + bombs=0 → `no_bombs` (already present for `bomb_5b`) rather than getattr.
- Status: open

### Issue 7 -- Severity: suggestion
- File: nes/zelda_i/tests/test_level3_dungeon.py:142
- Description: `test_5c_and_5d_specs_are_dest_rooms` only checks room ids,
  `occupancy_patrol is False`, and `0x2B not in ROOM_5D_SPEC.enemy_types`.
  Nothing asserts a y floor, diamond seed, or that an engage from waist at a
  Darknut at y≈101 does not command UP. The trial-3 death coordinate is
  untested.
- Suggestion: RAM-step `GenericDungeonRoomController(ROOM_5C_SPEC)` from
  leftover `(120,141)` with a Darknut at `(153,101)` and require the action is
  not UP, or require `occupancy_bounds` ymin≥109 / blocked diamond cells.
- Status: open

### Issue 8 -- Severity: nit
- File: nes/zelda_i/scripts/run_level3_complete.py:1
- Description: Lane exclusive writes are `level3/**`. New runner
  `scripts/run_level3_complete.py`, `docs/LEVEL3_ROUTE.md`,
  `docs/tasks/rr-npv.1-residual.md`, and `tests/test_level3_*.py` sit outside
  that tree. They are L3-only and do not collide with other npv lanes or
  integrator files. `zelda_i.runner.open_env` is correctly avoided (`load_state`
  import is broken); `make_env(GAME, args.from_state, GAME_DIR)` is the
  isolated pin path. `add_common_args` still advertises `--no-infinite-life`
  as “STATUS-eligible”; this runner then sets `route_eligible: False`.
- Suggestion: Keep the thin A/B CLI. Do not edit `runner.py` from this lane.
  A one-line runner docstring that STATUS-eligible is L1 only is enough.
- Status: open

### Issue 9 -- Severity: nit
- File: nes/zelda_i/level3/boss_path.py:160
- Description: `UP_4D_SPEC` (`0x4d` → `0x3d`) is built and exported but
  `level3_boss_suffix_stages` drives TF inside `Level3ManhandlaController`
  (`dungeon_align_then_push` UP). Dead DoorHop row. `boss_path.py` is 839 LOC
  after packing spawn-clear, door rows, bomb-wall factories, Manhandla, and
  Survival `path_to_5d` into the Composer — under 1000, but the next `0x5c`
  dest-combat sitting will press the bar. Standards: merge or delete, no
  sibling extract.
- Suggestion: Use `UP_4D_SPEC` after heads die, or delete the row. Keep further
  `0x5c` policy in `ROOM_5C_SPEC` / this Composer, not a new file.
- Status: open

### Issue 10 -- Severity: nit
- File: nes/zelda_i/level3/clear5b.py:206
- Description: Clean `north_chain` uses `Level3NorthChainController()` with
  default `clear_darknuts=True`, so `report()["intervention_class"]` is
  `"survival"` / `"assisted"` on a Clean tape. Top-level
  `run_level3_entrance_tf` overrides `intervention_class: "clean"`. Stage
  reports disagree. Pre-existing controller, pulled onto the Clean composer
  in this sitting.
- Suggestion: Pass a Clean flag, or let the entrance-tf runner stamp each
  stage report. Do not treat stage `intervention_class` as STATUS evidence.
- Status: open

## Checks (requested flags)

| Flag | Result |
|------|--------|
| `poke_bombs` on Clean path | **No.** Pause-select bombs only. Survival recon poke remains on `Level3BossPathController`. |
| `idle(n)` as a hop | **Not on Clean dest hops.** Survival `path_to_5d` still has `_PlayWait` + `exit_raft_passage` idle 120. DoorHop stand-on-no-path is the contract. |
| File >1000 LOC without merge/delete | **No.** `boss_path.py` 839. Diff is large; fold stayed in the Composer. |
| Exclusive-tree violations | **No shared-file writes.** Supporting tests/docs/scripts are outside `level3/**` (Issue 8). Wave 0 `door_hop.py` / integrator `STATUS.md` / `spine/survival.py` not touched by this lane. |
| STATUS claims | **None.** Residual: do not STATUS, do not close. `STATUS.md` clean. LEVEL3_ROUTE completeness overclaim is Issue 2. |

## Verdict

Worker stopped correctly. Fixture-live is green through `bomb_5b` in the
residual; Clean Entrance→TF is red at `clear_5c`. Next sitting is dest-safe
`0x5c` combat (occupancy seed or bomb-from-waist, no y≤109 chase), then
re-run the pin with `--no-infinite-life` until TF `0x04`. Do not promote,
do not close `rr-npv.1`.
