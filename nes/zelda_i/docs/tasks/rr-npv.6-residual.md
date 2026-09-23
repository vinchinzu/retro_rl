> Historical lane note. The live sitting is the gathering prefix in [PRE_L1.md](../PRE_L1.md). This file stays because the clean-tip ladder or a route doc still cites it. It is not the current plan.

# rr-npv.6 residual — Clean OW stitch + heart farm

Fixture-live only. Do not STATUS. Do not close the bead. `route_eligible`
stays false.

## This sitting (2026-09-11) — heart economy

Clean tip is still L1→L2 door-path death on `0x5C`. Recovery was a farm
divert that chased enemies then scooped **rupees** (`0x60`). Floor hearts
and the 16-kill fairy (`$0627`) were unused. `path.py` occupancy / 0x4C
peel stays `rr-ps7.3`.

Work this sitting (parallel, no STATUS):

Live At4A probe: **every floor drop is ObjType `0x60`**. Item identity is
**ObjState**: heart `0x22`, fairy `0x23`, rupee `0x18`, 5-rupee `0x0F`,
clock `0x21`. Type `0x22` stays `ghini_flying`. `hp` 0 (flash `0x80`).

Landed:
- `combat.floor_drops` / `is_heart_or_fairy_drop` / `nearest_heart_or_fairy`
  plus `world_kill_count` / `help_drop_*` on the snapshot.
- Heart farm scoops heart/fairy (by state) before chase, then rupees.
  14–15 kill streak holds restock for the `$0627` fairy.
- Hop/nav `_rupee_scoop` scoops hearts when not full. `path.py`
  `_occupancy_align_action` / 0x4C peel untouched.
- Dungeon engine scoops a nearby heart in FIGHT (if no contact enemy)
  and COLLECT_REWARD (before the spec key).

Unit: `uv run pytest nes/zelda_i/tests -q` 1209 passed.

### Live Clean farm on At4A (no pokes, no assists)

- Pin: `At4A` (`custom_integrations/LegendOfZelda-Nes/At4A.state`), screen `0x4A`,
  mode 5, initial pos `(0, 149)`, initial health `0x32` (2 filled / 4 containers),
  kill count 1.
- Controller fixes in `heart_farm.py` (486 LOC):
  - Screen entry: `_is_entering_screen` pushes East (`x < 36` on `0x4A`) onto corridor
    before chasing off-axis threats into walls or edge-snapping.
  - Wall slide: `_advanced(last_xy, xy, last_dir)` detects actual axis progression
    rather than strict xy equality, preventing perpendicular slide false misses.
  - Occupancy double-grading: set `walker._graded = True` in `_grade_occupancy` so
    `walker.next_dir()` does not invoke `observe()` with stale claims.
  - Action lock immunity: Link swinging sword (`objects[0].state != 0`) or in hit stun
    (`mode != PLAY_MODE`) clears `last_dir`/`last_xy` to avoid fencing valid cells.
  - Drop states: added clock (`0x21`) to `_RUPEE_STATES`.
- Result (`nes/zelda_i/scripts/probe_at4a_farm.py`):
  - Link walks onto 0x4A corridor, engages red octorok (`0x0D`), kills at f=112.
  - Floor drop: Obj 5, type `0x60`, state `0x22` (heart) at `(64, 156)`.
  - Link prioritizes drop scoop over chase, collects heart at f=145.
  - Health: `0x32` (2/4) → `0x33` (3/4 filled hearts). Phase: `DONE` (`farm_ok_2_to_3`).
  - Duration: 145 frames (~2.4s). Occupancy misses: 22. Kills: 1.
  - Screenshot: `recordings/at4a_clean_farm.png`.

Leftover: L2 door path `0x5C` still `rr-ps7.3`. Fairy drop was not
live-picked (only 1 kill needed for 3 hearts; `$0627` at 2). Do not close.

## Previous sitting (2026-09-10)

Packed L4/L5 leave xy from pin settle + `screen_glance`. Hardened
`HeartFarmController` occupancy (miss → block cell → replan; no path →
stand). Unit tests green. No `path.py` / `graph.py` / `nav.py` edits.

## Measured leaves (one ROM each, `--no-video`, no pokes)

`zelda_i.runner.open_env` ImportError (`load_state` missing). Fallback:
`make_env(GAME, pin, GAME_DIR)` + `reset_obs` + idle settle +
`leftover_from_snapshot`. Did not grant Whistle. Did not poke hearts.

### L4 leave (`Level4Complete` → `PostL4TriforceSettleController`)

- Entry: dungeon `0x03` mode 18 `(120,149)` TF `0x0C` health `0x74` whistle 0
- Leave: OW play **`0x45` `(128,125)` mode 5** TF `0x0C` health `0x77`
  (8 HC, 7 filled) keys 0 bombs 7 rupees 19 raft 1 sword 1
- Settle 633f. Packet packed `x=128 y=125 ±4`, stitch TF stays **`0x0F`**
  (natural cumulative). Pin is a development skip of L1/L2 (`0x0C`).

### L5 leave (`Level5Complete` → `PostL5TriforceSettleController`)

- Entry: dungeon `0x14` mode 18 `(120,149)` TF `0x1C` whistle **1 already
  on pin**
- Leave: OW play **`0x0B` `(112,125)` mode 5** TF `0x1C` health `0xDD`
  (14 HC, 13 filled) keys 1 bombs 6 rupees 26 whistle 1 food 0 candle 0
- Settle 1116f. Packet packed `x=112 y=125 ±4`, stitch TF stays **`0x1F`**,
  `whistle=1` copied from pin (not granted this sitting).

## Heart farm

Occupancy on chase/patrol only (restock leave still cardinal). True
no-move miss (not 1px grade — OW 2px slides). `min_filled<=0` is the
`farm_below_hearts=0` analog (inert skip). Unit-only: no low-heart OW pin
used.

## Blockers

- **`path.py` lock / `rr-ps7.3`:** "L2 door path does not die on `0x5C`
  under Clean" is blocked. Do not edit the L2 walk.
- Integrator still owns `route_eligible` / `verified=True` / STATUS.
- Dev pins are not natural cumulative TF (`0x0C`/`0x1C` vs `0x0F`/`0x1F`).
- `scripts/run_l4_to_l5.py` / `run_l5_to_l6.py` are missing (pyc only).
- `zelda_i.runner.open_env` cannot load pins (`load_state` import).

## Leftover

Promote L4/L5 packets after a natural-segment leave (real predecessor TF
bits, full health, `verified=True`). Live Clean farm verified on At4A
(2→3 hearts in 145f). L2 `0x5C` stays `rr-ps7.3`.
