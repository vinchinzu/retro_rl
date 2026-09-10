# rr-npv.6 residual — Clean OW stitch + heart farm

Fixture-live only. Do not STATUS. Do not close the bead. `route_eligible`
stays false.

## This sitting (2026-09-10)

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
bits, full health, `verified=True`). Live Clean farm on an OW screen with
`filled_hearts < 3` once a pin exists. L2 `0x5C` stays `rr-ps7.3`.
