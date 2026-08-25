# rr-ahjo worker — WS Entrance → Main Shaft (`ws_entrance_to_main`)

Worker hop only. No STATUS-promote. No attic/basement/Phantoon rewrite.
`--to ws` / `WS_ONLY_HOPS` / `DEFAULT_CONTINUOUS_TIP` untouched.

## Pin
`scratch/post_ws_poweron.state` `0xCA08` `(57,139)` p1 gs=8 items `0x3105`
beams `0x1007` max PB 5 selected=2 (supers leftover from green Super).

## BEFORE (scaffold 240f RIGHT+B)
Timeout. Closed blue door; speed-booster crash wall `x=987` p137.
Report: `scratch/ws_entrance_to_main_before.json`

## Dump
Coverns `0xD87F` (phase in/out; tank, no fight). Dash hits `x>=960` at
f136 `~(968,139)` p9 speed=4. Crash `x=987` p137 if the door is closed.
selected_item=2 at boot — must SELECT to beam before any X.

## AFTER (product)
`play_ws_entrance_to_main`: require `0xCA08` → `select_weapon(0)` →
RIGHT+B until door seat `x>=900` → `play_run_shoot_exit` beam
(`super_door=False`, run 0, spin 0) → `wait_ordinary_room` `0xCAF6` gs=8.

Dual GREEN **403f** ×2 exact `(1063,907)` p9 gs=8 dt=0 selected=0.
Leave: `scratch/post_ws_entrance_to_main.state`
Reports: `scratch/ws_entrance_to_main.json` + `_dual.json`

| | frames | seconds | clock | result |
|---|---:|---:|---|---|
| before (scaffold) | 240 | 3.993 | 00:04.00 | TIMEOUT `0xCA08` `(987,139)` p137 |
| after (product) | 403 | 6.706 | 00:06.72 | dual GREEN `0xCAF6` `(1063,907)` p9 gs=8 |
| Δ (after − before) | +163 | +2.712 | +00:02.72 | fail → success (timeout is not a time) |

Times via `format_segment_time` (NTSC 60.0988).

## Tests
`tests/test_wrecked_ship_scaffold.py`: seat / action / never Super the
blue door / settle gs=8 / product is not `_scaffold_exit`. Basement and
Phantoon stay scaffold. `WS_ONLY_HOPS` unchanged (`["west_ocean_to_ws"]`).

## Probe
`snes/super_metroid/scripts/probe/ws_entrance.py` `bench` / `dump` /
`pure --dual`. Default source `post_ws_poweron.state`, boot settle 5,
no free-place.

## Manager
Residual / docs / beads / commit not landed by this worker.

## LAND NOW (parent, worker complete)
Worker dual-green is on disk. Do NOT re-implement the hop.
Leave pin: scratch/post_ws_entrance_to_main.state
Reports: scratch/ws_entrance_to_main.json + _dual.json (403f ×2, 0xCAF6 (1063,907) p9 gs=8)
Probe: scripts/probe/ws_entrance.py

Land: residual rr-ahjo-residual.md; STATUS/plan/AGENTS/SOURCE_STATES next-line only
(NOT promote); new bead Main Shaft → basement discovered-from:rr-ahjo;
bd close rr-ahjo if you agree; bd sync; tests; commit code+beads; do NOT push.
Do NOT append to --to ws / WS_ONLY_HOPS. Default CLI stays ice. rr-g3nj stays open.

## Manager close
- Dual GREEN accepted **403f** ×2 `0xCAF6` `(1063,907)` p9 gs=8.
- Residual `docs/tasks/rr-ahjo-residual.md`. Next bead `rr-4btp`.
- `bd close rr-ahjo`. No STATUS-promote. No push.
