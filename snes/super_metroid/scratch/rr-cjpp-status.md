# rr-cjpp worker — WS Basement → Phantoon's room (`ws_basement_to_phantoon`)

Claimed **DONE** (worker hop). Dual GREEN. Manager owns residual / docs /
beads / commit. Do **not** STATUS-promote. Do **not** fight Phantoon.

## Pin (source)

`scratch/post_ws_main_to_basement.state` `0xCC6F` `(657,92)` p24 gs=8
items `0x3105` beams `0x1007` max PB 5 selected=2. After boot settle 5:
`(657,91)` p165 facing=4 on the center platform.

Target: `0xCD13` Phantoon's Room, ordinary gs=8 (`door_transition=0`).
Leave pin: `scratch/post_ws_basement_to_phantoon.state`
`(39,124)` p81 gs=8 dt=0 items `0x3105` beams `0x1007` selected=2 max PB 5.

## Public policy

Unpowered basement hallway. Map station LEFT is dead — skip. Walk RIGHT.
Bomb the morph-tunnel obstruction (X while morph; morph bombs are X, not A).
Unmorph. Gadora eye door: Super Missile (already selected). Gadora leaves a
blue shell — shoot and walk through. Enter Phantoon `0xCD13`. Coverns
possible; tank (spin-escape knockback 137/138). Do not fight Phantoon.

## Product

`routes/kpdr/k6/ws_basement.py` `play_ws_basement_to_phantoon` (re-exported
from `wrecked_ship.py`). Not `_scaffold_exit`. Probe:
`scripts/probe/ws_basement.py` bench / dump / pure --dual.

## Verify paste

```bash
uv run python snes/super_metroid/scripts/probe/ws_basement.py pure --dual
# → GREEN dual=True 0xCD13 (39,124) p81 gs=8 dt=0 frames=718 ×2 exact
#    items 0x3105 beams 0x1007 selected=2 max PB 5
#    leave pin scratch/post_ws_basement_to_phantoon.state
```

## BEFORE / AFTER / Δ

| | frames | seconds | clock | where |
|---|---:|---:|---|---|
| BEFORE scaffold (timeout) | 240 | 3.993 | 00:04.00 | still `0xCC6F` `(863,187)` p137 |
| AFTER product dual GREEN | 718 | 11.947 | 00:11.97 | `0xCD13` `(39,124)` p81 gs=8 ×2 |
| Δ | +478 | +7.954 | +00:07.97 | success vs timeout scaffold |

Reports: `scratch/ws_basement_to_phantoon_before.json`,
`scratch/ws_basement_to_phantoon.json`,
`scratch/ws_basement_to_phantoon_dual.json`.

## Dump (halt at first miss)

Land idle on platform `(657,91)` p165. RIGHT+B drop to floor, Coverns at
~x=863 knockback-stall (p137). Morph-tunnel bomb-block stall **x=1051**
y=201. Alcove **x≳1160**. Gadora Super then remaining blue shell.

## Hard nos honored

- Did **not** implement a Phantoon fight
- Did **not** edit `play_ws_entrance_to_main` / `play_ws_main_to_basement`
- Did **not** append to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`
- Did **not** STATUS-promote / commit / push / write `recordings/ws.json`

Unit tests: `tests/test_wrecked_ship_scaffold.py` (+ `test_post_ice_spine.py`
WS_ONLY_HOPS lock) **20 passed**.

## LAND NOW (parent — worker dual-green, manager still running)

Worker 01a0352c-3332-7f82-b04a-c63499a7f67c is DONE. Dual GREEN **718f** ×2
`0xCD13` `(39,124)` p81 gs=8. Leave pin on disk. Do **not** re-implement.
Do **not** fight Phantoon.

Land now:
1. Review `routes/kpdr/k6/ws_basement.py` + re-export. Confirm Main/Entrance untouched. `WS_ONLY_HOPS` still `["west_ocean_to_ws"]`.
2. Residual `docs/tasks/rr-cjpp-residual.md`. Next bead: Phantoon **fight** (NEW). Do not start it.
3. Tiny docs next-line only. NOT STATUS-promote. Default CLI ice.
4. `bd close rr-cjpp` if you agree dual-green. `bd sync`. Tests. Commit code+beads. Do **not** push.

## Manager close

- Dual GREEN accepted **718f** ×2 `0xCD13` `(39,124)` p81 gs=8 dt=0.
- Residual `docs/tasks/rr-cjpp-residual.md`. Next bead `rr-tlaq` (Phantoon fight).
- `bd close rr-cjpp`. No STATUS-promote. No push.
- Main/Entrance hop bodies untouched. `WS_ONLY_HOPS` still `["west_ocean_to_ws"]`.
  Default CLI stays `ice`. `rr-g3nj` stays open.
