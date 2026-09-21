# Living residuals — Sanctuary spine fan-out

Planner owns `docs/STATUS.md`. Each claimed bead overwrites **its** residual.
Session: `.grok/skills/alttp-session/SKILL.md`.

Closed 2026-09-20 (sitting 1): `rr-ccxt.1`–`.8`, `rr-a3s`, `rr-ejn`.

| Bead | Residual | Owns |
|------|----------|------|
| `rr-ccxt.9` Compose 0x50→well→0x72 | `docs/tasks/compose-stair-residual.md` | `escape_graph.py`, `tests/test_escape_graph.py`, `maps/room_01.json` notes, `scratch/probe_compose_stair.py` |
| `rr-ccxt.10` B1 key pin census | `docs/tasks/key-census-residual.md` | `scratch/probe_key_pins.py`, `recordings/probe_key_pins/` (no maps/graph) |
| `rr-ccxt.11` 0x72 north-ledge drop | `docs/tasks/ledge-drop-residual.md` | `scratch/probe_ledge_drop.py`, `recordings/probe_ledge_drop/`, `maps/room_72.json` notes |
| `rr-ccxt.12` 0x82 water maze | `docs/tasks/water-maze-residual.md` | `scratch/probe_82_water.py`, `recordings/probe_82_water/`, `maps/room_82.json` notes |
| `rr-ccxt.13` 0x81 cell door | `docs/tasks/cell-door-residual.md` | `scratch/probe_cell_door.py`, `recordings/probe_cell_door/`, `maps/room_81.json` landing |
| `rr-ccxt.14` Stair settle cap | `docs/tasks/stair-settle-residual.md` | `opening_route/room_engine.py` settle_destination, `tests/test_room_engine_edges.py` |
| `rr-ccxt.15` Glance 0x01/0x72 | `docs/tasks/glance-72-residual.md` | `screen_glance.py`, `tests/test_screen_glance.py` |
| `rr-ccxt.16` Spine CLI room_72 | `docs/tasks/spine-cli-residual.md` | `scripts/run_opening_spine.py`, `tests/test_opening_spine.py` |
| `rr-ccxt.17` Drop leftover → 0x72 south | `docs/tasks/drop-south-residual.md` | `scratch/probe_drop_south.py`, `maps/room_72.json` notes |
| `rr-ccxt.18` SecondKey out of 0x71 | `docs/tasks/second-key-residual.md` | `scratch/probe_second_key.py`, `maps/room_71.json` |
| `rr-ccxt.19` Cell door from SecondKey | `docs/tasks/cell-key-residual.md` | `scratch/probe_cell_key.py`, `maps/room_81.json` landing, optional isolated graph hop |

Leave proof is RAM glance, not MP4. Do not STATUS-promote.

## 2026-09-20 sitting 2 leftover

Continuous tip is still room `0x50`. `room_50_east_to_0x01` and
`room_01_down_to_0x72` are **natural_entry** (not continuous). Reverse
`room_72_north_to_0x01` stays isolated.

- F1 well: 0x50-east leftover 0x01 ~(560, 120) → (760, 99) **UP** → 0x72 ~(1273, 3665)
- 0x72 north ledge: `CastleB1Key` spends the small key on the south key door → lower floor (1272, 3945) keys=0, then west/south → 0x82 (1190, 4108)
- 0x82 water: 0x72-south leftover chains east-wall x≥1312 then south walkway → 0x81 (996, 4495) `$F3CC==0`
- `CastleB1SecondKey` wraps out of the 0x71 east pocket (NW lip UP+LEFT at y=3960) → 0x81 **(632, 4155) keys=1** `$F3CC==0` layer `$00EE==1`
- 0x81 `west_to_0x80`: LEFT at (608, 4169) with keys=1 **stays 0x81**, key not spent, gold jail door. Next: layer vs door type (`rr-ccxt.20`)
- Cell pin `CastleZeldaFollower` still 0x80 (352, 4168) `$F3CC==1` as loaded.
