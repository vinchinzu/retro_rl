# Zelda I — architecture hygiene

Rules that keep L4–L9 from repeating L2/L3 copy-expand debt.

## Layers

| Layer | Module(s) | Owns |
|-------|-----------|------|
| Anchors | `anchors.py` | Door screens, entry rooms, TF bits (L3–L9) |
| OW path engine | `overworld/path.py` | Hop/maze/door frame policy |
| OW TF settle | `overworld/settle.py` | `TriforceSettleSpec` rows + one idle controller (L1–L5) |
| OW geometry | `level*/overworld.py` | Hop tables + thin controller subclasses |
| Room combat | `dungeon/engine.py` + `level*/dungeon.py` | `DungeonRoomSpec` tables only |
| L6 wizzrobe combat | `level6.wizzrobe` | 0x7a/0x78 backstep; re-exported from `level6.dungeon` |
| Bomb walls | `level2.puzzles.BombWall` + `dungeon/bomb_wall.py` | Geometry + one traverse controller. L4+ leftover B-slot is a hop input: pass `select_item=B_SLOT_BOMBS`. L2/L3 omit it (poke-assisted). |
| Pause-select | `dungeon/pause_select.py` | Already-owned B-item. Never poke `$0656`, never emit B. RIGHT only counts when `$0656` actually changes. |
| Multi-room paths | `level2.bomb_path` (`make_*`), `level3.path`, `level3.raft_path`, `level4.path` / `level4.maze_path` / `level4.stepladder` / `level4.exit60` / `level4.west31` / `level4.keyup20` / `level4.map21` / `level4.mappick` / `level4.bomb11` / `level4.key01` / `level4.clear12` / `level4.gleeok13`, `level5.path` (0x76/0x77 steps + `Level5NavSpec` rows + `walk_axis` glue) / `level5.west_path` (`WestLeaveSpec` rows) / `level5.whistle_path` (`BombWallSpec` rows) / `level5.cellar_path`, `level6.path` (occupancy north 0x78→0x68, 0x38 push, 0x18 settle; not a dest facade), dest hops (`L6_THROUGH` / per-hop modules + shared `level6.door_hop`), `level6.gleeok18` (0x44 south-stand + post-body census), `level6.room19` (0x18→0x19 cluster), `level6.north39` (0x3A west enter-stop), `level8.path` (factories) / `level8.north_column` (fixture-live 0x7E→0x1E) / `level8.cellar` (0x0F return), `level7.stairs0d` (0x0D `block_stairs` push + ring walk-on of cellar 0x7B), `level*.boss_*` | Path controllers + path timing knobs |
| L3 raft | `level3.raft_path` (canonical) | Raft passage controller; **not** `level3.path` |
| L3 geometry | `level3.geometry` | Door bands, bomb stands, raft channel ints |
| Door planner | `door_graph/` (L2–L5 + L9 fixture) | Offline BFS; stands must match `BombWall` |
| Room geometry | `dungeon/tilemap.py` | Read-only cart-WRAM `$6530` room tile map (floor / block / stairs / door cells). Measure rooms with this, not with direction-sensitive `$049E` `colliding_tile` sweeps. Never writes. |
| Walk physics | `walk/physics.py`, `walk/predict.py` | OccupancyWalker grades `move`; miss → block that cell → replan; no path → stand |
| L6 dest helpers | `level6.occupancy` | leftover / dest success / occupancy halt (L6-prefixed dest). Halt-on-miss came from the (deleted) east3a diagnostic, not the OccupancyWalker default. Distinct from `level4.occupancy` (seeds). |
| L3 dest spine | `level3/spine.py` | `--through level3` dest 0x5b (west key closed) |
| L5 dest spine | `level5/spine.py` | `--through level5` TF `0x10` in room `0x14` |
| Eligibility | `route/eligible.py`, `route/natural_entry.py` | Lab-fixture vs route pin; STATUS claim gate |
| Dungeon treasures | `route/treasures.py` | First-quest wiki items vs default-spine collection |
| OW locations | `overworld/locations.py` | First-quest cave dests, open method, rando ROM decode, enemy-drop farms (`farm_at`, `worth_rupee_farm`, restock pairs) |
| Combat helpers | `combat.py` + `dungeon/behaviors.py` + `dungeon/gleeok.py` | Hitbox swing gate, reusable enemy policies, shared Gleeok sensors (L4+L6) |
| Continuous spine | `spine/survival.py` + `spine.hops.attach_hops` + `level*/spine.py` hop tables | One env, power-on, stop at first fail. New dest hops are `SpineHop` rows, not `*_stages`/`*_success` pairs. |
| Scripts | thin CLIs + library controllers | Env/assist/report only — **no path logic** |

## Hard rules

1. **No new bomb-wall or pause-select phase machine.** Configure
   `BombWallController` with a `BombWall`. Drive `PauseSelectController`.
   Leftover B-slot is a hop input (`select_item=`). L2/L3 omit it (poke-assisted).
2. **`level*/dungeon.py` = specs + stop predicates only.** Path controllers and
   path timing (`*_MAX_FRAMES`, `SPAWN_SETTLE_FRAMES`, raft channel knobs) go in
   `*_path` / `level3.geometry` / boss modules / `level6.wizzrobe` — not the
   room table.
3. **No new screen/TF hex** outside `anchors.py` (L3+) or `overworld`/`ram` (L1–L2).
4. **No path logic in `scripts/`.** Call library controllers; use `zelda_i.runner`.
5. **Enemy type IDs** live in `dungeon.ids` (and re-exports in `dungeon/engine.py` for engine types).
6. **door_graph bomb stands** import from `level2.puzzles` — do not hardcode `(120, 101)` again.
7. Prefer files under **~600 lines**; never grow a dungeon table past **1k** with controllers.
8. **No state-seamed viewing tapes.** The spine is one continuous emulator
   session (`continuous_emulator_session=true`) or it is not a spine tape.
   Seam cards / clip concat are deleted, not a product.

## Artifact retention

- STATUS evidence JSON/states: keep under `recordings/` and named checkpoints.
- Lab probes, PNG dumps, one-off agent logs: prune or gitignore; do not commit bulk PNGs.
- `recordings/` is local evidence, not product source.
- L5 `_probe_l5_*` / `_stitch_*` lab one-offs were pruned (rr-cq5z). Durable
  CLI is `scripts/run_survival_spine.py` (`SpineHop` dests, including L5).
  Local `recordings/stitches/` leftovers are not a product.
- L4 parked segment CLIs (`run_level4_rooms.py`, `run_level4_entrance_tf.py`,
  `run_level4_continuous_tf.py`, `run_level4_entry.py`, `run_level4_gleeok.py`)
  were pruned. Durable L4 CLI is `scripts/run_survival_spine.py`.
- L9 parked recon CLIs (`run_level9_ganon.py`, `run_level9_patra.py`,
  `run_level9_room62.py`, `run_level9_stairs.py`) were pruned. Controllers live
  in `level9/`; Composer binds the fixture dests.

## Backward compatibility

- Bomb walls: prefer `make_bomb_north_controller()` etc. from `level2.bomb_path`
  (or `level2.dungeon` shim). Class-named aliases still resolve to the same factories.
- Raft: import `Level3RaftPathController` from `level3.raft_path`.
- Inventory poke (`ADDR_SELECTED_ITEM` / `B_ITEM_BOMB`) lives in `dungeon.ops`, not
  `dungeon.bomb_wall`. `$0656` is **1=bombs, 2=arrows, 4=candle** — never `bombs=2`.
