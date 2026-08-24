# Zelda I — architecture hygiene

Rules that keep L4–L9 from repeating L2/L3 copy-expand debt.

## Layers

| Layer | Module(s) | Owns |
|-------|-----------|------|
| Anchors | `anchors.py` | Door screens, entry rooms, TF bits (L3–L9) |
| OW path engine | `ow_path.py` | Hop/maze/door frame policy |
| OW geometry | `level*_overworld.py` | Hop tables + thin controller subclasses |
| Room combat | `dungeon.py` + `level*_dungeon.py` | `DungeonRoomSpec` tables only |
| Bomb walls | `level2_puzzles.BombWall` + `bomb_wall_path` | Geometry + one traverse controller |
| Multi-room paths | `level2_bomb_path` (`make_*`), `level3_path`, `level3_raft_path`, `level4_path` / `level4_maze_path` / `level4_stepladder` / `level4_exit60` / `level4_room_nav`, `level5_path` (facade; west/whistle/cellar/tf), `level*_boss_*` | Path controllers + path timing knobs |
| L3 raft | `level3_raft_path` (canonical) | Raft passage controller; **not** `level3_path` |
| L3 geometry | `level3_geometry` | Door bands, bomb stands, raft channel ints |
| Door planner | `door_graph/` (L2–L5 + L9 fixture) | Offline BFS; stands must match `BombWall` |
| Walk physics | `walk_physics.py`, `predict.py` | OccupancyWalker grades `move`; no path → stand |
| L3 dest spine | `level3_spine.py` | `--through level3` dest 0x5b (west key closed) |
| Route catalog | `routes.py` (L1–L2), `routes_later.py` + `route_legs_later.py` (L3–L5 + L9 fixture) | NamedRoute / RouteLeg; L6–L8 stay stubs |
| Composer | `route_composer.py` | Bind existing controllers to leg ids; no path geometry |
| Eligibility | `route_eligible.py`, `natural_entry.py` | Lab-fixture vs route pin; STATUS claim gate |
| Resource cost | `health_cost.py`, `damage_heatmap.py` | Hop heart costs + Survival heatmap ranker |
| Item gates | `item_gate_hops.py`, `item_gate_routes.py` | Candle / white sword / bomb shop NamedRoutes |
| Combat helpers | `combat.py` + `combat_behaviors.py` | Hitbox swing gate + reusable enemy policies |
| Continuous spine | `survival_spine.py` + `scripts/run_survival_spine.py` | One env, power-on, stop at first fail |
| Scripts | thin CLIs + library controllers | Env/assist/report only — **no path logic** |

## Hard rules

1. **No new phase machine for bomb walls.** Configure `BombWallController` with a `BombWall`.
2. **`level*_dungeon.py` = specs + stop predicates only.** Path controllers and
   path timing (`*_MAX_FRAMES`, `SPAWN_SETTLE_FRAMES`, raft channel knobs) go in
   `*_path` / `level3_geometry` / boss modules — not the room table.
3. **No new screen/TF hex** outside `anchors.py` (L3+) or `overworld`/`ram` (L1–L2).
4. **No path logic in `scripts/`.** Call library controllers; use `zelda_i.runner`.
5. **Enemy type IDs** live in `dungeon_ids` (and re-exports in `dungeon.py` for engine types).
6. **door_graph bomb stands** import from `level2_puzzles` — do not hardcode `(120, 101)` again.
7. Prefer files under **~600 lines**; never grow a dungeon table past **1k** with controllers.
8. **No state-seamed viewing tapes.** The spine is one continuous emulator
   session (`continuous_emulator_session=true`) or it is not a spine tape.
   Seam cards / clip concat are deleted, not a product.

## Artifact retention

- STATUS evidence JSON/states: keep under `recordings/` and named checkpoints.
- Lab probes, PNG dumps, one-off agent logs: prune or gitignore; do not commit bulk PNGs.
- `recordings/` is local evidence, not product source.
- L5 `_probe_l5_*` / `_stitch_*` lab one-offs were pruned (rr-cq5z). Durable
  CLIs are `scripts/run_level5_*.py` and `scripts/run_survival_spine.py`.
  Local `recordings/stitches/` leftovers are not a product.

## Backward compatibility

- Bomb walls: prefer `make_bomb_north_controller()` etc. from `level2_bomb_path`
  (or `level2_dungeon` shim). Class-named aliases still resolve to the same factories.
- Raft: prefer `from zelda_i.level3_raft_path import Level3RaftPathController`.
  One shim: `level3_dungeon` (`__getattr__`). **Not** re-exported from `level3_path`.
- Inventory poke (`ADDR_SELECTED_ITEM` / `B_ITEM_BOMB`) lives in `dungeon_ops`, not
  `bomb_wall_path`. `$0656` is **1=bombs, 2=arrows, 4=candle** — never `bombs=2`.
