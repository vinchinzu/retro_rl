# rr-npv.1 residual — Clean L3 Entrance→TF dest hops

Stopped at fixture-live. `route_eligible=false`. Three serial ROM reds.
Do not STATUS. Do not close the bead.

## Green dest hops (Level3Entrance pin, `--no-infinite-life --no-video`)

| Stage | Frames | Leftover |
|-------|--------|----------|
| west_key | 507 | 0x7b key dest |
| north_chain | 2791 | 0x5b Darknuts cleared (natural bombs) |
| bomb_5b | 364 | play 0x5c; bombs 8→7; pause-select, no poke |

Glance at bomb_5b leave (trial 2/3): room 0x5c, mode 5, bombs 7, keys 4, tf 0x03.

## Serial reds (one change each, then stop)

1. `bomb_5b` stand_timeout at **(176,125)** tile 117, bombs=10. Naive `_goto_stand` RIGHT into a block. Fix: y-first `approach_waypoints=((192,141),)`.
2. `clear_5c` death at **(120,125)** after occupancy_patrol boxed in 0x5c diamonds (GAME OVER, 2 Darknuts live). Fix: waist/south patrol, `occupancy_patrol=False`, `contact_backstep=8`.
3. `clear_5c` death again at **(153,101)** 2787f. Engage still chased north of the waist onto diamonds. hearts lo=0, mode 17, bombs 7 unused in the fight.

## Blocked

Wooden-sword 0x5c Darknut clear on diamond floor with 4 hearts, Clean, no poke, no infinite life. Dest hop entered 0x5c; combat is not dest-safe.

Do not poke bombs/keys/doors. Do not extend hop timeouts without a new miss. Next sitting: occupancy-seed 0x5c diamond cells (miss → block → replan; no path → stand) or a bomb-from-waist dest policy that does not chase y≤109.

## Glance (trial 3 leftover)

room **0x5c**, mode **17**, xy **(153,101)**, tf **0x03**, keys **4**, bombs **7**, health **0x70** (lo 0 ≠ hi 7). PNG `recordings/l3_entrance_tf_t0_final.png`.

## Notes

- `zelda_i.runner.open_env` imports missing `load_state`; isolated runner uses `make_env(GAME, Level3Entrance, GAME_DIR)`.
- Survival Raft→TF still uses `Level3BossPathController.path_to_5d` dest-hop drive (no `idle(n)` / `push_dir` holds). Integrator owns spine.
