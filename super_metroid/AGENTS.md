# Agent Instructions — Super Metroid

Super Metroid scripted full-clear project. Shared process:
[`snes_oneshot/docs/FULL_RUN_PROCESS.md`](../snes_oneshot/docs/FULL_RUN_PROCESS.md).

## Evaluation contract

- Target: one continuous power-on-to-ending run.
- Allowed assists: unlimited health (in-game energy) and unlimited ammo,
  exactly as defined in
  [`docs/ASSIST_CONTRACT.md`](docs/ASSIST_CONTRACT.md).
- Resource assists remove attrition only. They must not grant uncollected ammo
  types, capacity, equipment, items, movement abilities, door state, map
  state, boss/event flags, rooms, or completion.
- Record every assist write in the full-run manifest.
- Completion requires the natural endgame escape and ending/credits evidence;
  defeating the final boss alone is not a clear.

## Organization

- Keep RAM addresses, route logic, maps, states, logs, recordings, and policy
  in `super_metroid/`.
- Save states belong under `custom_integrations/<GameId>/`.
- Use clean states for fast development and natural-entry states for
  acceptance.
- Prefer room/door/inventory progress vectors over coordinate-only watchdogs.
- Keep the last successful full-run baseline; candidates use separate reports.

## Immediate goal

**Primary: play every completion-path room** (controller/policy). Door-warps
are topology diagnostics only — not route evidence.

1. **Path board:** `docs/PATH_ROOM_BOARD.md` — 107 rooms / 199 hops; regenerate
   with `scripts/export_path_room_board.py`.
2. **Current bottleneck:** pure approach onto PB sill (wall@x≈613 / y1051 ledge) and mid-maze **405→225** after wall@437; sill **entry**, wall@437 **pure break**, and pocket **collect** (x≤225) exist.
3. **Continuous next:** close remaining place bridges, then power-on through PB and next open hops.
4. Boss fights only after natural entry to that boss room exists on the chain.
5. Topology warps (`probe_route.py full` / `full-hybrid`) — debug only.

### Shared dev helpers

`dev_common.py` owns reusable development primitives:

- `boot_from_state`, `door_warp` (waits for game state 8), `place_samus`,
  `apply_dev_survivability`, `enemy_hps`, `select_weapon`, `save_dev_state`

`mother_brain_dev.py` and `kraid_dev.py` re-use those; do not re-implement
door-warp settle logic in new probes. Door-warp settle must wait for
**game state 8** (not merely `ordinary` phase) — multi-screen loads sit in
state 11 for 50–100+ frames.

### Path room board (play clearance — primary)

```bash
# Regenerate 107-room / 199-hop board (JSON + markdown)
uv run python super_metroid/scripts/export_path_room_board.py

# Furthest post-Super controller probe (no door-warp)
uv run python super_metroid/scripts/probe_post_spore_pb.py --to main

# Pink PB sill entry (place bridge onto sill if not already there)
uv run python super_metroid/scripts/probe_post_spore_pb.py --to pb-door \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_intercept.state

# Pure morph-bomb open wall@437 from bottom-door spawn
uv run python super_metroid/scripts/probe_post_spore_pb.py --to pb-maze-wall \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_pb_door_entered.state
# PB collect (wall pure + place(220,395) mid-maze bridge if needed)
uv run python super_metroid/scripts/probe_post_spore_pb.py --to pb-collect \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_b1_pb_door_entered.state
```

Board: `docs/PATH_ROOM_BOARD.md`, `maps/path_room_board.json`.

### Topology skeleton (door-warp diagnostics only)

```bash
# List hop coverage (all 22 completion legs)
uv run python super_metroid/scripts/probe_route.py list

# Door-warp all 22 legs — connectivity probe, NOT route evidence
uv run python super_metroid/scripts/probe_route.py full
uv run python super_metroid/scripts/probe_route.py full-tour
uv run python super_metroid/scripts/probe_route.py full-hybrid
```

Reports are labeled `developmentOnly: true`. Hop table:
`maps/full_route_hops.json`. Runner: `route_dev.py`.

### Late-game route skeleton (boss fights skipped)

```bash
# Phantoon → Ridley (Gravity, Botwoon, Draygon; fights skipped)
uv run python super_metroid/scripts/probe_route.py phantoon-to-ridley

# Ridley → Mother Brain (statues + Tourian; fights skipped)
uv run python super_metroid/scripts/probe_route.py ridley-to-mb

# Phantoon → Landing Site finish
uv run python super_metroid/scripts/probe_route.py late-full

# Single leg
uv run python super_metroid/scripts/probe_route.py leg draygon ridley
```

Late hop table: `maps/late_game_route_hops.json` (9-leg subset of full).

### Post-Spore Super collect → Phantoon (Track B)

Working route board: [`docs/ROUTE_SUPERS_TO_PHANTOON.md`](docs/ROUTE_SUPERS_TO_PHANTOON.md).

```bash
# Continuous power-on → Super Missile collect (STATUS baseline)
uv run python super_metroid/scripts/record_start_to_supers.py --no-video
uv run python super_metroid/scripts/record_start_to_supers.py

# From natural_post_spore_spawn: Supers → farming → Big Pink → pocket crest
uv run python super_metroid/scripts/probe_post_spore_pb.py --to crest
uv run python super_metroid/scripts/probe_post_spore_pb.py --to super-block
uv run python super_metroid/scripts/probe_post_spore_pb.py --to big-pink
uv run python super_metroid/scripts/probe_post_spore_pb.py --to farming
uv run python super_metroid/scripts/probe_post_spore_pb.py --to supers
# Double-tap morph to tunnel floor; full main shaft (controller)
uv run python super_metroid/scripts/probe_post_spore_pb.py --to tunnel-floor
uv run python super_metroid/scripts/probe_post_spore_pb.py --to main
uv run python super_metroid/scripts/probe_post_spore_pb.py --to tunnel-west \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_big_pink_open.state
```

Controller: `post_spore_controller.py` (no progression RAM writes). Crest lands
~x=1125 standing; crouch-Super clears (69,87); **double-tap DOWN** morphs
(standing y≈1387 = morph y≈1401 pose height); tunnel-west + X bombs → main
x≲750. Climb to Pink PB open.

### Kraid / Power Bombs / Phantoon entry

```bash
# Dev Kraid fight (door-warp from eye door + Super spray)
uv run python -c "from super_metroid.kraid_dev import run_kraid_fight; print(run_kraid_fight())"

# Power Bombs + ship route → Phantoon entry (development only)
uv run python super_metroid/scripts/probe_phantoon.py collect-pb
uv run python super_metroid/scripts/probe_phantoon.py capture-entry
uv run python super_metroid/scripts/probe_phantoon.py ship-route
```

Furthest topology: **Landing Site via full late door-warp skeleton**. Boss
fights intentionally deferred.

Key door pointers (bank `$83` → destination room):

| Door | Dest |
|------|------|
| `0x8DDE` | Pink PB top (`0x9E11`) |
| `0x901E` | Hellway |
| `0x908A` | Caterpillar |
| `0x90BA` | Elevator to Cat |
| `0x8AF6` | Crateria Kihunter |
| `0x8A36` | Moat |
| `0x8AEA` | West Ocean |
| `0x89D6` | WS Entrance |
| `0xA1BC` | WS Main Shaft |
| `0xA21C` | WS Basement |
| `0xA2AC` | Phantoon (`0xCD13`) |
| `0x91B6` | Kraid |
| `0xAAC8` | Mother Brain |
| `0xAA8C` | Escape Room 1 |

### Mother Brain / escape development

```bash
# Capture door-warp fixtures (development only; not continuous evidence)
uv run python super_metroid/scripts/probe_mother_brain.py capture-mb
uv run python super_metroid/scripts/probe_mother_brain.py capture-escape1

# Spray probe / coarse escape nav
uv run python super_metroid/scripts/probe_mother_brain.py spray-mb --frames 3600
uv run python super_metroid/scripts/probe_mother_brain.py run-escape --frames 12000
```

Helpers live in `mother_brain_dev.py` (shared warp/place from `dev_common.py`).
High WRAM (events/boss bits at `$7E:D820+`) must use `read_bank7e_wram` /
`write_wram_u8` — raw `env.get_ram()[0xD820]` is open-bus garbage.

sm_rev reference: `../snes_editor/super_metroid_rl/sm_rev/src/enemy_mother_brain.c`
and `enemy_ridley_zebetite.c` (zebetites regen 1 HP/frame up to 1000).

### Room-development commands

```bash
uv run python super_metroid/scripts/export_room_problems.py
uv run python super_metroid/scripts/run_room_problem.py ready --run
uv run python super_metroid/scripts/run_room_problem.py route 0x9B5B 0x9E11 \
  --capability morph_ball --capability bombs --capability missiles \
  --capability spore_spawn_defeated --capability super_missiles
```

The generated graph/catalog, development states, reports, and screenshots are
gitignored local artifacts. The compact policies under
`policies/room_clears/` are curated source.
