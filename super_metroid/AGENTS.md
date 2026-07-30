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

## Layout

| Path | Role |
|------|------|
| `ram.py`, `assist.py`, `policy.py`, `progression.py`, `paths.py`, `room_timer.py` | Core package surface |
| `routes/continuous.py` | Power-on chain (Morph → Bombs → Spore → Supers) |
| `routes/runtime.py` | Shared session, report harness, integrity |
| `routes/*_controller.py` | Movement/combat only (no env ownership) |
| `rooms/` | Full-room graph, problem catalog, practice loop |
| `dev/` | Door-warp / boss probes (not continuous evidence) |
| `scripts/record/` `verify/` `probe/` `export/` `room/` | CLI entry points |
| `docs/` | `STATUS.md`, `plan.md`, `ASSIST_CONTRACT.md`, `ram_map.md` |
| `docs/routes/` | Accepted / working route boards |
| `docs/research/` | Path board, room catalog, legacy notes |
| `policies/room_clears/` | Curated room policies (tracked) |
| `maps/` | Generated graphs (gitignored except README) |
| `recordings/` | Baseline videos/manifests (gitignored) |
| `debug/` | Probe screenshots (gitignored) |
| `custom_integrations/SuperMetroid-Snes/` | Emulator integration + **anchor** states |
| `…/scratch/` | Ephemeral probe save-states |

## Immediate goal

**Primary: play KPDR continuous spine** (controller/policy). Door-warps
are topology diagnostics only — not route evidence.

1. **KPDR board:** [`docs/routes/ROUTE_KPDR.md`](docs/routes/ROUTE_KPDR.md)
   — authoritative continuous order (K→P→D→R). Spore Supers (no mockball);
   Alpha PB after Ice; **not** ship-first / early Pink PB.
2. **Path board:** [`docs/research/PATH_ROOM_BOARD.md`](docs/research/PATH_ROOM_BOARD.md)
   — 107 rooms / 199 hops; hop table is topology, not human KPDR order.
3. **★ Next play (pure):** compose the Kraid fight from the natural
   Warehouse→Hi-Jump→Warehouse→Kraid controller entry, take the rear door,
   and collect Varia. The full safer Warehouse suffix is 15,356 frames and
   collects Hi-Jump from the real PLM before Kraid.
4. **Dev topology (green):** `kpdr.py route-to-hijump` — 24 hops Big Pink →
   Hi-Jump room; anchors `dev_kpdr_*` / `dev_hijump_*`.
5. **Parked:** pure Pink PB maze; ship-first skip (not KPDR).
6. Boss fights only after natural entry exists on the continuous chain.

Tracker (chartable CSV/JSON/MD):
[`docs/routes/KPDR_TRACKER.csv`](docs/routes/KPDR_TRACKER.csv) · export
`scripts/export/kpdr_tracker.py`.

Status: [`docs/STATUS.md`](docs/STATUS.md). Plan: [`docs/plan.md`](docs/plan.md).
KPDR board: [`docs/routes/ROUTE_KPDR.md`](docs/routes/ROUTE_KPDR.md).

## Commands

### Continuous baselines

```bash
uv run python super_metroid/scripts/record/start_to_supers.py --no-video
uv run python super_metroid/scripts/record/start_to_supers.py
# Opt-in room timing (separate JSON under recordings/room_timings/; no integrity change)
uv run python super_metroid/scripts/record/start_to_supers.py --no-video --room-timing
uv run python super_metroid/scripts/record/start_to_spore_spawn.py --no-video
uv run python super_metroid/scripts/record/start_to_bombs.py --no-video
uv run python super_metroid/scripts/record/start_to_morph.py --no-video
```

### Path board + post-Super controller

```bash
uv run python super_metroid/scripts/export/path_room_board.py

# From natural_post_spore_spawn: Supers → Big Pink main, etc.
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main
uv run python super_metroid/scripts/probe/post_spore_pb.py --to crest
uv run python super_metroid/scripts/probe/post_spore_pb.py --to tunnel-floor

# KPDR K1: Big Pink main (controller)
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main

# Historical Kraid-before-Hi-Jump topology (dev door-warps; 24 hops)
uv run python super_metroid/scripts/probe/kpdr.py route-to-hijump --grant-hijump
uv run python super_metroid/scripts/probe/kpdr.py varia-to-hijump
uv run python super_metroid/scripts/probe/kpdr.py list

# Pure room controllers (no warp/write inside each segment)
uv run python super_metroid/scripts/probe/kpdr.py pure ghz-to-noob \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_kpdr_ghz.state
uv run python super_metroid/scripts/probe/kpdr.py pure noob-to-red \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_kpdr_noob.state
uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-hijump-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/red_to_warehouse_controller.state

# Chartable progress tracker
uv run python super_metroid/scripts/export/kpdr_tracker.py
```

Controllers: `routes/post_spore_controller.py`, `routes/kpdr_controller.py`.
KPDR plan: `docs/routes/ROUTE_KPDR.md`. Tracker: `docs/routes/KPDR_TRACKER.csv`.

### Topology skeleton (door-warp diagnostics only)

```bash
uv run python super_metroid/scripts/probe/route.py list
uv run python super_metroid/scripts/probe/route.py full
uv run python super_metroid/scripts/probe/route.py full-hybrid
uv run python super_metroid/scripts/probe/route.py phantoon-to-ridley
uv run python super_metroid/scripts/probe/route.py ridley-to-mb
uv run python super_metroid/scripts/probe/route.py late-full
```

Reports are labeled `developmentOnly: true`. Hop tables:
`maps/full_route_hops.json`, `maps/late_game_route_hops.json`. Runner:
`dev/route_dev.py`.

### Boss / late entry (development only)

```bash
uv run python -c "from super_metroid.dev.kraid_dev import run_kraid_fight; print(run_kraid_fight())"
uv run python super_metroid/scripts/probe/phantoon.py collect-pb
uv run python super_metroid/scripts/probe/phantoon.py capture-entry
uv run python super_metroid/scripts/probe/phantoon.py ship-route
uv run python super_metroid/scripts/probe/mother_brain.py capture-mb
uv run python super_metroid/scripts/probe/mother_brain.py spray-mb --frames 3600
uv run python super_metroid/scripts/probe/mother_brain.py run-escape --frames 12000
```

### Room practice

```bash
uv run python super_metroid/scripts/export/room_problems.py
uv run python super_metroid/scripts/room/run_problem.py ready --run
```

### Room timing (stock ROM / emulator frames)

```bash
uv run python super_metroid/scripts/probe/room_timer.py self-check
uv run python super_metroid/scripts/probe/room_timer.py offline -i samples.json
# docs: docs/ROOM_TIMER.md  ·  core: room_timer.py
```

## Dev traps

- `dev/common.py` owns `boot_from_state`, `door_warp`, `place_samus`,
  `apply_dev_survivability`, `enemy_hps`, `select_weapon`, `save_dev_state`.
  Reuse them; do not re-implement warp settle.
- Door-warp settle must wait for **game state 8** (not merely `ordinary`
  phase) — multi-screen loads sit in state 11 for 50–100+ frames.
- High WRAM (events/boss bits at `$7E:D820+`) must use `read_bank7e_wram` /
  `write_wram_u8` — raw `env.get_ram()[0xD820]` is open-bus garbage.
- Save **named anchors** under `custom_integrations/SuperMetroid-Snes/`;
  dump probe noise into `scratch/`.
- Prefer room/door/inventory progress vectors over coordinate-only watchdogs.
- Keep the last successful full-run baseline; candidates use separate reports.
- Do not route an infinite bomb jump for the current KPDR suffix. The Hi-Jump
  return uses the intended left-shaft ledges plus ordinary bombs in the top
  morph tunnel; Charge needs a conventional return.
