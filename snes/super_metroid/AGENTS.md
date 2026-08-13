# Agent Instructions — Super Metroid

Scripted full-clear package `super_metroid` (disk: `snes/super_metroid/`).
Shared process: [`docs/FULL_RUN_PROCESS.md`](../../docs/FULL_RUN_PROCESS.md).

## Evaluation contract

- Primary: unlimited energy + ammo only ([`docs/ASSIST_CONTRACT.md`](docs/ASSIST_CONTRACT.md)); no free items/doors/map/bosses/capacity.
- Natural ending/credits required; final boss alone is not a clear.
- Clean track: [`docs/CLEAN_TRACK.md`](docs/CLEAN_TRACK.md); `*_clean` stems only — never overwrite assisted baselines.

## Layout

| Path | Role |
|------|------|
| `routes/continuous.py`, `early_continuous.py`, `catalog.py` | Power-on chain + tip registry |
| `routes/kpdr/` | Pure movement/combat controllers |
| `routes/kpdr/spazer/` | **Gold-standard** multi-hop package (mirror for new tips) |
| `tas/` | Sniq any%/100% movies + `snes12_rle` slices + harness replay/annotate (`tas/README.md`, `docs/TAS_ADAPT.md`) |
| `scripts/record/`, `probe/`, `export/` | Daily CLIs |
| `maps/maprando_room_*.json`, `maps/maprando_tech_catalog.json`, `run_splits.py`, `skill_bank.py`, `materialize.py` | Map Rando names + tech tree/builders + room PBs + hop bank ([docs/RUN_TIMING_AND_SKILL_BANK.md](docs/RUN_TIMING_AND_SKILL_BANK.md), [docs/TECH_TREE.md](docs/TECH_TREE.md)) |
| `custom_integrations/SuperMetroid-Snes/` | Anchors; probes → `scratch/` |
| `practice_repertoire.py`, `maps/practice_repertoire.json` | Practice-hack preset spine: human demos, policy tune/graduate, stitch, AP recovery |
| `docs/` | STATUS, plan, routes, tasks, contracts |


**Room-policy layout / tip-extension prevention:** multi-hop → package from
day 1 (no megafiles), room-prefixed geometry, RLE as JSON data, shared
helpers, ≤~500 lines/file — checklist in
[`docs/tasks/PROCESS.md`](docs/tasks/PROCESS.md) § Room policy layout.

## Immediate goal

**Verified tip:** continuous power-on → Ice Beam (default `ice`,
**148,167f** ×2, room `0xA890`, beams `0x1007`, items `0x3105`).
**History:** Wave **136,361f** ×2, Speed **130,388f** ×2, and non-Spazer Bat
Cave **122,304f** ×2 remain valid previous tips.
**Next:** K5 Alpha PB pure (`rr-dbu.8`) / post-Ice KPDR; optional ice demo
video. `rr-kxge` dual continuous **CLOSED** (Business floor climb harden +
cont-tuned 907 ladder). Residual `docs/tasks/rr-kxge-residual.md`.

**Spazer mainline:** Charge + Spazer on continuous spine through Speed dual.
Warehouse dual **89,416 + 90,904f** is a promoted prefix. Details:
[`docs/STATUS.md`](docs/STATUS.md).

**Work tracker:** monorepo **bd (beads)** — `bd ready -l super_metroid`.
Product evidence stays in STATUS / MILESTONES (not beads alone).

[`docs/STATUS.md`](docs/STATUS.md) · [`docs/plan.md`](docs/plan.md) ·
[`docs/routes/ROUTE_KPDR.md`](docs/routes/ROUTE_KPDR.md) ·
[`docs/tasks/PROCESS.md`](docs/tasks/PROCESS.md) ·
[`docs/tasks/QUEUE.md`](docs/tasks/QUEUE.md).

## Reports vs video (read this)

Continuous runs write a full machine report JSON under `recordings/` for
integrity/diff. **Do not dump that JSON into chat or treat it as the demo.**

| Artifact | Role |
|----------|------|
| `recordings/<tip>.json` | Machine integrity (loads/prog/deaths, splits) — **on disk only** |
| `recordings/<tip>.mp4` | **Human-readable proof** — watch this |
| CLI stdout | Short GREEN/RED summary (tip, frames, room, beams, tail splits, paths) |

```bash
# Integrity-only (fast; no mp4)
uv run python snes/super_metroid/scripts/record/continuous.py --to warehouse --no-video

# Demo / publishable proof — always with video for human review
uv run python snes/super_metroid/scripts/record/continuous.py --to warehouse \
  --video snes/super_metroid/recordings/warehouse_with_charge.mp4 \
  --report snes/super_metroid/recordings/warehouse_with_charge.json \
  --video-start zebes

# Charge prefix only (Below Spazer with Charge)
uv run python snes/super_metroid/scripts/record/continuous.py --to below_spazer \
  --video snes/super_metroid/recordings/below_spazer_with_charge.mp4 \
  --report snes/super_metroid/recordings/below_spazer_with_charge.json
```

When reporting a continuous result to a human: quote **frames, room, beams,
integrity flags, and the `.mp4` path** — not a pasted JSON body.

## Commands

From repo root (`snes/` on pythonpath → `import super_metroid` works).

```bash
# Practice ROM + repertoire spine (human + bot policy/stitch/AP recovery)
uv run python snes/super_metroid/scripts/setup_practice_rom.py
uv run python -m super_metroid.practice_repertoire --route          # stitch order
uv run python -m super_metroid.practice_repertoire --policy-board   # tune/graduate
uv run python -m super_metroid.practice_repertoire --recovery 0x9E9F --items 0x0004
# docs/PRACTICE_ROM.md

# Continuous default (ice) / named tips
uv run python snes/super_metroid/scripts/record/continuous.py --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to ice --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to wave --no-video  # previous tip
uv run python snes/super_metroid/scripts/record/continuous.py --to speed --no-video  # previous tip
uv run python snes/super_metroid/scripts/record/continuous.py --to frog --no-video

# Early Spazer human wall-jump (guide on same window; see docs/tasks/EARLY_SPAZER_HUMAN.md)
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from below-spazer --route early-spazer --name spazer_human

# Pure greens on Speed path (sources under scratch/)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bat-cave-to-speed-hall \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bat_cave_continuous.state
uv run python snes/super_metroid/scripts/probe/kpdr.py pure speed-hall-to-speed \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bat_cave_to_speed_hall_pure.state

uv run python snes/super_metroid/scripts/export/kpdr_tracker.py
bd ready -l super_metroid
./snes/super_metroid/scripts/dispatch_opencode.sh SM-K4-03

# Area-basemap CoG paths (pixel-aligned; same-room segments only)
uv run python -m super_metroid.map_viewer serve --open --export-defaults
uv run python -m super_metroid.map_viewer export-path tasks/parlor_left_human.json

# TAS movie replay + WRAM annotate (pose/x,y/vel/rooms; no L+R sanitize)
uv run python -m super_metroid.tas.replay --list-slices
uv run python -m super_metroid.tas.replay --slice sniq_any_menu --annotate --series-stride 1
# Zebes resync: product → Landing + Sniq movie body (Landing→Parlor @ movie_start=15000)
uv run python -m super_metroid.tas.resync --to landing --movie-start 15000 --body 12000
# Full any%/100% (long; desync mid-Ceres — pins + extract board still useful)
# --slice sniq_any_full|sniq_100_full --series-stride 8
uv run python -m super_metroid.tas.extract_hops \
  snes/super_metroid/recordings/tas_import/sniq_100_full
uv run python -m super_metroid.tas.extract_hops --list-stages

# YouTube KPDR reference (gitignored refs/yt_reference/; default Kentroid TFsGVxQReMw)
uv run python snes/super_metroid/scripts/tools/yt_ref.py status
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk \
  --start 1338 --end 1351 --name moat_shinespark --spark
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk --segment-id k0_ceres
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk --segment-id k2_spazer
# Ceres phase seed (tracked): policies/early_game/ceres_kentroid_spans.json

# Shinespark practice (Landing Site) + K6 Moat/West pure — see docs/tasks/SHINE_PRACTICE.md
# Store trap: releasing RIGHT (B alone/idle) dumps echoes 4→0 in 1f; DOWN while still holding RIGHT.
# Short charge: magic-frame dash (NTSC 25/50/70/85) or stutter — charge_mode full|short|stutter
uv run python snes/super_metroid/scripts/probe/shine_practice.py drill
uv run python snes/super_metroid/scripts/probe/shine_practice.py human --series ls_edge_v1
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py pure
# Product WO → green Super WS 0xCA08 (natural Moat handoff; stutter charge)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure-ws
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py watch-ws
# Compose Kihunter/Moat → Moat spark → over-ocean → WS pin (then ship record)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py chain-ws
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py --preset moat-to-ws
# Edge bowling practice (0xC98E; not product WS)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py short-charge --mode stutter
# Human record: optional WO practice, or ship free-record from product WS pin
uv run python snes/super_metroid/scripts/record/guided_human.py --from west-ocean --name west_ocean_ws_human
uv run python snes/super_metroid/scripts/record/guided_human.py --from ws-entrance --name ws_ship_human
uv run python snes/super_metroid/scripts/record/practice_takes.py --segment ws-entrance --series ws_ship_v1
# Bot-beat Phantoon from human entry end → post_phantoon_defeated pin
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py strategy --state ws_ship_human_end
uv run python snes/super_metroid/scripts/record/guided_human.py --from post-phantoon --name gravity_path_human --no-guide
# Bubble Save practice (0xB0DD items 0x1105) — live EARLY/LATE walljump frames_off
# SELECT+L2 reload pin (CP1 seeded) · SELECT+R2 mid seat · R hard reset · F5 take
uv run python snes/super_metroid/scripts/probe/bubble_save_practice.py
./snes/super_metroid/play bubble-save full_start_v1   # free-record Save → climb
./snes/super_metroid/play bat-cave full_start_v1      # continue after Bat 0xB07A
./snes/super_metroid/play wave full_start_v1          # Wave 0xADDE beams 0x1005 → Ice
./snes/super_metroid/play alpha-pb full_start_v1      # Alpha PB → Moat / Phantoon
./snes/super_metroid/play grapple full_start_v1       # Grapple 0xAC2B (0x7125) → Main Street
./snes/super_metroid/play main-street full_start_v1   # Main Street 0xCFC9 (0x7125) → Maridia
./snes/super_metroid/play plasma-beam full_start_v1   # Plasma Room 0xD2AA (0x7325 / beams 0x100F)
./snes/super_metroid/play gt full_start_v1            # Golden Torizo 0xB283 left door (0x7325 / 0x100F)
./snes/super_metroid/play metal-pirates full_start_v1 # Metal Pirates 0xB62B right door (0x732F / Screw)
./snes/super_metroid/play post-ridley full_start_v1   # Ridley Tank 0xB698 post fight + tank (0x732F)
./snes/super_metroid/play --pb full_start_v1          # RTA board (segment timing stitch)
# DC→Wave re-stitch practice (docs/tasks/DC_MISSILE_WAVE_HUMAN.md)
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment dc-missile-wave --series wave_dc_v1
# Post-Gravity Caterpillar tail (0xA322 items 0x3125) → Grapple + Maridia free-record
uv run python snes/super_metroid/scripts/record/guided_human.py --from post-gravity --name maridia_grapple_human --no-guide
# Post-Grapple (items 0x7125) → Maridia Main Street; F6 mid-run pins, anchors ON by default
uv run python snes/super_metroid/scripts/record/guided_human.py --from post-grapple --name maridia_main_street_human --no-guide
# Main Street locked pin → deeper Maridia / Botwoon free-record
uv run python snes/super_metroid/scripts/record/guided_human.py --from main-street --name maridia_botwoon_path_human --no-guide
# Post-Space Jump (0xD9AA items 0x7325) or post-Draygon Precious — next segment
uv run python snes/super_metroid/scripts/record/guided_human.py --from post-space-jump --name post_sj_exit_human --no-guide
uv run python snes/super_metroid/scripts/record/guided_human.py --from post-draygon --name maridia_exit_human --no-guide
# LN Main Hall (items 0x7327 beams 0x100F) → Ridley / Screw free-record
uv run python snes/super_metroid/scripts/record/guided_human.py --from main-hall --name post-main-hall --no-guide
# Post-boss Landing Site (items 0x732F, all 4 bosses) → G4 / Tourian free-record
uv run python snes/super_metroid/scripts/record/guided_human.py --from post-bosses --name g4_tourian_human --no-guide
# Offline hop inventory + hop bodies + bank candidates
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/post-main-hall.json --materialize --bank --summary
# Hop open-loop from live pin; multi-hop compose (pin→body chain)
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/full_start_v1.json --hop 0 --dual
uv run python snes/super_metroid/scripts/tools/compose_human_hops.py \
  snes/super_metroid/tasks/full_start_v1.json --dual
# Free-record wrapper (archives prior take; materialize + bank on F5)
./snes/super_metroid/play
./snes/super_metroid/play --compose full_start_v1
```

**Human long takes:** full button tape + live room/item anchors + F6 pins under
`tasks/<name>_anchors/`. F5 runs `materialize_take` (settled hops, hop bodies
under `*_hops/`, `*_run_timing.json`) unless `--no-materialize`. Reusing
`--name` archives prior tape to `tasks/<name>_segments/sN/`. Open-loop unit is
**hop from live pin** (or hop-compose chain); do not invent mid pins via
multi-minute full-tape replay. Library: `super_metroid.human_tape` +
`materialize`. Pipeline: [`docs/tasks/HUMAN_TAPE_PIPELINE.md`](docs/tasks/HUMAN_TAPE_PIPELINE.md);
board: `tasks/LATE_SPINE_HOP_BOARD.json` via `scripts/tools/build_late_spine_board.py`.



## Dev traps

- Door-warp settle: wait for **game state 8** (not merely ordinary phase); state 11 can last 50–100+ frames.
- High WRAM (`$7E:D820+`): use `read_bank7e_wram` / `write_wram_u8` — raw `get_ram()[0xD820]` is open-bus garbage.
- Named anchors in `SuperMetroid-Snes/`; probe noise only in `scratch/`.
- Dual-track / door-warp / boss probes are **not** continuous evidence.
- Clean runs: `*_clean` stems only; never overwrite assisted `recordings/<tip>.json`.
- Prefer room/door/inventory progress vectors over coordinate-only watchdogs.
- Morph bombs are **X** while morph (not A).
- **D-pad vs shoulders:** `LEFT`/`RIGHT` walk; `L`/`R` are shoulders (aim /
  arm-pump). Never use `L` as a hop side. Constants:
  `SNES_DPAD_LEFT` / `SNES_SHOULDER_L` in `retro_harness.controls`.
- **Door entry speed/position matter** for TAS tech (speed-boost carry, shine,
  mockball, subpixel). Continuous transitions record
  `leave_kinematics` / `entry_kinematics`; use
  `door_kinematics.DoorKinematicsRequirement` or `StateRequirement` velocity/
  speed fields. In-room jumps use `takeoff.TakeoffWindow` / `PlatformHop`
  (same matcher) — do not invent a per-room hop type or N-frame runup.
  Practice doorway bootstrap **zeros** momentum — never treat
  those fixtures as natural leave-speed evidence.
- **Shinespark store:** harness **B**=dash, **A**=activate, **DOWN**=store
  (VOD swaps A/B). After echoes=4, press DOWN **while still holding RIGHT**
  (`DOWN+RIGHT+B` ok). Idle or **B alone** dumps echoes **4→0 in one frame** —
  then crouch never arms `$0A68`. Drill: `shine_practice.py drill`.
  **Short charge:** boost counter ticks only on magic frames (NTSC 25/50/70/85)
  while dash+forward held — spark still full speed after store. Stutter prefix
  ≈141–156 px on LS/WO spit. Skill: `charge_until_boost(..., mode="stutter")`.
  WO pure GREEN with `--charge-mode short|stutter`; Moat hop still needs `full`.
  Full notes: `docs/tasks/SHINE_PRACTICE.md`.
- **Ceres elev escape:** Falling→elev mid-transition can still read **y≈139**;
  ordinary **gs=8 remaps to bottom y≈651**. Ledge pin **y=571** (pose 2 or
  137/138 left seat) — walk LEFT on ledge. Product shaft s2–s10 is
  **debris-phase** sensitive. Product fight is tail-tank (not wait). Elev
  shaft is a platform hop (`elev_escape.CeresShaftClimb`) — not the old
  phase-idle s2–s10 tape. Top: walk right to
  **pose 137 @ x211 y171**, LEFT+A 38 + LEFT to pad **x≈145 y75** → gs 32.
  BB elev: `$0E16` elev flag toggles/frame — product parity 1 first, then
  parity 0 / reactive. Notes: [`docs/plan.md`](docs/plan.md) § Ceres arm-pump.
- **Spazer + HJ pillar:** single-frame peak down-shot fails under Spazer;
  `play_hj_room_collect` multi-taps DOWN+X at peak (no equip strip).
- **Spazer continuous K4:** Business tip green; Cathedral Entrance Super door
  still desyncs under always-Spazer mainline — see `bd ready` / `rr-n2v`.
