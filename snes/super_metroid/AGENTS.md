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
| `custom_integrations/SuperMetroid-Snes/` | Anchors; probes → `scratch/` |
| `docs/` | STATUS, plan, routes, tasks, contracts |


**Room-policy layout / tip-extension prevention:** multi-hop → package from
day 1 (no megafiles), room-prefixed geometry, RLE as JSON data, shared
helpers, ≤~500 lines/file — checklist in
[`docs/tasks/PROCESS.md`](docs/tasks/PROCESS.md) § Room policy layout.

## Immediate goal

**Verified tip:** continuous power-on → Wave Beam (default `wave`,
**136,361f** ×2, room `0xADDE`, beams `0x1005`, items `0x3105`).
**History:** Speed **130,388f** ×2 and non-Spazer Bat Cave **122,304f** ×2
remain valid previous tips.
**Next:** Dual continuous `--to ice` stabilize (`rr-kxge`) — compose **LANDED**
(11 hops). **Single** continuous GREEN once (`ice_r3` 148192f room `0xA890`
beams `0x1007`); dual still flaky on Business floor climb (residual
`docs/tasks/rr-kxge-residual.md`). Pure floor→Gate dual GREEN 3255f×2; elev
891f. Ice pure stack **CLOSED** (`rr-dbu.11`); Wave→Business pure **CLOSED**
(`rr-vqv3`). **Not** STATUS-promoted without dual continuous green.

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
# Continuous default (wave) / named tips
uv run python snes/super_metroid/scripts/record/continuous.py --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to wave --no-video
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
uv run python snes/super_metroid/scripts/probe/shine_practice.py drill
uv run python snes/super_metroid/scripts/probe/shine_practice.py human --series ls_edge_v1
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py pure
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure
uv run python snes/super_metroid/scripts/record/guided_human.py --from west-ocean --name west_ocean_ws_human
```


## Dev traps

- Door-warp settle: wait for **game state 8** (not merely ordinary phase); state 11 can last 50–100+ frames.
- High WRAM (`$7E:D820+`): use `read_bank7e_wram` / `write_wram_u8` — raw `get_ram()[0xD820]` is open-bus garbage.
- Named anchors in `SuperMetroid-Snes/`; probe noise only in `scratch/`.
- Dual-track / door-warp / boss probes are **not** continuous evidence.
- Clean runs: `*_clean` stems only; never overwrite assisted `recordings/<tip>.json`.
- Prefer room/door/inventory progress vectors over coordinate-only watchdogs.
- Morph bombs are **X** while morph (not A).
- **Door entry speed/position matter** for TAS tech (speed-boost carry, shine,
  mockball, subpixel). Continuous transitions record
  `leave_kinematics` / `entry_kinematics`; use
  `door_kinematics.DoorKinematicsRequirement` or `StateRequirement` velocity/
  speed fields. Practice doorway bootstrap **zeros** momentum — never treat
  those fixtures as natural leave-speed evidence.
- **Shinespark store:** harness **B**=dash, **A**=activate, **DOWN**=store
  (VOD swaps A/B). After echoes=4, press DOWN **while still holding RIGHT**
  (`DOWN+RIGHT+B` ok). Idle or **B alone** dumps echoes **4→0 in one frame** —
  then crouch never arms `$0A68`. Drill: `shine_practice.py drill`.
  Full notes: `docs/tasks/SHINE_PRACTICE.md`.
- **Ceres elev escape:** Falling→elev mid-transition can still read **y≈139**;
  ordinary **gs=8 remaps to bottom y≈651**. Ledge pin **y=571** (pose 2 or
  137/138 left seat) — walk LEFT on ledge. Product shaft s2–s10 is
  **debris-phase** sensitive (TAS boot same pin, needs idle 14); use
  `_ceres_product_shaft_with_phase` — no hop thrash. Top: walk right to
  **pose 137 @ x211 y171**, LEFT+A 38 + LEFT to pad **x≈145 y75** → gs 32.
  BB elev: `$0E16` elev flag toggles/frame — product parity 1 first, then
  parity 0 / reactive. Notes: [`docs/plan.md`](docs/plan.md) § Ceres arm-pump.
- **Spazer + HJ pillar:** single-frame peak down-shot fails under Spazer;
  `play_hj_room_collect` multi-taps DOWN+X at peak (no equip strip).
- **Spazer continuous K4:** Business tip green; Cathedral Entrance Super door
  still desyncs under always-Spazer mainline — see `bd ready` / `rr-n2v`.
