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
| `scripts/record/`, `probe/`, `export/` | Daily CLIs |
| `custom_integrations/SuperMetroid-Snes/` | Anchors; probes → `scratch/` |
| `docs/` | STATUS, plan, routes, tasks, contracts |

## Immediate goal

**Verified tip:** continuous power-on → Speed Booster (default `speed`,
**130,388f** ×2, room `0xAD1B`, beams `0x1004`, items `0x3105`).
**History:** non-Spazer Bat Cave **122,304f** ×2 remains a valid previous tip.
**Next:** stabilize wave after Speed (`rr-07b`); pure Speed return → Bubble
(`rr-g4i`). Do not claim pure Speed return yet.

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
# Continuous default (speed) / named tips
uv run python snes/super_metroid/scripts/record/continuous.py --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to speed --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to bat_cave --no-video  # previous tip
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

# YouTube KPDR reference (gitignored refs/yt_reference/; default Kentroid TFsGVxQReMw)
uv run python snes/super_metroid/scripts/tools/yt_ref.py status
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk \
  --start 1338 --end 1351 --name moat_shinespark --spark
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk --segment-id k0_ceres
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk --segment-id k2_spazer
# Ceres phase seed (tracked): policies/early_game/ceres_kentroid_spans.json```


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
- **Ceres elev escape:** Falling→elev mid-transition can still read **y≈139**;
  ordinary **gs=8 remaps to bottom y≈651**. Ledge pin **y=571 pose=2** — walk
  LEFT on ledge (not blind product s0). Top: walk right to **pose 137 @ x211
  y171**, then product LEFT+A 38 + LEFT to pad **x≈145 y75** → gs 32. BB elev
  later: `$0E16` elev flag toggles/frame — 1f parity before seed if Ceres is
  odd-frames early. Notes: [`docs/plan.md`](docs/plan.md) § Ceres arm-pump.
- **Spazer + HJ pillar:** Spazer multi-shot does **not** open the Hi-Jump room
  pillar with the classic peak down-shot (Charge/power OK). `play_hj_room_collect`
  temporarily clears equip bit `0x0004` for the pillar only (ownership unchanged).
- **Spazer continuous K4:** Business tip green; Cathedral Entrance Super door
  still desyncs under always-Spazer mainline — see `bd ready` / `rr-n2v`.
