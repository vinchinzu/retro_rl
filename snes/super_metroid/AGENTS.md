# Agent Instructions — Super Metroid

Scripted full-clear package `super_metroid` (disk: `snes/super_metroid/`).
Shared process: [`docs/FULL_RUN_PROCESS.md`](../../docs/FULL_RUN_PROCESS.md).
Tip + history: [`docs/STATUS.md`](docs/STATUS.md) (not this file).
Work: `bd ready -l super_metroid`. Plan / queue:
[`docs/plan.md`](docs/plan.md) · [`docs/tasks/QUEUE.md`](docs/tasks/QUEUE.md).

## Evaluation contract

- Primary: unlimited energy + ammo only ([`docs/ASSIST_CONTRACT.md`](docs/ASSIST_CONTRACT.md)); no free items/doors/map/bosses/capacity.
- Natural ending/credits required; final boss alone is not a clear.
- Clean track: [`docs/CLEAN_TRACK.md`](docs/CLEAN_TRACK.md); `*_clean` stems only — never overwrite assisted baselines.

## Layout

| Path | Role |
|------|------|
| `routes/continuous.py`, `catalog.py` | Power-on chain + tip registry |
| `routes/kpdr/` (+ `spazer/`) | Pure hops; Spazer package is gold-standard |
| `scripts/record/`, `probe/`, `export/` | Daily CLIs |
| `custom_integrations/SuperMetroid-Snes/` | Anchors; probes → `scratch/` |
| `docs/` | STATUS, plan, routes, tasks, contracts |

Room-policy checklist: [`docs/tasks/PROCESS.md`](docs/tasks/PROCESS.md) § Room policy.

## Commands

From repo root (`import super_metroid`). Long CLI catalog:
[`docs/plan.md`](docs/plan.md) · [`docs/tasks/QUEUE.md`](docs/tasks/QUEUE.md).

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to ice --no-video
uv run python snes/super_metroid/scripts/probe/kpdr.py --help
uv run python snes/super_metroid/scripts/probe/shine_practice.py drill
./snes/super_metroid/play
bd ready -l super_metroid
```

KPDR hops: `scripts/probe/kpdr.py`. Shine store / Moat / WO:
[`docs/tasks/SHINE_PRACTICE.md`](docs/tasks/SHINE_PRACTICE.md). Human tape:
[`docs/tasks/HUMAN_TAPE_PIPELINE.md`](docs/tasks/HUMAN_TAPE_PIPELINE.md).

## Traps

- Door-warp settle: wait **game state 8** (not ordinary phase); state 11 can last 50–100+ frames.
- High WRAM (`$7E:D820+`): `read_bank7e_wram` / `write_wram_u8` — raw `get_ram()[0xD820]` is open-bus garbage.
- Clean: `*_clean` stems only; never overwrite assisted `recordings/<tip>.json`. Dual-track / door-warp / boss probes are **not** continuous evidence.
- **D-pad vs shoulders:** `LEFT`/`RIGHT` walk; `L`/`R` are shoulders. Never use `L` as a hop side. `SNES_DPAD_LEFT` / `SNES_SHOULDER_L` in `retro_harness.controls`.
- **Shinespark store:** harness **B**=dash, **A**=activate, **DOWN**=store. After echoes=4, DOWN **while still holding RIGHT**. Idle or **B alone** dumps echoes **4→0 in 1f**. Drill: `shine_practice.py drill`.
- Morph bombs are **X** while morph (not A). Ceres elev / door kinematics / Spazer HJ+K4: [`docs/plan.md`](docs/plan.md).
