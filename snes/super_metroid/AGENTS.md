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

**Tip:** continuous power-on → Bat Cave (default `bat_cave`, **122,304f**).
**Next pure:** Bat → Speed Hall from `scratch/post_bat_cave_continuous` /
`post_bubble_to_bat_pure`.

[`docs/STATUS.md`](docs/STATUS.md) · [`docs/plan.md`](docs/plan.md) ·
[`docs/routes/ROUTE_KPDR.md`](docs/routes/ROUTE_KPDR.md) ·
[`docs/tasks/PROCESS.md`](docs/tasks/PROCESS.md) ·
[`docs/tasks/QUEUE.md`](docs/tasks/QUEUE.md).

## Commands

From repo root (`snes/` on pythonpath → `import super_metroid` works).

```bash
# Continuous default (bat_cave) / named tips
uv run python snes/super_metroid/scripts/record/continuous.py --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to bat_cave --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to frog --no-video

# Early Spazer human wall-jump (guide on same window; see docs/tasks/EARLY_SPAZER_HUMAN.md)
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from below-spazer --route early-spazer --name spazer_human

# Last pure GREEN on tip path (Bubble → Bat); next open pure is Bat → Speed Hall
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state

uv run python snes/super_metroid/scripts/export/kpdr_tracker.py
./snes/super_metroid/scripts/dispatch_opencode.sh SM-K4-03
```

## Dev traps

- Door-warp settle: wait for **game state 8** (not merely ordinary phase); state 11 can last 50–100+ frames.
- High WRAM (`$7E:D820+`): use `read_bank7e_wram` / `write_wram_u8` — raw `get_ram()[0xD820]` is open-bus garbage.
- Named anchors in `SuperMetroid-Snes/`; probe noise only in `scratch/`.
- Dual-track / door-warp / boss probes are **not** continuous evidence.
- Clean runs: `*_clean` stems only; never overwrite assisted `recordings/<tip>.json`.
- Prefer room/door/inventory progress vectors over coordinate-only watchdogs.
