# Agent Instructions — Super Metroid

Scripted full-clear package `super_metroid` (disk: `snes/super_metroid/`).
Docs: `CONTEXT.md`, `docs/STATUS.md`, `docs/plan.md`,
`docs/ASSIST_CONTRACT.md`, `docs/ram_map.md`. Session loop:
`.grok/skills/sm-session/SKILL.md`.
Tracker: `bd ready -l super_metroid -l spine`. Empty ready while a
spine bead is in_progress means continue the residual.

## Evaluation contract

- Primary: unlimited energy + ammo only
  ([`docs/ASSIST_CONTRACT.md`](docs/ASSIST_CONTRACT.md)); no free
  items/doors/map/bosses/capacity.
- Natural ending/credits required; final boss alone is not a clear.
- Clean track: [`docs/CLEAN_TRACK.md`](docs/CLEAN_TRACK.md);
  `*_clean` stems only — never overwrite assisted baselines.
- Dual-track: spine continuous vs room practice. Practice greens are
  not continuous evidence. Planner owns STATUS.

## Immediate goal

Living tip: `--to phantoon` ([STATUS.md](docs/STATUS.md),
[CONTEXT.md](CONTEXT.md)). Next spine bead: `rr-kw8t` Gravity.
Pin, checkbox, and probe CLI:
[`docs/tasks/rr-kw8t-residual.md`](docs/tasks/rr-kw8t-residual.md).

## Commands

```bash
# Watch (headed first when the user says watch). --headed is retro_harness.headed.
uv run python snes/super_metroid/scripts/probe/kpdr.py pure <hop> --source <pin> --headed
./snes/super_metroid/play <pin> --headed --assist-full

uv run python snes/super_metroid/scripts/record/continuous.py --to phantoon --no-video

bd ready -l super_metroid -l spine
```

`--no-video` on duals. Leave proof is RAM + dual JSON
(`super_metroid.hop_glance`), not an MP4. Dual CLI is the residual.

## Layout

| Path | Role |
|------|------|
| `routes/continuous.py`, `early_continuous.py`, `catalog.py` | Power-on chain + tip registry |
| `routes/kpdr/` | Pure movement/combat controllers |
| `routes/kpdr/spazer/` | Gold-standard multi-hop package |
| `tas/` | Sniq movies + harness replay (`docs/TAS_ADAPT.md`) |
| `custom_integrations/SuperMetroid-Snes/` | Anchors; probes → `scratch/` |

Multi-hop → package from day 1. Split a source file **before 1000 LOC**.
Continuous hops only via `tips.play_hops`.

## Traps

- Door-warp settle: wait for **game state 8**; state 11 can last 50–100+f.
- High WRAM (`$7E:D820+`): `read_bank7e_wram` / `write_wram_u8` — raw
  `get_ram()[0xD820]` is open-bus garbage.
- Named anchors in `SuperMetroid-Snes/`; probe noise only in `scratch/`.
  Overwrite `scratch/<hop>_dual.json`.
- Dual-track / door-warp / boss probes are practice, not continuous evidence.
- Morph bombs are **X** while morph (not A).
- Hop `side` is D-pad `LEFT`/`RIGHT`. Shoulders are `L`/`R`.
- Phase dumps are named scratch pins; leftover still is the next boot when
  the residual says so. RED dual keeps the controller.
