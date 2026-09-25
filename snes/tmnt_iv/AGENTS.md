# Agent instructions: tmnt_iv

SNES TMNT IV. Shared combat helpers: `retro_harness.combat` /
`segment_runner`. Docs: `CONTEXT.md`, `docs/ARCHITECTURE.md`,
`docs/STATUS.md`, `docs/plan.md`, `docs/ASSIST_CONTRACT.md`,
`docs/ram_map.md`. Tracker: `bd ready -l tmnt_iv`.

## Commands

```bash
uv run python -m tmnt_iv.scripts.setup_rom
uv run python -m tmnt_iv.scripts.boot_probe

SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_clean --stage 1 --suite

uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run
uv run python -m tmnt_iv.scripts.record_full_hard_run
uv run python -m tmnt_iv.scripts.record_full_hard_run --clean --dry-run

uv run pytest tmnt_iv/tests -q -m rom
```

Segment and bridge entry points are `run_segment --stage N` and
`run_bridge --to {2,3}`. Clean artifacts use
`retro_harness.artifacts.clean_artifact_stem` and must not overwrite
assisted `tmnt_iv_full_hard_*` files.

## Traps

- START after the Stage 1 HUD pauses the game.
- Special (A) drains HP. Do not use it.
- Global pizza seek soft-locks Skull and Crossbones. Big Apple may seek. Other stages take underfoot or between-wave pizza only.
- Blind `RIGHT+Y` stutters. `pickup_every=0`. Pizza seek owns boxes.
- Stage 1 hazard jump-dodge stays offline. It caused Clean mid-wave deaths.
- Slash spin dodge stays at adx 52. Porting 40 regressed the continuous run.
- A KEEP trace is not production. Do not rewrite Slash from one.
- Sewer dumpster skip and Starbase right-rail skip are different stalls. Starbase holds RIGHT. Do not skip x=126.
- `raph_starbase_close_gap` period below 4 times out.
- One trial loop: `run/trial.py` `run_trial`. Add a spec. Do not clone `run_stageN_segment.py`.
- The default continuous CLI stays assisted. Clean is parallel and is not the STATUS gate.
