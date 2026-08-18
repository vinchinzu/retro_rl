# Harvest Agent Notes

Package `harvest` (disk: `snes/harvest/`; nested import root). Repo-wide rules:
[root AGENTS.md](../../AGENTS.md).

## Commands

```bash
./run_bot.sh play --autoplay --state latest
uv run python -m harvest.runtime.harvest_bot play --autoplay --power-on --end-of-spring
uv run python -m harvest.scripts.boot_probe --state Y1_Inside_House
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --state Y1_Inside_House --end-of-spring --out recordings/run_spring_month.json
uv run python -m unittest tests.test_day_plan_sequences tests.test_task_progress -v
./kickoff.sh
PYTHONPATH=.. uv run --project .. python -m retro_harness.editor_launcher harvest -- --state latest
```

Probes: [`docs/town_day1_recon.md`](docs/town_day1_recon.md) ·
[`docs/INTERACT.md`](docs/INTERACT.md) · [`docs/STATUS.md`](docs/STATUS.md).

## Traps

- Viewport BFS is ~16×14 tiles; hop targets ≤7 tiles or use `densify_waypoints`.
- Walkable tile IDs come from **recordings**, not static save-state dumps.
- Never hand-roll `Integrations.add_custom_path` — use
  `harvest.runtime.retro_setup.register_harvest_integration`.
- Tasks must not import `day_plan` / orchestrator (circular); shared facts live
  in `ram_catalog` / `tile_catalog` / `map_config`.
- Interact: scan an existing tape / UnlinkedText before recording. Face-walk
  is movement. Held forage is Eat/Don't eat, not Gotz. [`docs/INTERACT.md`](docs/INTERACT.md).
- Nested import: workspace `snes/harvest/`; package `snes/harvest/harvest/`.
  Root `conftest` / `repo.ensure_import_paths` put the workspace on `sys.path`.
- Prefer skill composition (`tasks/skills.py`) over new phase machines.
  Soft max ~1000 LOC / file — extract; do not grow monofile thrash.

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[docs/CODE_QUALITY_REVIEW.md](docs/CODE_QUALITY_REVIEW.md) ·
[docs/PLANNING_STACK.md](docs/PLANNING_STACK.md) · [docs/MILESTONES.md](docs/MILESTONES.md).
