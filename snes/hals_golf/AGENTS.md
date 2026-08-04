# Hal's Hole in One Golf — Agent Notes

Package `hals_golf` (disk: `snes/hals_golf/`; nested import root). Repo-wide
rules: [root AGENTS.md](../../AGENTS.md).

## Commands

```bash
# Cold-boot probe / human / autoplay
HEADLESS=1 ./run_bot.sh probe --frames 2000
./run_bot.sh play --state Title
./run_bot.sh play --autoplay --state Title
./run_bot.sh play --autoplay --state Hole1_Command --skip-bootstrap

# Clears + video
HEADLESS=1 ./run_bot.sh clear --state Title
HEADLESS=1 ./run_bot.sh clear --mode vs-hal --state Title
HEADLESS=1 PYTHONUNBUFFERED=1 ./run_bot.sh clear --state Title --video
./record_vs_hal_win.sh
./record_metal_clear.sh

# Pro bootstrap / HIO search / tests
./run_bot.sh play --autoplay --difficulty pro --state Title
HEADLESS=1 ./run_bot.sh search-hio --state Hole1_Command --max-candidates 25
./run_bot.sh list
uv run --frozen pytest tests -v
```

## Layout

| Path | Role |
|------|------|
| `hals_golf/core/` | RAM, scenes, actions, recovery |
| `hals_golf/tasks/` | mission, shot policy, menus, routes |
| `hals_golf/runtime/` | CLI, video, hio_search, bootstrap |
| `custom_integrations/HalsHoleInOne-Snes/` | Save states |
| `docs/STATUS.md`, `docs/metal_stroke.md` | Gate facts + metal calibration |

Nested import: workspace is `snes/hals_golf/`; package is
`snes/hals_golf/hals_golf/`. Root `conftest` / `repo.ensure_import_paths` put
the workspace on `sys.path`.

## Traps

- Human ↔ bot: `~` or L+R+SELECT via `retro_harness.PlaySession`; resume runs
  `StrokePlayMission.on_autopilot_resume` then restarts the current shot.
- F5 disk QuickSave; F7/F8 load.
- Keep clears on `DeterministicRoutePolicy`; HIO exploration is `search-hio`
  only — do not wire it into the mission clear path.
- Pro overlays in `tasks/routes/pro.py` stay empty until calibrated.
- Aim byte `0x10B1` is **not** the round total (see STATUS RAM table).

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/metal_stroke.md](docs/metal_stroke.md) ·
[README.md](README.md)
