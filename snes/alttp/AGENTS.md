# Agent instructions: alttp

USA vanilla opening route. Docs: `docs/STATUS.md` (planner owns it),
`docs/plan.md`, `docs/ARCHITECTURE.md`, `docs/ROOM_ENGINE.md`,
`docs/TRIGGER_HANDOFF.md`, `docs/Z3_JSON_DATA.md`, `docs/ram_map.md`.
Session: `.grok/skills/alttp-session/SKILL.md`.
Tracker: `bd ready -l alttp -l spine`.
Assist contract: [docs/ASSIST_CONTRACT.md](docs/ASSIST_CONTRACT.md).
Living residual (overwrite this file only): `docs/tasks/residual.md`.
Skull Woods `rr-lvnu` detail stays in `docs/tasks/SkullWoodsEntry-residual.md`. No indoor pin yet.

Commands below assume cwd `snes/`.

## Commands

```bash
uv run python alttp/scripts/setup_rom.py
uv run python alttp/scripts/setup_z3_json_data.py

uv run python -m alttp.opening_route.catalog validate
uv run python alttp/scripts/export_work_queue.py

SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_to_verified_tip.py
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_opening_spine.py --through room_50 --no-video
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/castle_to_sword.py --natural
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/castle_dungeon_prefix.py

export QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
uv run python alttp/scripts/audit_power_on.py controller \
  --through sanctuary --no-survival --out alttp/recordings/clean_escape_20261002/controller.json
uv run python alttp/scripts/audit_power_on.py replay \
  --tape alttp/recordings/clean_escape_20261002/controller.tape.npz \
  --out alttp/recordings/clean_escape_20261002/replay.json

uv run python alttp/scripts/room_engine.py list
uv run python alttp/scripts/room_engine.py show room_61
PYTHONPATH="..:.:../nes" uv run python -m alttp.scripts.dungeon_board
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/room_engine.py run room_61 \
  --edge west_to_0x60 --state CastleMain

uv run pytest alttp/tests -q
```

## Layout

| Path | Role |
|------|------|
| package root | `ram`, `primitives`, `startup`, `overworld`, `session`, `paths` |
| `opening_route/` | Continuous trunk |
| `z3-json-data/` | Committed JSON dump (no LICENSE, no nested git) |
| `refs/z3-json-data/` | Gitignored live pin. Do not commit it. Do not delete the dump. |
| `gauntlet/`, `romhack/` | Shells. Not continuous claims. |

## Traps

- Ladder: planned, then isolated, then natural_entry, then continuous. A save-state pin is not continuous.
- C1 proof is the Clean Sanctuary power-on tape replay. Controller forecasts
  are development evidence; quote the replay's `route_goal_ok`, `clean_chain`,
  loads, writes and desync. The four-stage `full_tip` helper remains the
  historical room-`0x50` checkpoint. Keep its report intact on a red.
- Sanctuary acceptance is room `0x12`, control ready, follower `$F3CC == 0`,
  progress `$F3C5 == 2`, and `$F3C6 & 0x04`. Audit defaults to no assists.
  Development lookahead restores the same frame; committed replay has zero
  loads, writes and refill assists. Explicit Survival runs stay development.
- Leave proof is a RAM glance (`alttp.screen_glance`), not an MP4.
- Loader prefers `refs/z3-json-data/` when that checkout exists, else the committed dump. Import never downloads.
- Stair destination settle is `DEST_SETTLE_MAX_FRAMES` (480). Do not change the global `settle_control` default (240).
- Clean claims need `--natural` from the real predecessor. Work-queue pins are dev only.
- Geometry stays in `maps/`. Do not copy approach xy into Python.
- A named sitting is one room controller and one predecessor replay, then stop. Budget is in the session skill. `bd` has no local database.
- State-name meanings: `opening_route.anchors.STATE_SEMANTICS`.
