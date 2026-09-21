# Agent Instructions — alttp

A Link to the Past opening-route workspace. Docs: `docs/STATUS.md`,
`docs/plan.md`, `docs/ARCHITECTURE.md`, `docs/ROOM_ENGINE.md`,
`docs/TRIGGER_HANDOFF.md`, `docs/routes/ROOM_WORK_QUEUE.md`. Session:
`.grok/skills/alttp-session/SKILL.md`. Tracker: `bd ready -l alttp -l spine`.
Residuals: `docs/tasks/` (one living file per claimed bead).
Sanctuary: `docs/tasks/HANDOFF-sanctuary-spine.md`. Dungeon-start pins
(dev, not continuous): `docs/tasks/HANDOFF-dungeon-starts.md`.

## Commands

```bash
uv run python alttp/scripts/setup_rom.py
# Optional local z3-json-data (gitignored): setup_z3_json_data.py

uv run python -m alttp.opening_route.catalog validate
uv run python -m alttp.opening_route.catalog emit
uv run python alttp/scripts/export_work_queue.py

SDL_VIDEODRIVER=dummy uv run python alttp/scripts/boot_to_castle.py --save
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_to_verified_tip.py
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_opening_spine.py --no-video
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_opening_spine.py --through room_50 --no-video
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_opening_spine.py --through room_72 --no-video
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/castle_to_sword.py --natural
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/castle_dungeon_prefix.py

# Preferred for new rooms (compact context)
uv run python alttp/scripts/room_engine.py list
uv run python alttp/scripts/room_engine.py show room_61
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/room_engine.py run room_61 \
  --edge west_to_0x60 --state CastleMain --overlay

uv run --frozen pytest alttp/tests -q
uv run python -m alttp.tas.fetch_refs --list && uv run pytest alttp/tests/test_tas.py -q
```

## Layout

| Path | Role |
|------|------|
| root | `ram`, `primitives`, `startup`, `overworld`, `session`, `paths` |
| `opening_route/` | Continuous trunk (prefer for new code) |
| `gauntlet/`, `romhack/` | Experiments only — not continuous claims |

## Dual track

**State-load / work-queue pins** are development only. **Clean** claims need
`--natural` from the real predecessor. Planner owns `docs/STATUS.md`. Continuous
tip is room `0x50`; do not overwrite `recordings/verified_tip_run.json` on a red.

## Traps

- Multi-truth anchors: RAM + map/Yaze + screenshot. Route ≠ approach ≠ trigger.
- Short state names; meanings in `opening_route.anchors.STATE_SEMANTICS`.
- Prefer `fight_nearby` / `move_path` over open-loop mega-macros.
- State-load runs are **dev only**; clean natural-chain claims need `--natural`
  with full acceptance.
- Leave proof is RAM + `alttp.screen_glance` leftover (`docs/tasks/glance-residual.md`, `docs/tasks/glance-72-residual.md`), not an MP4.
- Tip today: continuous through NW chamber `0x50`; `0x50` east and the `0x01`
  well are natural_entry. Next: small key onto `0x81` west → Zelda cell →
  escort. Drive from ROOM_WORK_QUEUE + escape_graph blockers only.
