# Agent Instructions — alttp

A Link to the Past opening-route workspace. Prefer local docs over root
history: `docs/STATUS.md`, `docs/plan.md`, `docs/ARCHITECTURE.md`,
`docs/TRIGGER_HANDOFF.md`.

## Package layout

| Path | Role |
|------|------|
| `alttp/` root | Core: `ram`, `primitives`, `startup`, `overworld`, `session`, `paths` |
| `alttp/opening_route/` | Continuous trunk: graph, segments, anchors, catalog, work queue, live hops |
| `alttp/gauntlet/` | Combat experiments only (empty shell until Sanctuary is clean) |
| `alttp/romhack/` | Editor/asset experiments only (empty shell) |

Compat shims keep old imports working (`alttp.escape_graph`,
`alttp.castle_to_sword`, …). Prefer `alttp.opening_route.*` for new code.

## Norms

- Store `.state` files under `custom_integrations/Zelda3-Snes/`.
- ROM: `roms/zelda3.sfc` (symlink from repo `roms/zelda3.sfc`).
- Headless probes: `SDL_VIDEODRIVER=dummy`.
- Live controller stack is `alttp.primitives` + `alttp.route_report`. Prefer
  `fight_nearby` / `move_path` over new open-loop mega-macros.
- Multi-truth anchors: RAM + map/Yaze + screenshot. Route ≠ approach ≠ trigger.
- State filenames are short; meanings live in `opening_route.anchors.STATE_SEMANTICS`
  (e.g. `HyruleCastleGrounds` ≠ secret-hole approach).
- Do not mix gauntlet/romhack work into continuous claims.
- Shared session façade: `alttp.session.AlttpSession` / `bind_env`.

## Immediate goal

Main hall entry is verified (pocket bush-cut → door → room `0x61`).
Main hall clear + west exit to `0x60` is **isolated** (graph edge
`main_hall_west_to_0x60`; map `maps/room_61.json` + `room_engine`).
Room `0x60` north → `0x50` is **isolated** (`room_60_north_to_0x50`).
Next: after `0x50` → Zelda follower → escort → Sanctuary.
Drive probes from `docs/ROOM_ENGINE.md` + `docs/routes/ROOM_WORK_QUEUE.md`
and continuous-spine blockers on `opening_route.escape_graph` only.

**Low-context room work:** use `scripts/room_engine.py show|run` — do not dump
full segment modules into agent context for B1 doors.

## Commands

```bash
uv run python alttp/scripts/setup_rom.py

# Optional: local z3-json-data (gitignored; never auto-downloaded)
uv run python alttp/scripts/setup_z3_json_data.py
uv run python -m alttp.z3_json_data status
uv run python -m alttp.z3_json_data list-regions --opening

# Opening-route catalog (z3 regions/connections + gameplay checkpoints)
uv run python -m alttp.opening_route_catalog validate
uv run python -m alttp.opening_route_catalog emit
# After a boot run, attach only real observed milestone facts:
uv run python -m alttp.opening_route_catalog emit \
  --from-boot-report alttp/recordings/boot_to_castle.json

# Sanctuary-path save-state work queue (60 states)
uv run python alttp/scripts/export_work_queue.py

SDL_VIDEODRIVER=dummy uv run python alttp/scripts/boot_to_castle.py --save

# Castle grounds → secret hole approach / uncle / sword (dev state default)
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/castle_to_sword.py
# Natural chain (title → grounds → segment):
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/castle_to_sword.py --natural
# Approach only (no bush-lift search):
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/castle_to_sword.py --approach-only

# Post-sword → secret-entrance clear (dev FighterSword; partial until Zelda)
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/sword_to_zelda.py

# Courtyard pocket → main castle door / room 0x61 (composes stairs exit)
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/pocket_to_main_hall.py
# Tiered probe / rediscovery:
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/probe_courtyard_main_door.py --tier scripts

# Room engine (preferred for new rooms — compact agent context)
uv run python alttp/scripts/room_engine.py list
uv run python alttp/scripts/room_engine.py show room_61
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/room_engine.py run room_61 \
  --edge west_to_0x60 --state CastleMain --overlay

# Main hall segment wrapper (same as room_engine west edge)
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/main_hall_to_zelda.py --overlay

uv run --frozen pytest alttp/tests -q
```

## Contracts (quick)

- **Escape graph:** continuous through main hall `0x61`; **isolated** west
  `0x61→0x60` and north `0x60→0x50`; Zelda/Sanctuary planned; primary vs
  `internal_key` tags.
- **Room engine:** `maps/room_XX.json` geometry authority +
  `opening_route.room_engine` clear/path/door; see `docs/ROOM_ENGINE.md`.
- **Segments:** continuous: `castle_to_sword`, `sword_to_secret_entrance_clear`,
  `pocket_to_main_hall`; `main_hall_to_zelda` thin wrap (partial until Zelda);
  planned: `escort_to_sanctuary`. Prefer `secret_entrance_clear` over
  historical `sword_to_zelda` name.
- **Room sense:** sprite AABBs, edge detect, overlay, `load_room_map`.
- **Anchors:** multi-truth names + approach windows; tip resolve includes
  isolated `room_60` after west exit.
- **Work queue:** continuous tip `room_61` → Zelda first; key/0x55 alternate.
- **Trigger handoff:** hole/stairs/main-door solved; west edge measured;
  B1 → Zelda open.
- State-load runs are development-only; only `--natural` with full acceptance
  is a clean natural-chain claim.
