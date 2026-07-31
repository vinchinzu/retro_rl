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
Next: main hall B1 → Zelda follower → escort → Sanctuary.
Drive probes from `docs/routes/ROOM_WORK_QUEUE.md` and continuous-spine
blockers on `opening_route.escape_graph` only.

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

uv run --frozen pytest alttp/tests -q
```

## Contracts (quick)

- **Escape graph:** `alttp.opening_route.escape_graph` — continuous through
  main hall `0x61`; Zelda/Sanctuary legs planned.
- **Segments:** `alttp.opening_route.segment` — `castle_to_sword`,
  `sword_to_secret_entrance_clear`, `pocket_to_main_hall`.
- **Anchors:** `alttp.opening_route.anchors` — semantic multi-truth names.
- **Trigger handoff:** `docs/TRIGGER_HANDOFF.md` — hole/stairs/main-door
  solved; main hall → Zelda open.
- State-load runs are development-only; only `--natural` with full acceptance
  is a clean natural-chain claim.
