# Agent Instructions — alttp

A Link to the Past opening-route workspace. Prior external work lived under
`../snes_editor/alttp/` (maps, editor, arena). Keep this checkout focused on
the shared `retro_harness.snes` seam.

## Norms

- Store `.state` files under `custom_integrations/Zelda3-Snes/`.
- ROM: `roms/zelda3.sfc` (symlink from repo `roms/zelda3.sfc`).
- Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/ram_map.md`.
- Headless probes: `SDL_VIDEODRIVER=dummy`.
- Live controller stack is `alttp.primitives` + `alttp.route_report`. Prefer
  `fight_nearby` / `move_path` over new open-loop mega-macros.
- Do not drag arena/romhack/asset-editor trees in until the opening route is
  stable.

## Immediate goal

Secret entrance clear is verified (sword → south chamber → stairs outdoors).
Next: courtyard pocket → main castle door → Zelda follower → Sanctuary.
Drive probes from `docs/routes/ROOM_WORK_QUEUE.md`.

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

# Post-sword → Zelda progress (dev FighterSword state; partial until follower==1)
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/sword_to_zelda.py

uv run --frozen pytest alttp/tests -q
```

Structured region/connection/item JSON: `alttp/docs/Z3_JSON_DATA.md`.
Opening-route catalog: `python -m alttp.opening_route_catalog` (artifact
`alttp/recordings/opening_route_catalog.json`). z3 node names are logic
labels, not stable-retro screen coordinates.

Escape graph (offline capability plan): `alttp.escape_graph`.
Castle→sword / sword→Zelda: `castle_to_sword.py`, `sword_to_zelda.py`.
State-load runs are development-only; only `--natural` with full acceptance is
a clean natural-chain claim.
