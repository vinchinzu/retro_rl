# Agent Instructions — alttp

A Link to the Past opening-route workspace. Prior external work lived under
`../snes_editor/alttp/` (maps, editor, arena). Keep this checkout focused on
the shared `retro_harness.snes` seam.

## Norms

- Store `.state` files under `custom_integrations/Zelda3-Snes/`.
- ROM: `roms/zelda3.sfc` (symlink from repo `roms/zelda3.sfc`).
- Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/ram_map.md`.
- Headless probes: `SDL_VIDEODRIVER=dummy`.
- Do not drag arena/romhack/asset-editor trees in until the opening route is
  stable.

## Immediate goal

Power-on / boot state → title → fresh file → Link's House exit → controllable
on Hyrule Castle grounds (light-world screen `0x1B`).

## Commands

```bash
uv run python alttp/scripts/setup_rom.py

SDL_VIDEODRIVER=dummy uv run python alttp/scripts/boot_to_castle.py --save

uv run --frozen pytest alttp/tests -q
```
