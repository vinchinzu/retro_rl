# Plan — Super Mario Bros. 3 (NES)

## Goal

Advance from a verified World 1-1 clear toward continuous World 1 and full-game scripted completion.

## Next milestones

1. **World 1-3** — natural-entry clear from post-1-2 map state (`AfterLevel2`).
2. **World 1 chain** — 1-1 through fortress without warp assists.
3. **M2 instrumentation** — expand RAM (cards, inventory, stage id) as needed for later worlds.
4. **Full-game route** — world map graph + per-stage policies.

## Bottleneck

World 1-3 natural-entry clear.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- 1-1/1-2 policies live in `smb3/policies/level1_{1,2}.json`; re-optimize from
  natural entry if boot/map timing changes.
