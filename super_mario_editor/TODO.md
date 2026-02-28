# Super Mario Editor TODO

Last updated: 2026-02-27

## Done

- [x] Render all 32 SMB world levels to PNG.
- [x] Export all 32 SMB world levels to JSON.
- [x] Remove unknown object/enemy labels from current export coverage.
- [x] Add/verify overlay layers: `enemies`, `objects`, `grid`, `coins`, `powerups`, `pipes`, `warps`.

## Next

- [ ] Add a deterministic visual regression check against emulator captures for representative regions in each area style (overworld, underground, castle, water).
- [ ] Add automated tests for overlay toggling to ensure each layer can be enabled/disabled independently without cross-layer bleed.
- [ ] Add a single command/script to batch render all 32 levels and produce an artifact summary (pass/fail + missing labels).
- [ ] Validate edge cases for object/enemy page control commands in all levels and lock with tests.
- [ ] Add editor-side per-layer visibility toggles (if missing in current UI) and persist toggle state.

## Cleanup

- [ ] Remove stale debug images/scripts from repo root or move them under a dedicated `debug/` directory.
- [ ] Document the expected map output locations and naming conventions in both CLI help and README.
