# Plan — Super Mario Bros. (NES)

## Goal

Verified continuous any% warp (power-on → World 8-4) — **achieved** (M8).

## Tracks

### A — Autonomous completion

1. M2–M7 — done (Clean power-on → ending, 3/3)
2. M8 verified capture — done (`warp_finish_poweron_m8_capture.json` + MP4)

### B — Optional follow-ons

1. Silver/Gold runtime: natural-entry 4-2 and mushroom-cloud speed
2. Non-warp all-32-exit route
3. Transfer continuous fold patterns to SMB3 / platformer_common

## Notes

- Seed: `smb/models/smb_1_1_to_ending.json` (**21,731f**, −274f)
- Power-on: boot **350** + settle **16**
- Level1_1 continuous: settle **14**
- Rebuild: `uv run python -m smb.scripts.fold_continuous_policy`
