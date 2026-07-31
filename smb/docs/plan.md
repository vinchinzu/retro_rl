# Plan — Super Mario Bros. (NES)

## Goal

Verified continuous any% warp (power-on → World 8-4) — **achieved** (M8).
Next: claw back frames via hierarchical RLE polish + richer policies.

## Tracks

### A — Autonomous completion

1. M2–M7 — done (Clean power-on → ending, 3/3)
2. M8 verified capture — done (`warp_finish_poweron_m8_capture.json` + MP4)

### B — Optimizer architecture (in progress)

1. **Richer RAM / observations** — done: velocities, grounded, timer, camera
   in `smb/ram.py` + 210-dim `smb/obs.py` (legacy 189 still available).
2. **Hierarchical RLE-native search** — done: `platformer_common/rle_ops.py`,
   `rle_optimize.py`, CLI `smb.scripts.rle_polish`. Windows corrected:
   - `1-1-stairs` = **1050–1311** (not 1700–1974 castle idle)
   - `4-2-entry` / `4-2-full` (prefer fragment seed, not isolated Level4_2
     on continuous mid-route frames)
3. **Neuro upgrade** — done: deeper MLP / CNN head, discrete argmax combos,
   BC warm-start from RLE seed (`platformer_common/neuro.py`).
4. **Stairs micro (2026-07-30):** −63f 1-1 clear, 0 wall-slams.
5. **Reactive 1-2 (2026-07-31):** state-gated controller; World 4 in 3981f
   after stairs (−63f). Continuous promote blocked on **8-3** re-solve
   (no W4 pad). Then 4-2 RLE; optional CMA-ES / PPO.

### C — Optional follow-ons

1. Silver/Gold runtime: natural-entry 4-2 and mushroom-cloud speed
2. Non-warp all-32-exit route
3. Transfer continuous fold patterns to SMB3 / platformer_common

## Commands (polish)

```bash
# List bottleneck windows
uv run python -m smb.scripts.rle_polish --list-windows

# Hillclimb 1-1 stairs window on continuous seed
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.rle_polish --window 1-1-stairs --iters 400

# GA on 4-2 entry
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.rle_polish --window 4-2-entry --mode ga --gens 40
```

## Notes

- Seed: `smb/models/smb_1_1_to_ending.json` (**21,731f**, −274f)
- Power-on: boot **350** + settle **16**
- Level1_1 continuous: settle **14**
- Rebuild: `uv run python -m smb.scripts.fold_continuous_policy`
