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
   after stairs (−63f). No W4 phase pad.
6. **Route contracts + natural predecessor evaluator (2026-08-01):** done.
   `smb.reactive_route` tracks declared successors and entry fingerprints;
   `smb.scripts.run_reactive_warp` runs stairs + reactive 1-2 and a tail in
   one environment. The drop-5 8-2 retime and natural-control 8-3/8-4
   repairs now finish in **21,643f**, **3/3 Clean power-on**, with no pad.
7. **Default fold + capture:** make the verified reactive seed the default
   reproducible fold, regenerate its source-owned artifact, and record the
   improved Clean run.

### C — Optional follow-ons

1. Silver/Gold runtime: natural-entry 4-2 and mushroom-cloud speed
2. Non-warp all-32-exit route
3. Transfer continuous fold patterns to SMB3 / platformer_common

For all-32, use `ROUTE_ALL_EXITS` contracts and the explicit missing-policy
coverage report as the source of truth. Do not use the stitch renderer as
completion evidence; each new stage needs a natural-entry controller before
it can count toward a Clean route.

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

- Best verified seed: `smb/models/smb_1_1_to_ending_reactive_83_84.json`
  (**21,643f**, −88f versus the M8 baseline). The source-owned default fold
  remains `smb_1_1_to_ending.json` (**21,731f**) pending the next milestone.
- Power-on: boot **350** + settle **16**
- Level1_1 continuous: settle **14**
- Rebuild: `uv run python -m smb.scripts.fold_continuous_policy`
