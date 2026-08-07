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
2. **Hierarchical RLE-native search** — done: `retro_harness/platformer/rle_ops.py`,
   `rle_optimize.py`, CLI `smb.scripts.rle_polish`. Windows corrected:
   - `1-1-stairs` = **1050–1311** (not 1700–1974 castle idle)
   - `4-2-entry` / `4-2-full` (prefer fragment seed, not isolated Level4_2
     on continuous mid-route frames)
3. **Neuro upgrade** — done: deeper MLP / CNN head, discrete argmax combos,
   BC warm-start from RLE seed (`retro_harness/platformer/neuro/`).
4. **Stairs micro (2026-07-30):** −63f 1-1 clear, 0 wall-slams.
5. **Reactive 1-2 (2026-07-31):** state-gated controller; World 4 in 3981f
   after stairs (−63f). No W4 phase pad.
5b. **1-2 UG polish (2026-08-05):** `polish_1_2_ug` delete+hold on
    control-relative underground RLE → natural 1-2 **2032f** / W4 **3943f**
    (−38f). Surface still reactive RIGHT/DOWN.
5c. **1-2 W4 pipe top-land (2026-08-05):** late UG suffix rewrite
    (`polish_1_2_warp_pipe`) → clean pipe-lip enter, natural 1-2 **1973f** /
    W4 **3884f** (−59f vs post-delete, −159f vs original reactive 3981).
5d. **Natural 4-1 retime (2026-08-05):** idle to 4-1 control, resume cont
    index **218** (no W4 pad). 4-1 split **2335→2314** (−21); unblocks
    4-2@8962 and 8-1@12628.
5e. **Control-relative 4-2 retime (2026-08-05):** freeze source at 4-1
    score/load, idle to 4-2 control, resume cont **2487**. Lead-idle trim
    and `fast_w8` splice both fail on this phase.
5f. **Natural 8-2 + late retime (2026-08-05):** +1 lead at cont **8917**,
    8-3 +2 lead + patches (first-control handoff), 8-4 bowser patch.
    Full natural ending **21,559f** (−84 vs prior reactive 21,643). Seed
    `smb_1_1_to_ending_natural_82.json`.
5g. **Clean power-on 3/3 (2026-08-05):** natural seed, boot 350 + settle 16,
    zero mid loads, `benchmark_eligible: true`. RTA **05:58.726**.
6. **Route contracts + natural predecessor evaluator (2026-08-01):** done.
   `smb.reactive_route` tracks declared successors and entry fingerprints;
   `smb.scripts.run_reactive_warp` runs stairs + reactive 1-2 and retimed
   tails in one environment.
7. **Default fold + capture:** promote natural_82 as source-owned continuous
   seed and re-record Clean power-on MP4.

### C — Optional follow-ons

1. Silver/Gold runtime: natural-entry 4-2 and mushroom-cloud speed
2. Non-warp all-32-exit route
3. Transfer continuous fold patterns to SMB3 / retro_harness.platformer

For all-32, use `ROUTE_ALL_EXITS` contracts and the explicit missing-policy
coverage report as the source of truth. Do not use the stitch renderer as
completion evidence; each new stage needs a natural-entry controller before
it can count toward a Clean route.

## Commands (polish)

```bash
# List bottleneck windows (continuous seed)
uv run python -m smb.scripts.rle_polish --list-windows

# Hillclimb 1-1 stairs window on continuous seed
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.rle_polish --window 1-1-stairs --iters 400

# Isolated 1-1 TAS toolkit (analyze / optimize / verify)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.tas_1_1 analyze
uv run python -m smb.scripts.tas_1_1 optimize --window stairs,first-pipe --iters 400
uv run python -m smb.scripts.tas_1_1 optimize --delete-stride 1 --iters 0 --window stairs

# GA on 4-2 entry
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.rle_polish --window 4-2-entry --mode ga --gens 40
```

## Notes

- Best Clean power-on: `smb/models/smb_1_1_to_ending_natural_82.json`
  (**21,559f**, 3/3). Prior reactive `smb_1_1_to_ending_reactive_83_84.json`
  (**21,643f**). Source-owned default fold remains `smb_1_1_to_ending.json`
  (**21,731f**) until promote.
- Power-on: boot **350** + settle **16**
- Level1_1 continuous: settle **14**
- Rebuild: `uv run python -m smb.scripts.fold_continuous_policy`
