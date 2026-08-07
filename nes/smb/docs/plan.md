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
5h. **8-1 Level8_1 polish (2026-08-06):** `smb.scripts.polish_8_1` — hold
    trim @2573 (−6) + A-edge @2835 → isolated clear **3444→3402** (−42f).
    Natural continuous splice probe: 8-1 exit **12628→12585**, split
    **3666→3624** (−42). Artifact: `models/smb_8_1_control_best.json`.
    Unpromoted full seed: `smb_1_1_to_ending_natural_81_polished_unpromoted.json`.
    Kept; **not** the main claw-back path (see 5i).
    Evidence: `recordings/segment_8_1/polish_8_1_report.json`.
5i. **TAS adapt track (2026-08-07):** stop primary hill-climb. Pull open
    WR-class movies + trick catalog; adapt under fceumm.
    - Vendored: `tas/ref/happylee_warps_1715M.fm2` (TASVideos #1715),
      `tas/ref/flamexx_warps_rta_4_54_099.fm2`
    - Tooling: `smb.tas.fm2`, `smb.tas.slice`, `smb.scripts.import_fm2`
      (parse, verify, align-search, **--verify-1-2-slice** / export / search).
      **L+R preserved**; FM2 `T`=Start.
    - Power-on full FM2 **desyncs** on fceumm (blackout longer than FCEUX).
    - **Isolated Level1_1** HappyLee slice **1733f clear**
      (`models/smb_1_1_happylee_slice.json`) vs our ~1903 — **≈−170–190f**.
    - **Natural-entry 1-1:** odd settle (default 1) clears **1749f**; even dies.
    - **Control-relative 1-2 W4:** FM2 start **2109**, body **1657f**
      (`models/smb_1_2_happylee_slice.json`). Chain ≈ **3555f to W4**
      (−329 vs natural_82 3884). Odd start indices after odd ctrl_wait.
    - **Control-relative 4-1 / 4-2 → W8:** FM2 **3968**/2062f + **6207**/1516f
      (`smb_4_1_happylee_slice.json`, `smb_4_2_happylee_slice.json`).
      Chain ≈ **7512f to W8** (−5116 vs natural_82 12628). Even 4-1 /
      odd 4-2 parity; 4-2 gate allows timer=0. **4-2 is glitch/warp path**
      (not natural vine) — video+RAM audit before full promote.
    - **W8 8-1/8-2 exported + verified:** wait81=209 → 8-1 @7930/2881;
      wait82=165 → 8-2 @10910/2209; to 8-3 load **12976**. Seeds
      `smb_8_1_happylee_slice.json`, `smb_8_2_happylee_slice.json`.
    - **Hybrid v1 (3/3):** natural_82@**15933** full 8-3/8-4 → **18769f /
      5:12.3**. Seed `smb_happylee_hybrid_ending.json` (kept baseline).
    - **Hybrid v2 (3/3):** same through 8-3 control, then **flamexx 8-4
      @15210 / 2661f** → **18031f / 5:00.02** (−3528 vs n82; **+1** vs
      18030 sub-5 budget). Seeds `smb_happylee_hybrid_v2_fx84.json`,
      `smb_8_4_flamexx_slice.json`, `smb_8_3_natural_for_hl_hybrid.json`.
    - **Pure HL 8-3 still open** (phase max_x≈834); 8-4 TAS body done via FX.
    - **8-3 stitchless path (rr-34v, 2026-08-07):** user direction = **no
      natural_82 hybrid stitch** as primary. Pure HL/FX after HL 8-2 control.
      Gated FM2 raw best max_x≈**1030** (si≈13081 lead0); multi-round heal
      del/ins/A-edge → **≈1730** (still no leave). Multi-leave 8-2 does not
      change 8-3 ctrl fp (t=301). Tools: `probe_stitchless_8_3`, heal notes
      under `recordings/tas_import/happylee_8_3_stitchless_*.json`. Hybrid
      v2 18031 + `polish_8_3` kept as showcase/secondary only.
    - **Next:** continue stitchless heal / grounded reactive 8-3 until leave;
      then FX/HL 8-4 continuous; Clean power-on fold after full chain.
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

# 8-1 Level8_1 body polish (natural_82 slice; best first: late window)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.polish_8_1 --windows late --delete-stride 1
uv run python -m smb.scripts.polish_8_1 --baseline-only

# TAS / FM2 import (prefer over blind hill-climb)
uv run python -m smb.scripts.import_fm2 --summary-only
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.import_fm2 --verify
# HappyLee 1-1 slice (Level1_1, settle=2, fm2 index 190)
uv run python -m smb.scripts.tas_1_1 verify \
  --seed nes/smb/models/smb_1_1_happylee_slice.json
# Natural-entry 1-1 + control-relative 1-2 → W4
uv run python smb/scripts/run_1_1.py --natural-entry \
  --seed nes/smb/models/smb_1_1_happylee_slice.json
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.import_fm2 --verify-1-2-slice
# HL 4-1 + 4-2 → W8
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.import_fm2 --verify-4-1-4-2-slice
```

## Notes

- Best Clean power-on: `smb/models/smb_1_1_to_ending_natural_82.json`
  (**21,559f**, 3/3). Prior reactive `smb_1_1_to_ending_reactive_83_84.json`
  (**21,643f**). Source-owned default fold remains `smb_1_1_to_ending.json`
  (**21,731f**) until promote.
- Power-on: boot **350** + settle **16**
- Level1_1 continuous: settle **14**
- Rebuild: `uv run python -m smb.scripts.fold_continuous_policy`
