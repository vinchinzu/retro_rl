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
    - Tooling: `smb.tas.fm2`, `smb.tas.slice`, `smb.scripts.annotate_fm2`
      (parse, verify, align-search / export / search).
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
    - **8-3 stitchless leave (rr-34v, 2026-08-07):** **done 2/2** — no
      natural_82 mid-splice. Progress heal max_x **3390** then skill-resume
      (land-pin cut1478 + pure hop jh=44/gap=6/hops=3 + idle fold) → **8-4
      control 2374f** (`smb_8_3_stitchless_skills_leave.json`). Evidence
      `happylee_8_3_skills_leave.json`. Rich FP + skills in `ram.py` /
      `tas/skills_8_3.py`. Pure continuous FM2
      8-3 still phase-blocked; hybrid v2 showcase-only. Grounded/pit-jump
      long grids **paused**.
    - **Pure capture handoff (2026-08-09):** the original 17,868-frame
      HappyLee FM2 replays through 8-3 and the axe under FCEUX 2.6.6. The
      stable-retro/fceumm break is local at movie frame **13222** (same
      x/y/y-fraction, FCEUX vy=−3 vs fceumm vy=−5), so do not repair it by
      mutating track-3 input. A provisional 299.490s MP4 exists under
      `recordings/tas_import/pure_hl/`, but the experimental FCEUX raw-pipe
      encoder had unreliable shutdown and was not retained.
    - **Next for pure track 3:** replay the untouched FM2 in BizHawk with the
      matching ROM checksum, use native A/V capture, and verify 17,868 movie
      inputs, duration <300s, 8-3 passage, and visible axe completion. Keep
      the stable-retro 8-3 adaptation issue separate and still open.
    - **Next:** fold HL…8-2 + skills 8-3 + FX 8-4 continuous; verify 3/3;
      Clean power-on; optional 21f polish / FPG macro on leave class.
6. **Route contracts + natural predecessor evaluator (2026-08-01):** done.
   `smb.reactive_route` tracks declared successors and entry fingerprints;
   `smb.scripts.run_reactive_warp` runs stairs + reactive 1-2 and retimed
   tails in one environment.
7. **Default fold + capture:** promote natural_82 as source-owned continuous
   seed and re-record Clean power-on MP4.

### E — All-32-exits track (2026-08-27)

Standalone segment work on `smb_all_exits` (no warp-line changes). Gate
order: 1-2 **flag** exit → 1-3 control state → per-stage extract/polish.

- **Pin audit:** old `all_exits_v1` 1-3/1-4 were pre-fb4118e9 AreaNumber
  mislabels. 1-3 pin is now a real control spawn (this session). 1-4 pin
  is still a 1-3 castle tally (not extractable). 1-1/1-2 genuine; 2-1 is
  mid-stage (x=2431). No full tape saved. Evidence:
  `recordings/segments_all_exits/evidence/`.
- **Tooling:** `smb.scripts.extract_stage_state` wrote `Level1_3.state`.
  `smb_1_3` uses `SMB_DASH_COMPUTED` (not default `_smb_level`). 1-4 pin
  is **not** extractable.
- **1-2 flag exit (2026-08-27):** DOWN pipe on the brick platform after
  the UG lifts (`flag_12` truth table). HL last physics-grounded lift pose
  `(2520, 148)`, A-only 19f, land `(2620, 128)`, walk to short pipe
  `player_state=2`. Outdoor flag → 1-3 control pin + `Level1_3.state`.
  Plant pipes A/B/C and the warp room are not this exit.
- **1-2 flag body (2026-08-27):** HL UG prefix + lift/pipe tail, **2/2**
  into 1-3 control (2796f, `smb.flag_12` / `run_1_2_flag`). `smb_1_3`
  registered with `SMB_DASH_COMPUTED` (`$075C`), completion `[3]`. Isolated
  1-3 clear from `Level1_3` still open (athletic pits; bunny-20 dies
  ~x=844). Full details: `docs/HANDOFF_32EXIT_1_3.md`.
- **Warpless TAS import (2026-08-27):** no 32-exit movie was on disk.
  Source is HappyLee & Mars608 [warpless #3728M](https://tasvideos.org/3728M)
  (18:36.78, 67,117f). Fetch `smb.tas.fetch_refs`, NesHawk BK2 via
  `smb.scripts.convert_fm2` (same mapping as `happylee_warps_1715M.fm2.bk2`),
  then isolated 1-1 / dash-level annotate like the warp track. Do not fold
  into #1715M warp slices. Encode (MKV/MP4) was not local; movie file is
  the extract source. Isolated 1-1 **1754f @190** (3/5 even starts);
  1-2 flag **2544f @2109 → 1-3** (not W4); 1-3 **1740f @4653 → 1-4**
  (wait=0, `--verify-1-3`). Isolated `Level1_3` TAS body misses phase.
  Seeds: `smb_1_1_warpless_slice.json`, `smb_1_2_warpless_flag_slice.json`,
  `smb_1_3_warpless_slice.json`. Play/record the same-file chain (not
  #1715M warps): `smb.scripts.record_warpless --to 1-3` (**6205f** to
  1-4 control). Next: 1-4 @~6393 → 2-1, then 2-1…8-4. Recipe:
  `docs/HANDOFF_32EXIT.md`.

### D — Residual observation + approximate stepper (2026-08-13)

Scaffold only (not a route-clear claim). Same `R(τ)` lattice as Super Metroid,
SMB addresses. Thin flat-ground `step(player, action, world)`.

- Map + rules: `docs/RESIDUAL.md`
- Code: `observation.py` (`Observation` / `PlayerPhysics` / `World`),
  `approx.py`, `residual.py`, `residual_harness.py`
- Shared compare: `retro_harness.residual` (`LatticeSpec`)
- Measure: `uv run python -m smb.scripts.measure_residual`

Jump tables, brake `$98`/`$D0`, LEFT first-kick (`$FED0`), air walk-max
keeping `xf`, and InitJS wipe of `$0416` done. Next: collision as a
`World` query; SMW reuses `retro_harness.residual`.

### C — Optional follow-ons

1. Silver/Gold runtime: natural-entry 4-2 and mushroom-cloud speed
2. Non-warp all-32-exit route — recorder is `./play smb` (power-on →
   `all_exits_v1`, stage pins, archive-on-reuse). Each stage still needs a
   natural-entry controller before it counts as a Clean route.
3. Transfer continuous fold patterns to SMB3 / retro_harness.platformer

For all-32, use `ROUTE_ALL_EXITS` contracts and the explicit missing-policy
coverage report as the source of truth. Do not use the stitch renderer as
completion evidence; each new stage needs a natural-entry controller before
it can count toward a Clean route.

## CLI catalog (parked)

Daily commands live in `AGENTS.md`. These are not the living 32-exit
extract. Prefer TAS adapt (`docs/TAS_ADAPT.md`) over hill-climb.

```bash
# Warp A/B (M8)
uv run python -m smb.scripts.run_warp_finish --mode poweron --record
uv run python smb/scripts/run_1_1.py --natural-entry --trials 3
uv run python -m smb.scripts.run_1_2 --predecessor stairs --trials 3
uv run python -m smb.scripts.run_reactive_warp --retime-4-1 --retime-4-2 --retime-8-2
uv run python -m smb.scripts.fold_continuous_policy

# 32-exit flag body / isolated 1-3
uv run python -m smb.scripts.run_1_2_flag --record --trials 2
uv run python -m smb.scripts.run_1_3 --search --trials 2
uv run python -m smb.scripts.extract_stage_state --list

# TAS import
uv run python -m smb.scripts.convert_fm2
uv run python -m smb.scripts.annotate_fm2 --search 2-2 --from-pred --export
uv run python -m smb.scripts.record_happylee --to ending
uv run python -m smb.scripts.pure_hl status
```

## Notes

- Best Clean power-on: `smb/models/smb_1_1_to_ending_natural_82.json`
  (**21,559f**, 3/3). Prior reactive `smb_1_1_to_ending_reactive_83_84.json`
  (**21,643f**). Source-owned default fold remains `smb_1_1_to_ending.json`
  (**21,731f**) until promote.
- Power-on: boot **350** + settle **16**
- Level1_1 continuous: settle **14**
- Rebuild: `uv run python -m smb.scripts.fold_continuous_policy`
