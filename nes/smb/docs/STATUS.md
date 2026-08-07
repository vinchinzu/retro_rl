# Status — Super Mario Bros. (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M8 |
| Best verified result | Power-on → 8-4 ending, Clean, **21,559f** natural retime |
| Last verification | 2026-08-05 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **M8** Clean power-on any% warp ending (natural retime) |
| Integration | `SuperMarioBros-Nes` (boot) / `SuperMarioBros-Nes-v0` (autobot) |
| ROM zip | `roms/Nintendo/NES/Super Mario Bros..zip` |
| Best seed | `models/smb_1_1_to_ending_natural_82.json` (**21,559f**, 3/3 Clean) |
| Evidence | [natural 3/3 poweron](../recordings/reactive_warp/natural_82_poweron_trials_report.json), [M8 baseline](../recordings/warp_finish/warp_finish_poweron_trials_report.json), [M8 capture](../recordings/warp_finish/warp_finish_poweron_m8_capture.json), [video](../recordings/fullgame_replays/smb_warp_any_percent_poweron.mp4) (**21,559f** continuous + YouTube intro, 2026-08-06) |

## Done

- **M2** RAM: absolute `player_x`, `level_id`, death, World 4 / ending detect
- **M3/M4 1-1**: isolated `Level1_1` + natural power-on segment
- **M5** continuous eight-exit seed (no mid-1-2 splice): `smb_1_1_to_ending.json`
- **M7/M8 baseline Clean power-on → ending** (3/3):
  - `env.reset()` only — **zero mid-attempt state loads**
  - Fixed boot script **350** frames + idle **16** frames
  - Baseline continuous seed **21,731f** → 1-1…8-4, lives 2→2, 120-frame ending settle
  - `benchmark_eligible: true`, intervention class Clean
- **M8 verified capture**: power-on MP4 with **native audio**, footer HUD
  (frame/timestamp, NES buttons, level/lives/x), trials report + capture
  manifest + public TAS time validation
- **Optimizer architecture (2026-07-30)**: hierarchical RLE ops + window
  polish (`retro_harness/platformer/rle_*`, `smb.scripts.rle_polish`); richer obs
  (velocities/grounded/timer/camera, 210-dim `smb/obs.py`); neuro MLP/CNN +
  BC warm-start. The source-owned M8 baseline remains **21,731f**.
- **1-1 stairs research (2026-07-30)**: bottleneck window was wrong
  (1700–1974 = castle score idle). Real wall-slams at **f≈1164 / 1210**
  (x=2962 / 2994). Micro A-hold hillclimb on frames 1120–1290 → flag
  **1311→1242**, level-load **1975→1912** (**−63f**), **0** wall-slams.
  Artifacts: `models/smb_1_1_stairs_best_frames.json`,
  `models/smb_1_1_stairs_clear_fragment.json`.
- **Reactive 1-2 (2026-07-31)**: replaced phase-sensitive absolute 1-2 macro
  stitch with state-gated controller (`smb/reactive_12.py`):
  wait surface control → reactive RIGHT/DOWN pipe → wait underground control
  → control-relative underground RLE → World 4. Fragments in
  `models/smb_1_2_reactive_fragments.json`. Verified **2/2** after stairs
  1-1 (settle=14): 1-1 **1911f** + 1-2 **2070f** = **3981f to W4**
  (**−63f** vs baseline 4044; no pad). CLI:
  `uv run python -m smb.scripts.run_1_2 --predecessor stairs`.
  W4 player physics matches; 4-1→8-1 clear without pad.
- **1-2 UG polish (2026-08-05)**: control-relative systematic delete + hold
  trim on `underground_from_control` only (`smb.scripts.polish_1_2_ug`).
  UG clear **1545→1507** (**−38f**). Isolated Level1_2 **1854→1816**.
  Natural stairs: 1-2 **2070→2032**, W4 **3981→3943** (1-1 still 1911).
  Surface remains reactive (no absolute stitch). Full power-on/any% fold
  not yet re-captured with this fragment (later legs need natural retime).
  Evidence:
  [polish report](../recordings/segment_1_2/polish_1_2_ug_report.json).
- **1-2 W4 pipe top-land (2026-08-05)**: rewrote late UG suffix (from ug
  index **1344**) so Mario reverse-jumps from the right platform onto the
  **rightmost (W4) pipe lip** and holds DOWN — no face-slam at x≈2830 and
  **no floor bounce** before enter. UG clear **1507→1448** (**−59f**).
  Isolated Level1_2 **1816→1757**. Natural stairs 2/2: 1-2 **2032→1973**,
  W4 **3943→3884** (1-1 still 1911). Residual: W4 land still carries
  leftward speed (xs≈−40); enter is safe because DOWN is already held.
  Tool: `smb.scripts.polish_1_2_warp_pipe`. Evidence:
  [warp-pipe report](../recordings/segment_1_2/polish_1_2_warp_pipe_report.json),
  [natural trials](../recordings/segment_1_2/1_2_reactive_trials_report.json).
- **Natural 4-1 retime (2026-08-05)**: after polished 1-2 (W4@**3884**),
  absolute continuation leaves **~9f** of post-control idle before the
  intentional RIGHT+B (old control was @tail 218; new control @210). No
  W4 pad — idle through the pipe transition, then resume continuation at
  index **218** (`KNOWN_41_CONTROL_RESUME`). Results (stairs + reactive
  1-2, control-relative 4-1):

  | Path | Unretimed | Retimed 4-1 | Δ |
  |------|-----------|-------------|---|
  | 4-1 split | 2335 | **2314** | **−21** |
  | Abs 4-1 | 6219 | **6198** | −21 |
  | 4-2 | death @~7452 | **8962** clear | unblocked |
  | 8-1 | — | **12628** clear | unblocked |

  Control-relative body matches the pre-1-2-polish 2104f (vs 2125
  unretimed). 4-2/8-1 clear without further retime; **8-2 still dies**
  (old drop-5 is phase-stale for the extra −97f). CLI default:
  `run_reactive_warp --retime-4-1` (use `--no-retime-4-1` to compare).
  Evidence: `recordings/reactive_warp/retime_4_1_report.json`.
- **Natural 4-2 chain (2026-08-05)**: human `late_v1` (3221f from 4-2
  control) live-cleared W8 but is not TAS-stable on replay. Solved
  control-relative body from the retimed continuation instead:
  **2599f** surface→UG→W8 pipe (`x=810,timer=324`), lives 2→2.
  Artifact: `models/smb_4_2_natural_control.json`. Evidence:
  [chain report](../recordings/human/late_v1_chain_solve_report.json).
  Human still useful for jump geometry; skills under
  `models/human_skills/late_v1/`.
- **Control-relative 4-1→4-2 retime (2026-08-05)**: 4-1 castle tally length
  is game-driven; absolute continuation can desync if the body ends early.
  Runner now (`--retime-4-2`, default on): after 4-1 control body, **freeze
  source** on score/load (`player_state=5`, timer=0, x≥3000) and idle until
  natural 4-2 control, then resume cont index **2487**
  (`KNOWN_42_CONTROL_RESUME`). No pad. Verified same path as 4-1-only retime:
  4-1@**6198**, 4-2@**8962**, 8-1@**12628**, death in 8-2. Lead 12 idles at
  4-2 control are phase-critical (any trim dies). Vine climb ~316f is auto.
  `smb_4_2_fast_w8` does **not** splice onto this natural UG control (phase
  mismatch). Evidence: `recordings/reactive_warp/natural_41_42_v2_report.json`.
- **Natural 8-2 retime (2026-08-05)**: after 1-2 −97f, drop-5 @12,898 and all
  lead-idle **drops** die (clusters x≈450/1450/2271). From natural 8-2 control
  (cont **8917** / abs 12,898): **+1 lead idle** then body clears 8-2 → 8-3
  control @ stage **3151f**. Late controllers re-solved for the new phase:
  8-3 = **+2 lead** + existing mid-level patches (handoff on first 8-4
  control, not forced exhaust); 8-4 = Bowser/axe patch, no lead. Full natural
  Level1_1 path (retime 4-1/4-2/8-2 + late):

  | Exit | Frame | Seg |
  |------|------:|----:|
  | 1-1 | 1911 | 1911 |
  | 1-2 | 3884 | 1973 |
  | 4-1 | 6198 | 2314 |
  | 4-2 | 8962 | 2764 |
  | 8-1 | 12628 | 3666 |
  | 8-2 | 15779 | 3151 |
  | 8-3 | 17985 | 2206 |
  | 8-4 | **21559** | 3574 |

  **21,559f** ending (−84 vs prior reactive 21,643; −172 vs M8 21,731), lives
  2→2. CLI: `run_reactive_warp --retime-4-1 --retime-4-2 --retime-8-2`.
  Seed: `models/smb_1_1_to_ending_natural_82.json`. Evidence:
  [natural_82_retime_report](../recordings/reactive_warp/natural_82_retime_report.json).
- **Clean power-on 3/3 (2026-08-05)**: same natural seed under
  `env.reset` + boot **350** + settle **16**, zero mid-attempt loads,
  `benchmark_eligible: true`, intervention Clean. All three trials ending
  @ **21,559f**, 8/8 exits, lives 2→2.

  | Contract | Ours | Public | Δ |
  |----------|------|--------|---|
  | `rta_any_percent` | **05:58.726** (21,559f) | HappyLee 04:54.032 | +01:04.693 |
  | `tasvideos_poweron` | **06:04.816** (21,925f) | HappyLee 04:57.31 | +01:07.505 |

  Evidence:
  [natural_82 poweron 3/3](../recordings/reactive_warp/natural_82_poweron_trials_report.json).
  Default fold / published MP4 still M8 baseline pending promote.
- **Reactive late-route repair (2026-08-01)**: original **drop-5** @12,898 +
  8-3/8-4 patches finished in **21,643f** on the pre−97f 1-2 path (3/3 Clean
  power-on). Superseded for the polished 1-2 path by the natural 8-2 retime
  above. Evidence (historical):
  [reactive power-on trials](../recordings/reactive_warp/reactive_83_84_poweron_trials_report.json).
- **Reactive route infrastructure (2026-08-01)**: `smb.reactive_route` now
  declares normal/warp successor contracts for all 32 stages, records
  control-entry and successor fingerprints, and reports missing controllers
  explicitly. `run_warp_finish` uses those contracts for the M8 eight-exit
  report (reverified **21,731f**, 8/8). The source-owned first-pipe patch in
  `fold_continuous_policy` now reproduces the published M8 seed byte-for-byte.
  `smb.scripts.run_reactive_warp` runs stairs + reactive 1-2 with control-
  relative 4-1/4-2/8-2 retimes and late 8-3/8-4.

## Autobot commands

```bash
# M7/M8 Clean power-on → 8-4 ending
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_warp_finish --mode poweron --trials 3

# Record full MP4 (audio + button/timestamp HUD)
uv run python -m smb.scripts.run_warp_finish --mode poweron --record

# Level1_1 continuous (no boot)
uv run python -m smb.scripts.run_warp_finish --mode continuous --trials 3

# Rebuild continuous seed
uv run python -m smb.scripts.fold_continuous_policy
```

| Mode | Path | Result |
|------|------|--------|
| `poweron` | reset + 350 boot + 16 idle + natural_82 seed | **8/8 Clean, 3/3 @ 21,559f** |
| `continuous` | Level1_1 + 14 idle + seed | 8/8, no mid load |
| `suffix` / `chain` | legacy mid-1-2 paths | development only |

## Traps

- Power-on: **exactly** boot=350 + settle=16 before the seed.
- Level1_1 continuous: settle **14** (different phase).
- Underground `level_id=2` is not completion.
- Ending = World 8-4 + `oper_mode=2`, held 120 idle frames.

## 1-1 segment polish (2026-07-28 → 2026-08-04)

- `smb_1_1_clear.json`: clear **2029f** (was 2239; **−210f**) via leading-idle
  trim 38 + flagpole/castle auto-input cleanup (`player_state` 3/4/5 idle).
- Natural-entry still clears (settle=1).
- Raw hillclimb 400 iters on the trimmed seed: no further improvements.
- Recording (`--record`): native emulator audio + footer with frame/time stamp,
  level/lives/x, and NES button presses (auto-states blanked in HUD only).
- **TAS toolkit (2026-08-04):** `smb/tas/` + `scripts/tas_1_1.py`
  (analyze / multi-window hillclimb / systematic delete+edge polish).
  Prior isolated seed: `models/smb_1_1_tas_best.json` — leave **1903f**,
  flag **1242f** (−21f vs stairs fragment 1924; −126f vs clear 2029).
  Verified isolated + natural-entry (settle=1). Default `DEFAULT_1_1_SEED`
  still points here until continuous fold promotes the HappyLee slice.
- **HappyLee FM2 adapt (2026-08-07):** vendored #1715 warps movie;
  `import_fm2` + `smb.tas.slice` + `docs/TAS_ADAPT.md`. **L+R preserved**.
  Full power-on FM2 desyncs on fceumm — adapt **per level from control**.
  Prefer this track over 8-x hill-climb. Unpromoted (no Clean full-run yet).

  | Segment | Artifact | Result |
  |---------|----------|--------|
  | 1-1 isolated | `smb_1_1_happylee_slice.json` | **1733f** leave (FM2 @190, settle=2); **≈−170f** vs 1903 |
  | 1-1 natural-entry | same seed, settle **odd** (default 1) | **1749f** clear; even settle dies |
  | 1-2 → W4 | `smb_1_2_happylee_slice.json` | FM2 @**2109**, **1657f** W4; chain ≈**3555** to W4 (**−329** vs natural_82 3884) |
  | 4-1 | `smb_4_1_happylee_slice.json` | FM2 @**3968**, **2062f** leave; **≈−252f** vs 2314 |
  | 4-2 → W8 | `smb_4_2_happylee_slice.json` | FM2 @**6207**, **1516f** W8; chain ≈**7512** to W8 (**−5116** vs natural_82 12628) |
  | 8-1 | `smb_8_1_happylee_slice.json` | FM2 @**7930**, leave **2881** (wait81=209); ≈**−785f** body |
  | 8-2 | `smb_8_2_happylee_slice.json` | FM2 @**10910**, leave **2209** (wait82=165); ≈**−942f** body |
  | 8-3 | nat bridge `smb_8_3_natural_for_hl_hybrid.json` | pure HL **open** (max_x≈834); nat@15933 → 8-4 ctrl **2227f** |
  | 8-4 | `smb_8_4_flamexx_slice.json` | FX @**15210**, ending **2661f** after nat 8-3 (HL alt 2833) |
  | hybrid v1 | `smb_happylee_hybrid_ending.json` | 18,769f / 5:12.3 (nat 8-3/8-4) |
  | **hybrid v2** | `smb_happylee_hybrid_v2_fx84.json` | Level1_1→axe **18,031f / 5:00.02** (−3528 vs n82; **+1** vs 18030) |

  ### HL chain vs natural_82 (exit-detect frames + NTSC)

  | Exit | n82 cum | n82 seg | n82 time | HL cum | HL body | HL seg time* | Δ cum |
  |------|--------:|--------:|---------:|-------:|--------:|-------------:|------:|
  | 1-1 | 1911 | 1911 | 00:31.798 | **1733** | 1733 | 00:28.836 | −178 |
  | 1-2 / W4 | 3884 | 1973 | 00:32.829 | **3555** | 1657 | 00:27.571 | −329 |
  | 4-1 | 6198 | 2314 | 00:38.503 | **5831** | 2062 | 00:34.310 | −367 |
  | 4-2 / W8 | 8962 | 2764 | 00:45.991 | **7512** | 1516 | 00:25.225 | −1450 |
  | 8-1 | 12628 | 3666 | 01:01.000 | **≈10602** | 2881 | 00:47.938 | ≈−2026 |
  | 8-2 | 15779 | 3151 | 00:52.430 | **≈12976** | 2209 | 00:36.756 | ≈−2803 |
  | 8-3 | 17985 | 2206 | 00:36.706 | bridge | nat | — | pure HL open |
  | 8-4 | **21559** | 3574 | 00:59.469 | **≈18031** v2 | **2661 FX** | 00:44.277 | hybrid v2 |

  \*HL body only (excludes control waits). Showcase continuous is **hybrid
  v2** (Level1_1, not Clean power-on). Full-run Clean seed still
  **natural_82** until fold. HL 4-2 is glitch/warp path — validate video+RAM
  before promote. Detail: `docs/TAS_ADAPT.md`.

  ### Full-run vs WR (promoted seed only)

  | Contract | Ours (natural_82) | HappyLee WR | Δ |
  |----------|------------------:|------------:|--:|
  | RTA | **05:58.726** (21,559f) | **04:54.032** (17,671f) | +01:04.693 |
  | Power-on | **06:04.816** (21,925f) | **04:57.31** (17,868f) | +01:07.505 |

  Commands: `import_fm2 --verify-8-1-8-2-slice`,
  `record_happylee --to ending` (hybrid v2 MP4).
  Evidence: `happylee_8_1_8_2_slice_verify.json`,
  `happylee_hybrid_v2_fx84_verify.json` (3/3),
  **video** [`happylee_ending.mp4`](../recordings/tas_import/happylee_ending.mp4)
  (Level1_1→axe **18,031f / 5:00.02**, HUD+audio+Peach; **not** Clean
  power-on; **+1f** vs 18030 sub-5 budget — pure HL 8-3 still open).
  Also: `happylee_w8.mp4` (to W8 only, 2:05).

### First-pipe landing fix (continuous seed)

- **Bug:** approach wall-slid the left face of the first pipe at **x≈898**, killed
  all horizontal speed, then slowly climbed onto the lip before DOWN-enter.
- **Fix** (phase-safe rejoin at original enter frame 468): run-jump lands on top
  at **x≈902 / y=112 / xs=40**, coasts+brakes to **x=920**, stands, then
  original DOWN tail enters. No side collision.
- Evidence screenshots: `recordings/segment_1_1/pipe_fix/final_f0369_*` (top
  land) … `final_f0469_*` (enter). Backup:
  `models/smb_1_1_to_ending_pre_pipefix.json`.
- Verified baseline: continuous 8/8 + power-on 8/8, **21,731f** / 1-1 clear
  **1973f**.
- **Still open:** end-stairs still face-slam at x≈2962 / x≈2994 (xs→0); needs a
  2-jump stair window without desyncing the flagpole tail.

## Warp and transition polish (2026-07-28)

- Removed an accidental Start-button pause in 1-2 and retained the clean warp
  entry from the real 1-1 predecessor state.
- Replaced the repeated 4-2 vine/exit attempts with a natural-entry hybrid.
  The optimized fragment rejoins the old controller tail at an identical
  player-physics state and saves 153f versus replaying the full new fragment.
  Its 2,764f split is still 146f slower than the old 2,618f isolated phase,
  so 4-2 remains the clearest next target.
- Phase-aligned the World 8 entries with controller idle only; no save-state
  reloads or runtime assists were added.
- Result: **22,005f → 21,731f** (**−274f / −4.559s NTSC**), verified
  **3/3 continuous and 3/3 power-on**, lives 2→2.

## Capture + TAS contracts (2026-07-28)

Video: `smb/recordings/fullgame_replays/smb_warp_any_percent_poweron.mp4`

Module: [`smb/timing.py`](../timing.py) · evidence:
[tas_validation.json](../recordings/warp_finish/warp_finish_poweron_tas_validation.json)

| Property | Value |
|----------|-------|
| Resolution | 720×720 (scale 3 + 16px footer) |
| Streams | H.264 + AAC stereo @ 32 kHz |
| Audio | native fceumm PCM muxed (mean ≈ −27 dB) |
| HUD | `F##### MM:SS.cc`, `P1:…` buttons, level/lives/x |
| Current policy | **21,559f** natural retime, **3/3 Clean power-on**; published MP4 still M8; boot 350 + settle 16 |
| Preserved MP4 | Historical **22,005f** pre-optimization capture |

### Timing contracts (same definitions as public figures)

| Contract | Start | End | FPS |
|----------|-------|-----|-----|
| `rta_any_percent` | first controllable 1-1 (post phase-align) | first `reached_ending` | NTSC 60.0988… |
| `tasvideos_poweron` | `env.reset` / power-on | first `reached_ending` | NTSC 60.0988… |
| `policy_seed` | seed frame 0 | first `reached_ending` | display 60.0 |

Ending = World 8-4 + `oper_mode=2` (axe / bridge clear). Excludes the 120f
capture settle. Segment splits = exit-detect (post-exit world/level RAM).

### Head-to-head under contracts

| Contract | Ours | Public | Δ |
|----------|------|--------|---|
| `rta_any_percent` | **05:58.726** (21,559f) | HappyLee RTA note **04:54.032** (17,671f) | **+01:04.693** (+3,888f) |
| `tasvideos_poweron` | **06:04.816** (21,925f) | [HappyLee #1715](https://tasvideos.org/1715M) **04:57.31** (17,868f) | **+01:07.505** (+4,057f) |
| RTA perfect band | — | Maru RTA-rules ≈**04:54.265** | ours +~67s |
| Human WR band | — | ≈**04:54.4** | ours +~67s |

Bronze gap is real performance, not a clock mismatch: same start/end as the
public anchors above.

### RTA segments (exit-detect @ NTSC)

Natural retime path (Clean power-on 3/3, RTA exit-detect @ NTSC):

| Exit | Cum | Seg | Seg time |
|------|-----|-----|----------|
| 1-1 | 1911 | 1911 | 00:31.798 |
| 1-2 | 3884 | 1973 | 00:32.829 |
| 4-1 | 6198 | 2314 | 00:38.503 |
| 4-2 | 8962 | 2764 | 00:45.991 |
| 8-1 | 12628 | 3666 | 01:01.000 |
| 8-2 | 15779 | 3151 | 00:52.430 |
| 8-3 | 17985 | 2206 | 00:36.706 |
| 8-4 | **21559** | 3574 | 00:59.469 |

Prior reactive (drop-5, pre−97f 1-2): ending **21,643f**. M8 baseline: **21,731f**.

## Next

1. **HappyLee W8 slices (rr-b8k):** encode 8-1/8-2 gates+export from probe
   (7930/2881, 10910/2209); hybrid v2 18031f with FX 8-4; pure HL 8-3 phase
   still open for ≤18030 / WR; full-chain
   verify + stage table vs WR before fold.
2. **Default fold + capture**: only after HL chain to axe (or keep natural_82
   as published baseline). Re-record Clean power-on MP4 on promote.
3. **4-2 path audit:** HL glitch/warp vs natural vine — video + RAM before
   treating 4-2 slice as route-canonical.
4. Optional all-32-exit; transfer reactive gates to SMB3.

The route contract layer makes the 32-exit inventory auditable, but only the
eight warp controllers have source material today. Missing normal-stage
controllers are deliberately reported rather than skipped or presented as a
continuous finish.
