# Status — Super Mario Bros. (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M8 |
| Best verified result | Power-on → 8-4 ending, Clean, reactive continuous |
| Last verification | 2026-08-01 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **M8** Clean power-on any% warp ending, captured |
| Integration | `SuperMarioBros-Nes` (boot) / `SuperMarioBros-Nes-v0` (autobot) |
| ROM zip | `roms/Nintendo/NES/Super Mario Bros..zip` |
| Evidence | [poweron trials](../recordings/warp_finish/warp_finish_poweron_trials_report.json), [M8 capture](../recordings/warp_finish/warp_finish_poweron_m8_capture.json), [TAS validation](../recordings/warp_finish/warp_finish_poweron_tas_validation.json), [video](../recordings/fullgame_replays/smb_warp_any_percent_poweron.mp4) |

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
- **Reactive late-route repair (2026-08-01)**: the documented **drop-5** at
  controller frame 12,898 makes 8-2 reach 8-3 in **15,863f** (−84f versus
  M8). `smb.reactive_late` takes control at the natural 8-3 gate, applies four
  stage-relative corrections, verifies the natural 8-4 handoff, then applies a
  final Bowser/axe jump window. The generated controller reaches the ending in
  **21,643f** (−88f versus M8), lives 2→2. It passes **3/3 Clean power-on**
  trials with zero mid-attempt loads; evidence:
  [reactive power-on trials](../recordings/reactive_warp/reactive_83_84_poweron_trials_report.json).
- **Reactive route infrastructure (2026-08-01)**: `smb.reactive_route` now
  declares normal/warp successor contracts for all 32 stages, records
  control-entry and successor fingerprints, and reports missing controllers
  explicitly. `run_warp_finish` uses those contracts for the M8 eight-exit
  report (reverified **21,731f**, 8/8). The source-owned first-pipe patch in
  `fold_continuous_policy` now reproduces the published M8 seed byte-for-byte.
  `smb.scripts.run_reactive_warp` runs the real stairs + reactive-1-2
  predecessor. Its unretimed continuation reaches 8-2 at **12,733f** then
  dies; `--retime-8-2` owns the natural 8-3/8-4 handoffs and completes.

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
| `poweron` | reset + 350 boot + 16 idle + reactive seed | **8/8 Clean, 3/3** |
| `continuous` | Level1_1 + 14 idle + seed | 8/8, no mid load |
| `suffix` / `chain` | legacy mid-1-2 paths | development only |

## Traps

- Power-on: **exactly** boot=350 + settle=16 before the seed.
- Level1_1 continuous: settle **14** (different phase).
- Underground `level_id=2` is not completion.
- Ending = World 8-4 + `oper_mode=2`, held 120 idle frames.

## 1-1 segment polish (2026-07-28)

- `smb_1_1_clear.json`: clear **2029f** (was 2239; **−210f**) via leading-idle
  trim 38 + flagpole/castle auto-input cleanup (`player_state` 3/4/5 idle).
- Natural-entry still clears (settle=1).
- Raw hillclimb 400 iters on the trimmed seed: no further improvements.
- Recording (`--record`): native emulator audio + footer with frame/time stamp,
  level/lives/x, and NES button presses (auto-states blanked in HUD only).

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
| Current policy | **21,643f** → 8-4 ending; boot 350 + settle 16 |
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
| `rta_any_percent` | **06:00.124** (21,643f) | HappyLee RTA note **04:54.032** (17,671f) | **+01:06.091** (+3,972f) |
| `tasvideos_poweron` | **06:06.214** (22,009f) | [HappyLee #1715](https://tasvideos.org/1715M) **04:57.31** (17,868f) | **+01:08.903** (+4,141f) |
| RTA perfect band | — | Maru RTA-rules ≈**04:54.265** | ours +~67s |
| Human WR band | — | ≈**04:54.4** | ours +~67s |

Bronze gap is real performance, not a clock mismatch: same start/end as the
public anchors above.

### RTA segments (exit-detect @ NTSC)

| Exit | Cum | Seg | Seg time |
|------|-----|-----|----------|
| 1-1 | 1911 | 1911 | 00:31.798 |
| 1-2 | 3981 | 2070 | 00:34.443 |
| 4-1 | 6303 | 2322 | 00:38.636 |
| 4-2 | 9067 | 2764 | 00:45.991 |
| 8-1 | 12733 | 3666 | 01:01.000 |
| 8-2 | 15863 | 3130 | 00:52.081 |
| 8-3 | 18069 | 2206 | 00:36.706 |
| 8-4 | 21643 | 3574 | 00:59.469 |

## Next

1. **Default fold + capture:** promote the verified reactive 21,643f policy
   into the source-owned default fold, then capture that Clean run. Do **not**
   idle-pad at W4 to restore the old phase.
2. Further natural-entry 4-2 polish: largest remaining split regression
   (prefer `smb_4_2_fast_w8.json`, not isolated Level4_2 on continuous frames).
3. Optional all-32-exit (non-warp) route as a separate track.
4. Transfer reactive control-gate patterns to SMB3 / other platformers.

The route contract layer makes the 32-exit inventory auditable, but only the
eight warp controllers have source material today. Missing normal-stage
controllers are deliberately reported rather than skipped or presented as a
continuous finish.
