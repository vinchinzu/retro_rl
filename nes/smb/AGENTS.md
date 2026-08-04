# Agent Instructions — smb

Scripted NES completion agent for **Super Mario Bros.** (platforming track;
maturity **M8** Clean power-on → 8-4 ending, captured).

## Identity

| Field | Value |
|-------|-------|
| Status | M8 Clean power-on → 8-4 ending (3/3, video capture) |
| Boot integration | `SuperMarioBros-Nes` |
| Full-run / autobot integration | `SuperMarioBros-Nes-v0` (symlink → snes_editor) |
| Shared ROM zip | `roms/Nintendo/NES/Super Mario Bros..zip` |
| Continuous finish seed | `smb/models/smb_1_1_to_ending.json` |
| Power-on phase | boot **350** frames + settle **16** idle |
| Level1_1 continuous phase | settle **14** idle |
| Practice traces | `smb/recordings/fullgame` → snes_editor fullgame sessions |

## Commands

```bash
uv run python smb/scripts/setup_rom.py
uv run python smb/scripts/boot_probe.py
uv run python -m pytest smb/tests -q

# M7 Clean power-on → 8-4 ending
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_warp_finish --mode poweron --trials 3

# Record full MP4 (native audio + button/timestamp footer HUD)
# Do not force SDL_AUDIODRIVER=dummy if you need the audio stream (core still
# exposes PCM under dummy; current path works either way).
uv run python -m smb.scripts.run_warp_finish --mode poweron --record
# Optional: --no-record-audio / --no-record-hud / --record-scale 2
# Evidence: recordings/warp_finish/warp_finish_poweron_tas_validation.json

# Level1_1 continuous (no boot)
uv run python -m smb.scripts.run_warp_finish --mode continuous --trials 3

# Rebuild continuous seed
uv run python -m smb.scripts.fold_continuous_policy

# 1-1 natural segment
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python smb/scripts/run_1_1.py --natural-entry --trials 3

# Reactive 1-2 (state-gated; works after stairs or baseline 1-1)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_1_2 --predecessor stairs --trials 3

# Development: real shifted World-4 predecessor through the old continuation.
# Without the retime it dies in 8-2; never phase-pad this result.
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_reactive_warp

# Apply drop-5, then take over at the natural 8-3/8-4 control gates.
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_reactive_warp --retime-8-2

# Recheck the promoted reactive controller from a Clean power-on.
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_warp_finish --mode poweron --trials 3 \
    --seed-continuous smb/models/smb_1_1_to_ending_reactive_83_84.json
```

## Architecture

| Module | Role |
|--------|------|
| `smb/ram.py` | M2 + velocities/timer/camera + segment / ending predicates |
| `smb/obs.py` | Shared 210-dim observation builder (neuro / future PPO) |
| `smb/policy.py` | RLE seeds + power-on/continuous phase constants |
| `smb/reactive_12.py` | State-gated 1-2 warp (wait control → reactive surface → ug RLE) |
| `smb/reactive_late.py` | Natural-control 8-3/8-4 repairs applied to M8 stage slices |
| `smb/reactive_route.py` | Reusable state gates, successor contracts, split/fingerprint tracking, coverage |
| `smb/scripts/run_1_2.py` | Natural 1-1 predecessor + reactive 1-2 → World 4 |
| `smb/scripts/run_reactive_warp.py` | Same-env reactive predecessor runner for late-leg natural re-solves |
| `smb/scripts/run_warp_finish.py` | poweron / continuous / suffix / chain finish + video |
| `smb/scripts/fold_continuous_policy.py` | Fold prelude + suffix into continuous seed |
| `smb/scripts/rle_polish.py` | Hierarchical RLE hillclimb/GA on bottleneck windows |
| `smb/scripts/run_1_1.py` | M3/M4 1-1 runner |
| `smb/routes.py` / `full_run.py` | Showcase stitch routes |
| `smb/timing.py` | TASVideos / RTA / policy timing contracts + public anchors |
| `retro_harness/platformer/rle_ops.py` | RLE compress/mutate/crossover |
| `retro_harness/platformer/rle_optimize.py` | Windowed RLE hillclimb/GA + phase-shift polish |
| `retro_harness/platformer/neuro.py` | MLP/CNN neuroevo + BC warm-start from RLE |

## Next milestone

Make the verified 21,643f reactive policy the default reproducible fold, then
capture it. Afterwards: 4-2 polish / optional all-32.

```bash
# Reactive 1-2 after stairs 1-1 (verified 2/2 → World 4, −63f to W4)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_1_2 --predecessor stairs --trials 3

# Hierarchical RLE polish (bottleneck windows)
# 1-1-stairs = frames 1050-1311 (wall-slam region), NOT castle idle.
uv run python -m smb.scripts.rle_polish --list-windows
```

## Traps

- Power-on: **exactly** 350 boot script frames + **16** idle, then the seed.
- Level1_1 continuous: **exactly 14** idle after `Level1_1` (different phase).
- Natural 1-1 segment alone: idle **1** after readiness (`NATURAL_SETTLE_FRAMES`).
- World 4 = `world` index **3**. Underground `level_id=2` is not completion.
- Ending = World 8-4 + `oper_mode=2`, held 120 idle frames.
- **Do not** absolute-frame stitch a faster 1-1 into the old 1-2 suffix — use
  `smb.reactive_12` (control gates). Do **not** W4 idle-pad to restore phase
  (erases stairs savings); retime later legs from natural control instead.
- Prefer `uv run python -m pytest` (project venv / stable_retro).

## Norms

- Prefer local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, seeds, and policies inside this game directory.
- After editing `docs/manifests/smb.yaml`, run
  `uv run python docs/generate_game_matrix.py`.
