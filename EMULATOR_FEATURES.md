# Emulator-Specific Features Inventory (Harvest + Super Metroid)

This list captures emulator-facing features already implemented in `harvest/` and `super_metroid_rl/` that should be ported to a shared root harness for consistent recording, human play, playback, training, and model workflows.

## Input + Controller Handling
- Shared SNES button mapping, keyboard+controller support, D-pad/hat + left-stick fallback, action sanitization: `harvest/controls.py`, `super_metroid_rl/controls.py`
- Human/bot hot-swap chord (L+R+SELECT) + cooldown handling: `harvest/controls.py`, `harvest/harvest_bot.py`

## Human Play / Bot Swap Runtime
- Human/bot mode toggle, auto-bot fallback when disabled, hotswap cancel frames safety behavior: `harvest/harvest_bot.py`

## Speed / Fast-Forward Controls
- `[ ]` speed adjust + TAB fast-forward during recording: `harvest/task_recorder.py`
- `[ ]` speed adjust + TAB fast-forward during play: `harvest/harvest_bot.py`

## Save / Load State UX
- F5 save + F9 quick-load during play: `harvest/harvest_bot.py`
- F5 save in RAM tooling: `harvest/find_ram.py`
- State naming conventions + list/rename/record flow: `super_metroid_rl/state_manager.py`
- Auto state save per room transition during replay extraction: `super_metroid_rl/recording/extractor.py`

## Recording (Human Demos)
- JSON input recording (frame-wise action arrays) + end-state capture: `harvest/task_recorder.py`
- BK2 recording with temp folder + finalize wait + recovery handling: `super_metroid_rl/recording/recorder.py`
- Manifested demo metadata (timestamps, tags, routes): `super_metroid_rl/record_tasker.py`, `super_metroid_rl/recording/manifest.py`

## Playback
- Task replay/test harness for JSON recordings: `harvest/task_recorder.py`, `harvest/recorded_task.py`
- BK2 movie replay with `retro.Movie` + Actions.ALL: `super_metroid_rl/replay_demo.py`
- BK2 replay for extraction + auto state save: `super_metroid_rl/recording/extractor.py`

## HUD / Debug Overlays
- HUD overlays (date/time/money/goal + pressed buttons): `harvest/harvest_bot.py`, `harvest/task_recorder.py`
- Recording HUD (state name, HP, frame count, blinking REC): `super_metroid_rl/recording/recorder.py`, `super_metroid_rl/record_tasker.py`

## RAM Watch / Analysis Tools
- RAM correlation recorder/analyzer to discover addresses: `harvest/find_ram.py`
- Additional RAM scan/dump tooling: `harvest/scan_ram.py`, `harvest/scan_ram_v2.py`, `harvest/dump_ram_more.py`

## Training / Model Structure
- Observation wrappers (frame stack, state augmentation) + reward shaping: `super_metroid_rl/metroid_env.py`
- Training scripts + model paths: `super_metroid_rl/train_*.py`, `super_metroid_rl/models/`, `super_metroid_rl/logs/`
- Generic task/harness runtime (determinism hooks + observation cache): `harvest/harness_runtime.py`

## Headless / SDL Setup
- SDL driver + dummy audio/headless toggles: `harvest/run_bot.sh`, `harvest/find_ram.py`
- X11 forcing for SDL: `super_metroid_rl/controls.py`, `super_metroid_rl/recording/recorder.py`
