> Historical feature inventory. Paths use monorepo layout (`snes/harvest/`,
> `snes/super_metroid/`). Some modules may have moved under package subdirs
> (`harvest/runtime/…`); treat paths as pointers, not a live import map.

# Emulator-Specific Features Inventory (Harvest + Super Metroid)

This list captures emulator-facing features already implemented in
`snes/harvest/` and `snes/super_metroid/` that should be ported to a shared
root harness for consistent recording, human play, playback, training, and
model workflows.

## Input + Controller Handling
- Shared SNES button mapping, keyboard+controller support, D-pad/hat + left-stick fallback, action sanitization: harvest controls / `super_metroid` controls
- Human/bot hot-swap chord (L+R+SELECT) + cooldown handling: harvest controls / `harvest.runtime.harvest_bot`

## Human Play / Bot Swap Runtime
- Human/bot mode toggle, auto-bot fallback when disabled, hotswap cancel frames safety behavior: `harvest.runtime.harvest_bot`

## Speed / Fast-Forward Controls
- `[ ]` speed adjust + TAB fast-forward during recording: harvest task_recorder
- `[ ]` speed adjust + TAB fast-forward during play: harvest_bot

## Save / Load State UX
- F5 save + F9 quick-load during play: harvest_bot
- F5 save in RAM tooling: harvest `utils/find_ram.py`
- State naming conventions + list/rename/record flow: super_metroid state_manager
- Auto state save per room transition during replay extraction: super_metroid recording/extractor

## Recording (Human Demos)
- JSON input recording (frame-wise action arrays) + end-state capture: harvest task_recorder
- BK2 recording with temp folder + finalize wait + recovery handling: super_metroid recording/recorder
- Manifested demo metadata (timestamps, tags, routes): super_metroid record_tasker / recording/manifest

## Playback
- Task replay/test harness for JSON recordings: harvest task_recorder / recorded_task
- BK2 movie replay with `retro.Movie` + Actions.ALL: super_metroid replay_demo
- BK2 replay for extraction + auto state save: super_metroid recording/extractor

## HUD / Debug Overlays
- HUD overlays (date/time/money/goal + pressed buttons): harvest_bot / task_recorder
- Recording HUD (state name, HP, frame count, blinking REC): super_metroid recording/recorder

## RAM Watch / Analysis Tools
- RAM correlation recorder/analyzer: harvest `utils/find_ram.py`
- Additional RAM scan/dump tooling: harvest `utils/scan_ram*.py`, `dump_ram_more.py`

## Training / Model Structure
- Observation wrappers + reward shaping: super_metroid env / train scripts
- Training scripts + model paths: super_metroid models/ logs/
- Generic task/harness runtime: harvest harness_runtime

## Headless / SDL Setup
- SDL driver + dummy audio/headless toggles: `snes/harvest/run_bot.sh`, find_ram
- X11 forcing for SDL: super_metroid controls / recording
