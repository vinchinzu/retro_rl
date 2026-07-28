# Agent Instructions — retro_rl

Multi-game **NES + SNES** automation monorepo. This root file is for repo-wide
rules only. Keep game-specific workflow, status, and implementation detail in
the corresponding game directory.

Program spine:

- Vision: `docs/VISION.md`
- Roadmap: `docs/ROADMAP.md`
- Maturity / phases: `docs/DEVELOPMENT_LADDER.md`
- Benchmark rules: `docs/BENCHMARK_SPEC.md`
- Live status: `docs/PROGRAM_STATUS.md`
- Game board: `docs/GAME_MATRIX.md` (from `docs/manifests/*.yaml`)
- Full-run process: `snes_oneshot/docs/FULL_RUN_PROCESS.md`

## Shared Layout

- `retro_harness/`: shared emulator/session/input/state utilities;
  `retro_harness.snes` is the concise new-game facade (`GameSpec`, named
  actions, `StartupPlan`, menu/input scripts); `retro_harness.nes` for NES
  helpers where present
- `retro_harness/editor/`: game-agnostic Qt editor ↔ subprocess
  bridge (stdio JSON protocol, script segments, map RGBA helpers, recording,
  `EmbeddedEmulatorPanelBase` for docked emulator UI, `CursorAgentPanel` for
  embedded Cursor SDK agent sessions)
- `retro_harness/editor_launcher.py`: shared entry point to launch registered
  game editors (`uv run python -m retro_harness.editor_launcher --list`)
- `fighters_common/`: shared fighting-game env/training code
- `platformer_common/`: shared platformer runtime/optimizer code
- `adventure_common/`: shared capability-aware route graphs (first consumer
  `zelda_i/`; promote richer APIs after a second consumer)
- `snes_oneshot/`: shared scripted-completion helpers (GameState, behavior
  trees, combat/cursor policy, watchdog, RAM discovery); historical package
  name — prefer “scripted completion” in human-facing prose; process applies
  to NES and SNES
- `<game>/`: game-specific code, integrations, docs, assets, and outputs
- `roms/`: shared ROM storage (gitignored)
  - `roms/Nintendo/NES/`: NES library zips
  - `roms/Nintendo/SNES` → `roms/Super Nintendo`: SNES library zips
- `docs/manifests/`: machine-readable game manifests

Authoritative directories include: `super_metroid/`, `SMW/`, `harvest/`,
`alttp/`, `tmnt_i/`–`tmnt_iv/`, `zelda_i/`, `zelda_ii/`, `metroid/`, and NES M1
trees (`smb/`, `smb3/`, `mega_man_2/`, `castlevania/`, `contra/`, `ducktales/`,
`kirby_adventure/`, `punch_out/`).
Do not invent `super_metroid_rl/` or `super_mario_bros/` paths in this checkout
(NES SMB lives under `smb/` / `smb3/`).

## Organization Rules

- Keep game-specific code, states, docs, plans, screenshots, and debug output
  inside the owning game directory.
- Save states belong under `<game>/custom_integrations/<GameId>/`.
  Do not leave `.state` files in the repo root.
- Game-specific docs and runbooks belong under `<game>/docs/`.
  Archive stale one-offs under `<game>/docs/archive/` or
  `<game>/scripts/archive/`.
- Generated artifacts belong in the owning game folder (`models/`, `logs/`,
  `recordings/`, `debug_*`, `maps/`, and similar), not in the repo root.
- Local docs split: `STATUS.md` = verified facts + one maturity gate;
  `plan.md` = future work; `AGENTS.md` = commands and traps only.
- Game editors can embed the shared Cursor SDK agent dock from
  `retro_harness/editor/cursor_agent_panel.py` (View → Agent Panel). Install
  with `uv sync --extra cursor` and set `CURSOR_API_KEY`.
- Promote shared abstractions only after a second consumer exists.
- Only add or expand top-level docs when the content is genuinely shared across
  multiple games. Prefer updating `docs/` spine files over new root markdown.
- Prefer Clean intervention; any assist used in a published result needs a
  local `docs/ASSIST_CONTRACT.md`.

## Working Norms

- Prefer the nearest local `AGENTS.md` for game-specific instructions.
  If a game accumulates nontrivial local rules and lacks one, add it there
  instead of growing this root file.
- When changing shared helpers, update the closest tests and any docs that
  describe their behavior.
- After editing `docs/manifests/*.yaml`, run
  `uv run python docs/generate_game_matrix.py`.
- Run the narrowest relevant tests for the code you changed; include
  `uv run pytest tests/test_docs.py -q` when touching docs or manifests.
- Follow the natural-entry rule: a segment is not route-ready until it clears
  from the real predecessor state.
