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
- Full-run process: `docs/FULL_RUN_PROCESS.md`

## Shared Layout

- `retro_harness/`: shared emulator, input, state, recording, and scripted-
  completion helpers (`GameState`/combat/cursor/segment runners, ROM setup).
  Facades: `retro_harness.snes`, `retro_harness.nes`. Genre subdomains:
  `retro_harness.platformer`, `retro_harness.fighters`,
  `retro_harness.adventure`. Editor bridge under `retro_harness/editor/`;
  launch with `uv run python -m retro_harness.editor_launcher --list`
- `snes/<game>/`, `nes/<game>/`: game-specific code, integrations, docs, assets
- `roms/`: shared ROM storage (gitignored; NES + SNES library zips)
- `docs/manifests/`: machine-readable game manifests
- Hygiene / agent-context budget: `docs/REPO_HYGIENE.md`

Games live under console folders but keep package import names
(`import alttp`, `import super_metroid`, `import smb`). Pytest/pythonpath
includes `snes/` and `nes/` (see root `conftest.py` / `pyproject.toml`).
Path helpers: `retro_harness.repo.resolve_game_dir`,
`retro_harness.repo.ensure_import_paths`.

Authoritative trees include `snes/super_metroid/`, `snes/SMW/`,
`snes/harvest/`, `snes/alttp/`, `snes/smz3/`, `snes/tmnt_iv/`,
`nes/tmnt_i/`–`nes/tmnt_iii/`, `nes/zelda_i/`, `nes/zelda_ii/`,
`nes/metroid/`, `nes/smb/`, `nes/smb3/`. Do not invent
`super_metroid_rl/` or `super_mario_bros/`.

## Organization Rules

- Keep game-specific code, states, docs, plans, screenshots, and debug output
  inside the owning game directory under `snes/` or `nes/`.
- Save states belong under `<console>/<game>/custom_integrations/<GameId>/`.
  Do not leave `.state` files in the repo root.
- Game-specific docs and runbooks belong under `<console>/<game>/docs/`.
  Delete stale one-offs; do not accumulate archive trees.
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
