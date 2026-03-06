# Agent Instructions — retro_rl

Multi-game emulator/RL monorepo. This root file is for repo-wide rules only.
Keep game-specific workflow, status, and implementation detail in the
corresponding game directory.

## Shared Layout

- `retro_harness/`: shared emulator/session/input utilities
- `fighters_common/`: shared fighting-game env/training code
- `platformer_common/`: shared platformer runtime/optimizer code
- `<game>/`: game-specific code, integrations, docs, assets, and outputs
- `roms/`: shared ROM storage (gitignored)

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
- Only add or expand top-level docs when the content is genuinely shared across
  multiple games.

## Working Norms

- Prefer the nearest local `AGENTS.md` for game-specific instructions.
  If a game accumulates nontrivial local rules and lacks one, add it there
  instead of growing this root file.
- When changing shared helpers, update the closest tests and any docs that
  describe their behavior.
- Run the narrowest relevant tests for the code you changed.
