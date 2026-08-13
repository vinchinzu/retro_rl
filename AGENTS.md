# Agent Instructions — retro_rl

Multi-game **NES + SNES** automation monorepo. Repo-wide rules only; keep
game-specific workflow and status in the owning game directory. Spine:
`docs/{VISION,ROADMAP,DEVELOPMENT_LADDER,BENCHMARK_SPEC,PROGRAM_STATUS,GAME_MATRIX,FULL_RUN_PROCESS}.md`
(matrix from `docs/manifests/*.yaml`). Hygiene: `docs/REPO_HYGIENE.md`.

## Shared Layout

- `retro_harness/`: shared emulator, input, state, recording, scripted-
  completion. Facades `.snes`/`.nes`; genres `.platformer`/`.fighters`/
  `.adventure`. Editor: `uv run python -m retro_harness.editor_launcher --list`
- `snes/<game>/`, `nes/<game>/`: game code, docs, assets
- `roms/` (gitignored), `docs/manifests/` (game board YAML)

Short import names (`import alttp`, `import smb`). Pytest path includes
`snes/`/`nes/`; nested packages via `repo.ensure_import_paths`. Trees:
`snes/{super_metroid,SMW,harvest,hals_golf,alttp,alttp_rando,sm_rando,smz3,tmnt_iv}/`,
`nes/{tmnt_i..iii,zelda_i,zelda_ii,metroid,smb,smb3}/`. Do not invent
`super_metroid_rl/` or `super_mario_bros/`.

## Organization Rules

- Game code/states/docs/plans/screenshots/debug → owning `<console>/<game>/`.
  Save states under `…/custom_integrations/<GameId>/` (never repo root).
  Artifacts (`models/`, `logs/`, `recordings/`, `debug_*`, `maps/`) stay in-game.
- Local docs: `STATUS.md` = verified + one maturity gate; `plan.md` = future;
  `AGENTS.md` = commands/traps only. Delete stale one-offs.
- Cursor agent dock: `retro_harness/editor/cursor_agent_panel.py`
  (`uv sync --extra cursor`, `CURSOR_API_KEY`). Promote shared code only after
  a second consumer; expand top-level docs only for multi-game content.
- Prefer Clean intervention; assisted published results need local
  `docs/ASSIST_CONTRACT.md`.

## Working Norms

- Prefer nearest local `AGENTS.md`; add one there instead of growing this file.
- **Soft max ~1000 LOC per source file.** Extract a focused module/helper
  before pushing past 1k; prefer deleting complexity over rearranging it.
- Shared-helper changes → update closest tests + describing docs. After
  `docs/manifests/*.yaml` edits: `uv run python docs/generate_game_matrix.py`.
- Narrowest relevant tests; include `uv run pytest tests/test_docs.py -q`
  when touching docs or manifests.
- Natural-entry: segment not route-ready until it clears from the real
  predecessor state.

## Issue tracking (bd)

**[bd](https://github.com/steveyegge/beads)** — commands, labels (`rr-`,
game/kind), STATUS-vs-beads split in [`docs/BEADS.md`](docs/BEADS.md).
Start with `bd ready`; claim one issue; `bd sync` + commit
`.beads/issues.jsonl` with matching code.

## Landing the plane

1. Update beads for remaining work; close finished issues honestly
2. Run narrowest tests for files you changed
3. `bd sync` and commit code + `.beads/issues.jsonl` together
4. Push only if requested; hand off with `bd ready` + one-line next action
