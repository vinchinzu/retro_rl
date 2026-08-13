# Agent Instructions - SMW

Root repo-wide instructions live in [../AGENTS.md](../../AGENTS.md).

## Organization

- Save states belong in `custom_integrations/SuperMarioWorld-Snes-v0/`.
- Active docs and runbooks belong in `docs/`.
- Delete stale one-offs; do not accumulate archive trees.
- Emulator recordings belong in `recordings/`; optimizer output belongs in
  `optimizer/`; model checkpoints belong in `models/`; logs belong in `logs/`.
- ROM files stay out of git. Put the local ROM in `roms/` or symlink it as
  `custom_integrations/SuperMarioWorld-Snes-v0/rom.sfc` when needed.
- Third-party source clones live under `refs/` and `tools/external/`. They are
  intentionally gitignored; update `docs/external_tools.md` when refreshing
  them.

## Active Stack

- `python -m SMW`: local CLI entrypoint, delegating platformer commands to
  `retro_harness.platformer.runner`.
- `retro_harness/platformer/levels/super_mario_world.py`: published SMW RAM/action
  config and initial stable-retro state registrations.
- `docs/ARCHITECTURE.md`: canonical architecture for autoplay, speedrun,
  editor, RAM decomposition, modding, and C-port work.
- `docs/autoplay_speedrun_plan.md`: first implementation sequence.
- `docs/ram_decomposition.md`: RAM map priorities and source-of-truth policy.
- `docs/external_tools.md`: cloned/reference tooling manifest.

## Working Norms

- Prefer **human controller multi-recordings** over TAS/glitch movies for route
  skills. Hillclimb only frame-shaves; real lines come from re-plays and picks.
- Capture missing anchors with `python -m SMW capture-state --from <alias>
  --name <StateName>` (F5 inside level, GameMode `0x14`), then
  `python -m SMW -l <alias> play`.
- **Iggy's Castle:** do not extract from package `YoshiIsland4` alone. After a
  standalone YI4 clear the castle path is closed; the north pipe is Donut
  Plains 1 (`trans=0x15`). Natural path: `chain-yi` → `IggysCastle` (`trans=0x25`,
  outdoor door; fences+lava are interior).
- **Yoshi Island chain:** `uv run python -m SMW chain-yi` rebuilds
  `Chained_YoshiIsland3/4`, `Chained_AfterYI4_OW`, and `IggysCastle`.
  YI4 clear seed: `recording_004_chained_clear.json` (from
  `--state Chained_YoshiIsland4`). Package YI4 seeds die on chained entry.
  Route: `smw_yoshi_island` / `yi_chain`. States: `README_STATES.md`.
- **Custom-state play vs verify:** Evaluator resyncs custom `*.state` after
  reset (drops stable-retro free frame). Chained clears include a 1-frame
  idle pad so verify matches play.
- Start from emulator-backed evidence for autoplay and speedrun behavior.
  Use SMWDisX, SMW Central, Data Crystal, and the C port as references, but
  confirm important RAM and ROM claims against a local ROM/state.
- Keep the first playable target small: one stable-retro level state, one human
  recording, one replay verifier. Expand to chained any% only after segment
  completion is reliable.
- Do not import leaked Nintendo source. Use public reverse-engineered
  disassemblies, public tools, and locally owned ROM-derived assets only.
