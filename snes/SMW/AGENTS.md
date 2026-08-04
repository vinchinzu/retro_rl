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

- Start from emulator-backed evidence for autoplay and speedrun behavior.
  Use SMWDisX, SMW Central, Data Crystal, and the C port as references, but
  confirm important RAM and ROM claims against a local ROM/state.
- Keep the first playable target small: one stable-retro level state, one human
  recording, one replay verifier, then hillclimb. Expand to chained any% only
  after segment completion is reliable.
- Do not import leaked Nintendo source. Use public reverse-engineered
  disassemblies, public tools, and locally owned ROM-derived assets only.
