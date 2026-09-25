# Agent instructions: Super Mario World

Root rules: [root AGENTS.md](../../AGENTS.md). Docs: `docs/STATUS.md`,
`docs/autoplay_speedrun_plan.md`, `docs/ARCHITECTURE.md`.
There is no `docs/plan.md`. STATUS does not name an M-number.

## Commands

```bash
uv run python -m SMW list-levels
uv run python -m SMW -l smw_yoshi_island_1 play
uv run python -m SMW -l smw_yoshi_island_1 verify --actions recording.json
uv run python -m SMW capture-state --from <alias> --name <StateName>
uv run python -m SMW chain-yi
./play_speedrun.sh
```

ROM: `roms/smw.sfc` or
`custom_integrations/SuperMarioWorld-Snes-v0/rom.sfc`.
SHA1 `6b47bb75d16514b6a476aa0c73a683a2a4c18765`.

## Traps

- Save states belong in `custom_integrations/SuperMarioWorld-Snes-v0/`.
- Prefer human controller recordings over TAS or glitch movies for route
  skills. Hillclimb only frame-shaves.
- Capture anchors in-level (F5, GameMode `0x14`), then
  `python -m SMW -l <alias> play`.
- Iggy: do not extract from package `YoshiIsland4` alone. After a standalone
  YI4 clear the castle path is closed. The north pipe is Donut Plains 1
  (`trans=0x15`). Natural path: `chain-yi` to `IggysCastle` (`trans=0x25`).
- YI4 clear seed is `recording_004_chained_clear.json` from
  `Chained_YoshiIsland4`. Package YI4 seeds die on chained entry.
- Evaluator resyncs custom `*.state` after reset. Chained clears include a
  1-frame idle pad so verify matches play.
- Do not import leaked Nintendo source. Confirm RAM against a local ROM.
- TAS ACE and red current-core movies are not movement skills. See STATUS.

## Pointers

[docs/STATUS.md](docs/STATUS.md) ·
[docs/autoplay_speedrun_plan.md](docs/autoplay_speedrun_plan.md) ·
[docs/ram_decomposition.md](docs/ram_decomposition.md) ·
[docs/external_tools.md](docs/external_tools.md) ·
[custom_integrations/SuperMarioWorld-Snes-v0/README_STATES.md](custom_integrations/SuperMarioWorld-Snes-v0/README_STATES.md)

Third-party clones under `refs/` and `tools/external/` are gitignored.
