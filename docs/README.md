# Program documentation

| Document | Responsibility |
|----------|----------------|
| [VISION.md](VISION.md) | Why the project exists; scriptably beatable |
| [ROADMAP.md](ROADMAP.md) | Multi-horizon strategy, NES+SNES library plan |
| [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md) | Capability phases and M0–M8 maturity gates |
| [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) | Stable runtime and intervention rules |
| [PROGRAM_STATUS.md](PROGRAM_STATUS.md) | Live facts and near-term priorities |
| [GAME_MATRIX.md](GAME_MATRIX.md) | All games (generated from `manifests/`) |
| [GLOSSARY.md](GLOSSARY.md) | Shared vocabulary |

Related process docs outside this folder:

- [../ADDING_GAMES.md](../ADDING_GAMES.md) — how to onboard a new title
- [../snes_oneshot/docs/FULL_RUN_PROCESS.md](../snes_oneshot/docs/FULL_RUN_PROCESS.md) — segment → continuous clear process
- [../snes_oneshot/docs/GAME_SELECTION_NOTES.md](../snes_oneshot/docs/GAME_SELECTION_NOTES.md) — research notes

Regenerate the matrix after editing manifests:

```bash
uv run python docs/generate_game_matrix.py
uv run pytest tests/test_docs.py -q
```

Local per-game docs stay under `<game>/docs/`:

- `STATUS.md` — verified facts + exactly one maturity gate
- `plan.md` — future work
- `ram_map.md` — addresses and meanings
- optional `ASSIST_CONTRACT.md` — required before assisted published results
