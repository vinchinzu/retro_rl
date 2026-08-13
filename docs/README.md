# Program documentation

| Document | Responsibility |
|----------|----------------|
| [VISION.md](VISION.md) | Why the project exists; scriptably beatable + solver |
| [ROADMAP.md](ROADMAP.md) | Multi-horizon strategy, NES+SNES library plan |
| [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md) | Layer stack, tapes demoted, flagship triangle |
| [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md) | Capability phases and M0–M8 maturity gates |
| [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) | Runtime/intervention rules + seed-robustness |
| [PROGRAM_STATUS.md](PROGRAM_STATUS.md) | Live facts and near-term priorities |
| [GAME_MATRIX.md](GAME_MATRIX.md) | All games (generated from `manifests/`) |
| [GLOSSARY.md](GLOSSARY.md) | Shared vocabulary |
| [FULL_RUN_PROCESS.md](FULL_RUN_PROCESS.md) | Segment → continuous clear process |
| [GAME_SELECTION_NOTES.md](GAME_SELECTION_NOTES.md) | Candidate / hard-game research notes |
| [REPO_HYGIENE.md](REPO_HYGIENE.md) | Agent-context budget and cleanup backlog |
| [BEADS.md](BEADS.md) | bd / beads issue-tracking rules |
| [ADDING_GAMES.md](ADDING_GAMES.md) | How to onboard a new title |

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
