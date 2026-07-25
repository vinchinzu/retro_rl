# Program documentation

| Document | Responsibility |
|----------|----------------|
| [VISION.md](VISION.md) | Why the project exists; scriptably beatable |
| [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md) | Capability phases and M0–M8 maturity gates |
| [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) | Stable runtime and intervention rules |
| [PROGRAM_STATUS.md](PROGRAM_STATUS.md) | Live facts and near-term priorities |
| [GAME_MATRIX.md](GAME_MATRIX.md) | All games (generated from `manifests/`) |
| [GLOSSARY.md](GLOSSARY.md) | Shared vocabulary |

Regenerate the matrix after editing manifests:

```bash
uv run python docs/generate_game_matrix.py
uv run pytest tests/test_docs.py -q
```
