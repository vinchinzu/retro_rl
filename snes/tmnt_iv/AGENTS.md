# Agent Instructions — tmnt_iv

SNES TMNT IV (M8 continuous hard clear). Shared combat helpers:
`retro_harness.combat` / `segment_runner`. Docs: `docs/STATUS.md`,
`docs/plan.md`, `docs/ASSIST_CONTRACT.md`, `docs/CLEAN_PLAYBOOK.md`.
Tracker: `bd ready -l tmnt_iv`.

## Commands

```bash
uv run python -m tmnt_iv.scripts.setup_rom
uv run python -m tmnt_iv.scripts.boot_probe

# Stage Clean suites
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage1_clean --suite
uv run python -m tmnt_iv.scripts.probe_stage2_clean --suite
uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite

# Full hard run (assisted default; Clean never clobbers assisted)
uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run
uv run python -m tmnt_iv.scripts.record_full_hard_run --clean --dry-run

# Segment runners: run_stage{N}_segment / run_stage{N}_bridge
# Raph grind states (char 8): capture_raph_states
# Local knob agent: run_local_grind_agent --focus slash --max-trials 2
uv run pytest tmnt_iv/tests -q
```

## Immediate goal

**Bronze / Clean** unassisted full run (maturity stays M8). Alleycat
Clean suite is 2/4 (BOSS+LATE); next is Stage 2 REACH then Sewer
`LiveHardStage3`. Do **not** re-open Stage 1 hazard jump-dodge, global
pizza seek, or sewer dumpster thrash.

## Traps

| Trap | Lesson |
|------|--------|
| Mash START after Stage 1 HUD | Pauses game |
| Special (**A**) | Drains HP — avoid |
| Global pizza seek all stages | Soft-locks Skull & Crossbones; scope by stage |
| Blind `RIGHT+Y` | Stutter; `pickup_every=0`; PizzaSeek owns boxes |
| Checkpoint-only tuning | Also prove power-on / continuous entry |
| Port Slash spin=40 blindly | Continuous damage regressed (keep 52) |
| Mid-run knob w/o full dry-run | Route desync |
| Clean artifact stems | Use `retro_harness.artifacts.clean_artifact_stem`; never overwrite assisted |

RAM: `docs/ram_map.md`. Ready work: `bd ready -l tmnt_iv`.
