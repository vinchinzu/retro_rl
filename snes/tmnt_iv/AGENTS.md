# Agent Instructions — tmnt_iv

SNES TMNT IV (M8 continuous hard clear). Shared combat helpers:
`retro_harness.combat` / `segment_runner`. Docs: `CONTEXT.md`, `docs/STATUS.md`,
`docs/plan.md`, `docs/ASSIST_CONTRACT.md`, `docs/CLEAN_PLAYBOOK.md`.
Raph hard speed: `docs/RAPH_SPEED_HANDOFF.md`,
`docs/SPEEDRUN_STRATEGIES.md` (`rr-iprz`). Tracker: `bd ready -l tmnt_iv`.

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

**Bronze / Clean** unassisted full run (maturity stays M8). Parallel
assisted Raphael speed+damage (`rr-iprz`): next is **`rr-iprz.5`
Starbase stall** — recover Fast ≤23,072f / 863 and Boss9 6,300f / 144
without restoring the Diag rail loop (24,645f). Not Slash. Alleycat
Clean suite is 2/4 (BOSS+LATE). Do **not** re-open Stage 1 hazard
jump-dodge, global pizza seek, sewer dumpster thrash, or Slash spin=40.

## Traps

| Trap | Lesson |
|------|--------|
| Mash START after Stage 1 HUD | Pauses game |
| Special (**A**) | Drains HP — avoid |
| Global pizza seek all stages | Soft-locks Skull & Crossbones; scope by stage |
| Blind `RIGHT+Y` | Stutter; `pickup_every=0`; PizzaSeek owns boxes |
| Checkpoint-only tuning | Also prove power-on / continuous entry |
| Port Slash spin=40 blindly | Continuous damage regressed (keep 52) |
| Rewrite Slash from KEEP trace | Four algs + three patches lost to 9,595/435 |
| Sewer-like dumpster skip on Starbase | 40k timeout; keep DOWN+JUMP (x=126 collision) |
| Starbase dumpster on right rail (x≥220) | Diag 7k loop; hold RIGHT. 96f budget / form-1 latch then RIGHT both 40k-timeout Diag |
| `raph_starbase_close_gap` period < 4 | `%3` timeout / `%2` jump-lock |
| Mid-run knob w/o full dry-run | Route desync |
| Clean artifact stems | Use `retro_harness.artifacts.clean_artifact_stem`; never overwrite assisted |

RAM: `docs/ram_map.md`. Ready work: `bd ready -l tmnt_iv`.
