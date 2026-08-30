# Agent Instructions — tmnt_iv

SNES TMNT IV (M8 continuous hard clear). Shared combat helpers:
`retro_harness.combat` / `segment_runner`. Docs: `CONTEXT.md`,
`docs/ARCHITECTURE.md`, `docs/STATUS.md`, `docs/plan.md`,
`docs/ASSIST_CONTRACT.md`, `docs/CLEAN_PLAYBOOK.md`.
Raph hard speed: `docs/RAPH_SPEED_HANDOFF.md`,
`docs/SPEEDRUN_STRATEGIES.md` (`rr-iprz`). Tracker: `bd ready -l tmnt_iv`.

## Commands

```bash
uv run python -m tmnt_iv.scripts.setup_rom
uv run python -m tmnt_iv.scripts.boot_probe

# Stage Clean suites (human stage 1–3)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_clean --stage 1 --suite

# Full hard run (assisted default; Clean never clobbers assisted)
# Video: shared VideoRecorder, 1080p60 YouTube layout ( --native-video to opt out)
uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run
uv run python -m tmnt_iv.scripts.record_full_hard_run
uv run python -m tmnt_iv.scripts.record_full_hard_run --clean --dry-run

# Segment / bridge: run_segment --stage N / run_bridge --to {2,3}
# Raph grind states (char 8): capture_raph_states
# Local knob agent: run_local_grind_agent --focus slash --max-trials 2
uv run pytest tmnt_iv/tests -q
```

## Immediate goal

**Bronze / Clean** unassisted full run (maturity stays M8). `run_trial`
is the loop. Alleycat suite **2/4** (LATE+Boss2). Next Clean: Stage2 /
stage1_clear 0x5E 24-dmg (`rr-1bmx`). Sewer LiveHard reaches Rat King at
10 HP then KO (`rr-t4s3`). Parallel Raph speed (`rr-iprz.5`). Do **not**
re-open Stage 1 hazard jump-dodge, global pizza seek, sewer dumpster
thrash, TTC/hy≥180 spike hop, or Slash spin=40.

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
| Starbase x=207 dumpster forever | Continuous auto-scroll loop (dmg stuck, stall_right/up/up_right). Three cycles then RIGHT (`starbase_unstick_right`). Do not skip x=126 |
| `raph_starbase_close_gap` period < 4 | `%3` timeout / `%2` jump-lock |
| Mid-run knob w/o full dry-run | Route desync |
| Clean artifact stems | Use `retro_harness.artifacts.clean_artifact_stem`; never overwrite assisted |
| Clone `run_stageN_segment.py` | Add a `StageSpec` / `CleanProbeSpec` / `BridgeSpec` |
| Clone a fourth emulator loop | `run/trial.py` `run_trial` is the loop |
| TTC / hy≥180 sewer hop | 6 jumps, four 16s, never reached RK. Keep `adx≤56` + air-frame lock |
| Fail 0x0B fade HP as unlabeled pizza | Metalhead already dead; LATE/Boss2 were `stage_advance` |

RAM: `docs/ram_map.md`. Ready work: `bd ready -l tmnt_iv`.
