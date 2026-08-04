# Repo hygiene

Cleanup policy. Live process: [FULL_RUN_PROCESS.md](FULL_RUN_PROCESS.md).
Maturity gates: [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md).

## Doc layout

| Location | Owns |
|----------|------|
| Root `README.md`, `AGENTS.md` | Entry + agent rules only |
| `docs/*.md` | Program spine (vision, roadmap, status, process) |
| `docs/manifests/*.yaml` | Machine-readable game board source |
| `<console>/<game>/AGENTS.md` | Commands + traps only |
| `<console>/<game>/docs/STATUS.md` | Verified facts + one maturity gate |
| `<console>/<game>/docs/plan.md` | Future work |
| Game process / architecture docs | Essays, queues, RAM maps as needed |

Do **not** keep deprecated trees (`docs/archive/`, `tasks/archive/`, dual
`CLAUDE.md`). Delete completed cards and stale session notes.

## Agent-context budget

Agents auto-load hierarchical `AGENTS.md`. Every extra section costs tokens on
**every** turn in that tree.

### Hard targets

| File | Soft max lines |
|------|---------------:|
| Root `AGENTS.md` | ~45 |
| Game / package `AGENTS.md` | ~50–60 |
| Dual `CLAUDE.md` | **0** (one AGENTS only) |

### Never put in AGENTS

- Frame counts, verified tip history, scorecards → `STATUS.md`
- Ticket boards / “next card” checklists → `plan.md` / queue
- Full RAM tables → `ram_map.md`
- Module encyclopedias / architecture diagrams → architecture docs
- Process essays and anti-pattern catalogs → process docs

### Keep in AGENTS

- The 3–8 commands you actually run every day
- Burned-once traps (wrong paths, settle frames, clean-vs-assisted stems)
- One-line pointers to STATUS / plan / process

## Slim backlog

| Priority | Path | Target | Status |
|---------:|------|--------|--------|
| 1 | `snes/super_metroid/AGENTS.md` | ~50–60 lines (commands + traps) | **done** (~58 lines) |
| 2 | Fat game AGENTS (`harvest`, `hals_golf`, `tmnt_iv`, `zelda_i`, MK, `smb`, `alttp`, `smz3`) | ~50 each | open |
| 3 | Root AGENTS further trim | ~40 | open |

## Engineering backlog (not docs)

| Item | Notes |
|------|--------|
| Unify video writers | `video.FrameVideoWriter` is the canonical pipe |
| Name clarity | `video.RecordingSession` vs `recorder.RecordingSession` |
| `ladder.py` vs manifests | Hardcoded ladder is stale vs `docs/manifests/*.yaml` |
| Artifact gitignore | Probe PNG spam under SM / tmnt_iv / final_fight — keep goldens only |

## Import cheat sheet (scripted completion)

```python
from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction, StartupPlan, run_startup
from retro_harness.ram_state import GameMode, GameState, diff_changed, snapshot
from retro_harness.bot_runner import Selector, Sequence, StuckDetector
from retro_harness.segment_runner import configure_headless, SegmentTracker
from retro_harness.combat import fight_nearest_action, build_segment_tree
from retro_harness.cursor import step_toward_target
from retro_harness.env import setup_game_rom, make_env, GameSpec
from retro_harness.video import FrameVideoWriter, RecordingSession, render_footer_frame
# Genre subdomains (optional):
# from retro_harness.platformer import Evaluator, LevelConfig
# from retro_harness.fighters import FightingEnv, get_game_config
# from retro_harness.adventure import RouteGraph, shortest_path
```

## Done recently (docs)

- Folded `snes_oneshot/` into `retro_harness/` (see historical import map in git).
- Removed root redirects (`BENCHMARK_STATUS.md`, `ARCHITECTURE_AND_CLEANUP_PLAN.md`).
- Moved onboarding to [ADDING_GAMES.md](ADDING_GAMES.md).
- Deleted dual CLAUDE files and game/task archive trees.
- Normalized DKC to `AGENTS.md` + `docs/STATUS.md` + `docs/plan.md`.
