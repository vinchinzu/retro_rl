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
- High-level structure bar only (e.g. soft max ~1000 LOC / file; extract
  thrash residuals instead of growing monofiles) — not full review essays

## Slim backlog

| Priority | Path | Target | Status |
|---------:|------|--------|--------|
| 1 | `snes/super_metroid/AGENTS.md` | ~50–60 lines (commands + traps) | **done** (session loop in `.grok/skills/sm-session/`) |
| 2a | `snes/harvest/AGENTS.md`, `snes/hals_golf/AGENTS.md` | ~50 each | **done** |
| 2b | Fat game AGENTS (`tmnt_iv`, `zelda_i`, MK, MKII, `smb`, `alttp`, `smz3`) | ~50 each | **done** |
| 3 | Root AGENTS further trim | ~45–60 | **done** (~59 lines; beads → `docs/BEADS.md`) |

## Engineering backlog (not docs)

| Item | Notes |
|------|--------|
| Unify video writers | `video.FrameVideoWriter` is the canonical pipe |
| Name clarity | **done** — `video.CaptureSession` (showcase/continuous) vs `recorder.RecordingSession` (labeled saves) |
| `ladder.py` vs manifests | **done** — ladder loads from `docs/manifests/*.yaml` `setup:` blocks |
| Nested package import roots | **done** — layout discovery in `repo.discover_nested_package_roots` (no slug map) |
| Artifact gitignore | **done** — `**/probe*.png`, `debug_frames/`, SM/tmnt/FF probe globs |
| Shared game layout + Clean stems | **done** — `game_layout.game_paths`, `artifacts.clean_artifact_stem` / `recording_artifacts` |
| Cross-game CLI clones | **done** — `boot_probe`, `setup_rom_cli`, `env.reset_obs`, `input_script.period_script`; fighter `watch`/`validate_*` live in `retro_harness.fighters` |


## Import cheat sheet (scripted completion)

Prefer **submodule** imports (not the package-root barrel) for new code.
Dual orchestration stacks are documented in
[retro_harness/docs/TOOLSET.md](../retro_harness/docs/TOOLSET.md) — do not mix
`protocol.WorldState`/`Task` with `ram_state.GameState`/`BehaviorNode` without
an adapter.

```python
from retro_harness.actions import buttons, idle_action
from retro_harness.input_script import FrameAction, StartupPlan, run_startup
from retro_harness.ram_state import GameMode, GameState, diff_changed, snapshot
from retro_harness.bot_runner import Selector, Sequence, StuckDetector  # BT oneshot
from retro_harness.bot_runner import BotRunner, TaskSequencer            # PlaySession Task
from retro_harness.protocol import Task, WorldState
from retro_harness.segment_runner import configure_headless, SegmentTracker
from retro_harness.combat import fight_nearest_action, build_segment_tree
from retro_harness.cursor import step_toward_target
from retro_harness.env import setup_game_rom, make_env, GameSpec
from retro_harness.video import CaptureSession, FrameVideoWriter, render_footer_frame
# Genre subdomains (optional):
# from retro_harness.platformer import Evaluator, LevelConfig
# from retro_harness.fighters import FightingEnv, get_game_config
# from retro_harness.adventure import RouteGraph, shortest_path
# Game-owned platformer packs (register LevelConfigs on import):
# import super_metroid.platformer_levels
# Standard paths + Clean artifacts (prefer over copy-paste in paths.py):
# from retro_harness.game_layout import game_paths
# from retro_harness.artifacts import clean_artifact_stem, recording_artifacts
```

## Done recently (docs)

- Folded `snes_oneshot/` into `retro_harness/` (see historical import map in git).
- Removed root redirects (`BENCHMARK_STATUS.md`, `ARCHITECTURE_AND_CLEANUP_PLAN.md`).
- Moved onboarding to [ADDING_GAMES.md](ADDING_GAMES.md).
- Deleted dual CLAUDE files and game/task archive trees.
- Normalized DKC to `AGENTS.md` + `docs/STATUS.md` + `docs/plan.md`.
