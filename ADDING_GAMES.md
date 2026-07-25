# Adding a SNES Game

New games should start on the shared `retro_harness.snes` API. Do not copy an
input loop, button table, Gym compatibility wrapper, state writer, or title
screen macro from another game.

## 1. Integration files

Create this game-local layout:

```text
new_game/
├── game.py
├── roms/                  # gitignored
├── tests/
└── custom_integrations/
    └── GameName-Snes/
        ├── data.json
        ├── metadata.json
        ├── scenario.json
        ├── rom.sfc -> ../../roms/GameName.sfc
        ├── rom.sha
        └── FirstAction.state
```

ROMs and generated states remain game-local and gitignored. Reusable Python
behavior belongs in a shared package.

## 2. Minimal game definition

```python
from pathlib import Path

from retro_harness.snes import GameSpec, StartupPlan, run_startup

GAME = GameSpec("GameName-Snes", Path(__file__).parent)
STARTUP = StartupPlan.title_menu("DOWN")


def player_has_control(env, _info: dict) -> bool:
    return int(env.get_ram()[0x1234]) == 1  # replace with verified RAM truth


def create_first_action_state() -> None:
    env = GAME.make_env(None)
    try:
        result = run_startup(
            env,
            STARTUP,
            is_ready=player_has_control,
            max_cycles=2,
            max_frames=1800,
        )
        if not result.ready:
            raise RuntimeError(f"startup failed after {result.frames} frames")
        GAME.save_state(env, "FirstAction")
    finally:
        env.close()
```

For irregular menus, use the compact script form:

```python
STARTUP = StartupPlan.parse(
    "WAIT:0:120 START:2:90 DOWN:2:12 A:2:180"
)
```

Each token is `BUTTON[+BUTTON]:hold_frames:wait_frames`. `WAIT`, `NOOP`,
`NONE`, and `IDLE` create release-only delays.

## 3. Add only the layer the game needs

- Use `retro_harness`: emulator/session lifecycle, actions, input scripts,
  states, recording, RAM schemas, and task protocols.
- Add `snes_oneshot`: behavior trees, watchdogs, RAM discovery, cursor/combat
  policy, and segment completion for scripted clears.
- Add `platformer_common`: platformer progress tracking, route evaluation,
  replay, and optimizers.
- Add `fighters_common`: fighting-game environments and training.
- Keep RAM addresses, readiness checks, menu choices, maps, and policies in the
  game directory until at least two games need the same behavior.

See [the toolset boundary](retro_harness/docs/TOOLSET.md) for ownership and
compatibility details.

## 4. Verify the first seam

The first test should compile the startup plan and assert the readiness signal
against a known state or recorded trace. Then run the narrow shared tests:

```bash
uv run python -m pytest retro_harness/tests/test_actions.py \
  retro_harness/tests/test_input_script.py -q
```

## 5. Plan the route to a verified full run

After the first controllable checkpoint, use the shared
[scripted full-run process](snes_oneshot/docs/FULL_RUN_PROCESS.md). Create the
game-local `AGENTS.md`, `docs/STATUS.md`, `docs/plan.md`, and `docs/ram_map.md`
before the project accumulates ad hoc scripts and states.

The first segment acceptance test should include both:

- a clean development checkpoint; and
- a state captured from the real predecessor route.

If the project writes RAM for health, ammo, lives, or another assist, add
`docs/ASSIST_CONTRACT.md` before implementing it. Resource assists should
refill only naturally unlocked capacity and must not silently grant item,
stage, boss, door, or other progression flags.

Do not overwrite the previous successful full-run report during experiments.
Write a candidate report/log, validate it, and promote it only after the
reset-to-ending integrity checks pass.
