# Adding New Games to retro_rl

This guide explains how to add a new SNES game to the project.

## Prerequisites

1. Run `./setup.sh` from the project root to create the shared virtual environment
2. Have your ROM file ready (legally obtained)

## Steps

### 1. Create Game Directory

```bash
mkdir -p new_game/custom_integrations/GameName-Snes
mkdir -p new_game/roms
```

### 2. Set Up ROM

```bash
# Copy ROM to roms directory (git-ignored)
cp /path/to/rom.sfc new_game/roms/

# Create symlink in custom_integrations
ln -s "$(pwd)/new_game/roms/rom.sfc" new_game/custom_integrations/GameName-Snes/rom.sfc

# Generate SHA hash
sha1sum new_game/roms/rom.sfc | cut -d' ' -f1 > new_game/custom_integrations/GameName-Snes/rom.sha
```

### 3. Create Integration Files

**data.json** - RAM address mappings:
```json
{
  "info": {
    "lives": {"address": 1234, "type": "|u1"},
    "score": {"address": 1236, "type": "<u2"}
  }
}
```

**metadata.json** - Game metadata:
```json
{
  "default_state": "Level1",
  "default_player_state": ""
}
```

**scenario.json** - Reward/done conditions:
```json
{
  "done": {
    "variables": {"lives": {"op": "equal", "reference": 0}}
  },
  "reward": {
    "variables": {"score": {"reward": 1}}
  }
}
```

### 4. Create Save States

Use an emulator (BizHawk, RetroArch, etc.) to create `.state` files at desired starting points. Copy them to `custom_integrations/GameName-Snes/`.

### 5. Create run_bot.sh

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find venv python
VENV_CANDIDATES=(
    "$ROOT_DIR/.venv/bin/python"
    "$ROOT_DIR/super_metroid_rl/retro_env/bin/python"
    "$ROOT_DIR/harvest/.venv/bin/python"
)

VENV_PYTHON=""
for candidate in "${VENV_CANDIDATES[@]}"; do
    if [[ -x "$candidate" ]]; then
        VENV_PYTHON="$candidate"
        break
    fi
done

if [[ -z "$VENV_PYTHON" ]]; then
    echo "No virtual environment found. Run: cd $ROOT_DIR && ./setup.sh"
    exit 1
fi

if [[ "${HEADLESS:-}" == "1" ]]; then
    export SDL_VIDEODRIVER="dummy"
    export SDL_AUDIODRIVER="dummy"
    export SDL_SOFTWARE_RENDERER="1"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
"$VENV_PYTHON" "$SCRIPT_DIR/run_bot.py" "$@"
```

### 6. Create run_bot.py

Use the shared harness:

```python
#!/usr/bin/env python3
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retro_harness import (
    make_env,
    keyboard_action,
    controller_action,
    sanitize_action,
    init_controller,
)

DEFAULT_GAME = "GameName-Snes"
DEFAULT_STATE = "Level1"

def play_game(game: str, state: str):
    import pygame

    env = make_env(game=game, state=state, game_dir=SCRIPT_DIR)
    obs, info = env.reset()

    screen = pygame.display.set_mode((obs.shape[1] * 3, obs.shape[0] * 3))
    joystick = init_controller(pygame)
    clock = pygame.time.Clock()
    running = True

    while running:
        keys = pygame.key.get_pressed()
        action = [0] * 12
        keyboard_action(keys, action, pygame)
        controller_action(joystick, action)
        sanitize_action(action)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    pygame.quit()

if __name__ == "__main__":
    from retro_harness import add_custom_integrations
    add_custom_integrations(SCRIPT_DIR)
    play_game(DEFAULT_GAME, DEFAULT_STATE)
```

## Directory Structure

```
new_game/
├── run_bot.sh              # Entry point
├── run_bot.py              # Game logic
├── roms/                   # ROM files (git-ignored)
│   └── rom.sfc
└── custom_integrations/
    └── GameName-Snes/
        ├── data.json       # RAM addresses
        ├── metadata.json   # Game metadata
        ├── scenario.json   # Reward/done conditions
        ├── rom.sfc         # Symlink to roms/rom.sfc
        ├── rom.sha         # SHA1 hash of ROM
        └── Level1.state    # Save states
```

## Shared Harness (retro_harness)

The root-level `retro_harness/` package provides:

- **Controls**: `keyboard_action()`, `controller_action()`, `sanitize_action()`, `init_controller()`
- **Env**: `make_env()`, `add_custom_integrations()`, `get_available_states()`
- **Protocol**: `Task`, `Skill`, `Plan`, `TaskResult`, `WorldState` for composable bot behaviors

## Finding RAM Addresses

Resources for finding game RAM addresses:
- [Data Crystal](https://datacrystal.tcrf.net)
- [GameHacking.org](https://gamehacking.org)
- [TASVideos](https://tasvideos.org)
- Use BizHawk's RAM Watch/Search tools
