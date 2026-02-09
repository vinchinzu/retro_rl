"""
Shared fighting game utilities for Street Fighter and Mortal Kombat series.

Provides:
- FightingEnv: Gymnasium wrapper with health-delta rewards, round/KO detection
- FighterCNN: CNN feature extractor for PPO training
- Discrete action maps tuned for fighting games
- Menu navigation utilities
- Save state creation helpers
"""

from fighters_common.fighting_env import (
    FightingGameConfig,
    FightingEnv,
    DirectRAMReader,
    FrameSkip,
    FrameStack,
    GrayscaleResize,
    DiscreteAction,
    FIGHTING_ACTIONS,
    make_fighting_env,
)

from fighters_common.menu_nav import (
    MenuNavigator,
    navigate_to_fight,
    create_fight_state,
)

from fighters_common.game_configs import (
    GAME_REGISTRY,
    get_game_config,
)

__all__ = [
    "FightingGameConfig",
    "FightingEnv",
    "FrameSkip",
    "FrameStack",
    "GrayscaleResize",
    "DiscreteAction",
    "FIGHTING_ACTIONS",
    "make_fighting_env",
    "MenuNavigator",
    "navigate_to_fight",
    "create_fight_state",
    "GAME_REGISTRY",
    "get_game_config",
]
