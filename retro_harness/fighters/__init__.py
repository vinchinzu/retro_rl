"""
Shared fighting game utilities for Street Fighter and Mortal Kombat series.

Provides:
- FightingEnv: Gymnasium wrapper with health-delta rewards, round/KO detection
- FighterCNN: CNN feature extractor for PPO training
- Discrete action maps tuned for fighting games
- Menu navigation utilities
- Save state creation helpers
"""

from retro_harness.fighters.fighting_env import (
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

from retro_harness.fighters.ram_observation import (
    RamObservation,
    make_ram_fighting_env,
    build_eval_env,
    MK1_RAM_FEATURES,
)

from retro_harness.fighters.menu_nav import (
    MenuNavigator,
    navigate_to_fight,
    create_fight_state,
)

from retro_harness.fighters.game_configs import (
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
    "RamObservation",
    "make_ram_fighting_env",
    "build_eval_env",
    "MK1_RAM_FEATURES",
    "MenuNavigator",
    "navigate_to_fight",
    "create_fight_state",
    "GAME_REGISTRY",
    "get_game_config",
]
