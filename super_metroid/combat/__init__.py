"""Full-knowledge combat: hitboxes, strategy controllers, RL feature vectors.

Vision-only policies are deferred until gold. Boss work here uses RAM
positions, HP, spritemaps, and known hitbox dimensions (sm-json-data), then
optional structured-state RL on top of a rule-based strategy.
"""

from super_metroid.combat.actions import (
    COMBAT_ACTION_NAMES,
    N_COMBAT_ACTIONS,
    action_names,
    action_vector,
    nearest_action_id,
)
from super_metroid.combat.bomb_torizo import (
    BombTorizoEvidence,
    BombTorizoStrategy,
    fight_bomb_torizo_action,
    play_bomb_torizo_fight,
)
from super_metroid.combat.features import (
    BOMB_TORIZO_INACTIVE_SPRITEMAPS,
    FEATURE_DIM,
    AxisAlignedBox,
    BossCatalogEntry,
    CombatFeatures,
    bomb_torizo_catalog,
    feature_vector,
    features_from_state,
    kraid_catalog,
)
from super_metroid.combat.kraid import (
    KraidEvidence,
    KraidStrategy,
    KraidVariaEvidence,
    VariaEvidence,
    fight_kraid_action,
    kraid_defeated,
    play_kraid_fight,
    play_kraid_fight_to_varia,
    play_kraid_rear_exit,
    play_kraid_to_varia,
    play_varia_collect,
)
from super_metroid.combat.natural_entry import (
    DEFAULT_NATURAL_ACTIVE_STATE,
    NaturalCaptureResult,
    capture_natural_bomb_torizo_activation,
)

__all__ = [
    "BOMB_TORIZO_INACTIVE_SPRITEMAPS",
    "COMBAT_ACTION_NAMES",
    "FEATURE_DIM",
    "N_COMBAT_ACTIONS",
    "AxisAlignedBox",
    "BombTorizoEvidence",
    "BombTorizoStrategy",
    "BossCatalogEntry",
    "CombatFeatures",
    "DEFAULT_NATURAL_ACTIVE_STATE",
    "KraidEvidence",
    "KraidStrategy",
    "KraidVariaEvidence",
    "NaturalCaptureResult",
    "VariaEvidence",
    "action_names",
    "action_vector",
    "bomb_torizo_catalog",
    "capture_natural_bomb_torizo_activation",
    "feature_vector",
    "features_from_state",
    "fight_bomb_torizo_action",
    "fight_kraid_action",
    "kraid_catalog",
    "kraid_defeated",
    "nearest_action_id",
    "play_bomb_torizo_fight",
    "play_kraid_fight",
    "play_kraid_fight_to_varia",
    "play_kraid_rear_exit",
    "play_kraid_to_varia",
    "play_varia_collect",
]
