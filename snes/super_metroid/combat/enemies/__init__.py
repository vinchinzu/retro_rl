"""Room-enemy Overlay: one scan and one per-frame choice.

Boss fights stay ``combat.protocol.BossStrategy``. This package is the
per-frame overlay a movement hop asks before it walks. Interface is
``list_enemies`` and ``choose``; Species and action helpers are internal.

See ``docs/adr/0008-room-enemy-overlay.md``.
"""

from super_metroid.combat.enemies.scan import (
    Enemy,
    list_enemies,
)
from super_metroid.combat.enemies.species import (
    ATOMIC_ID,
    COVERN_ID,
    WORKROBOT_ID,
    Stance,
)
from super_metroid.combat.enemies.stance import (
    Choice,
    Intent,
    choose,
)

__all__ = [
    "ATOMIC_ID",
    "COVERN_ID",
    "WORKROBOT_ID",
    "Choice",
    "Enemy",
    "Intent",
    "Stance",
    "choose",
    "list_enemies",
]
