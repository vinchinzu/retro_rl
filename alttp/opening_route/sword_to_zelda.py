"""Compat shim — prefer ``alttp.opening_route.secret_entrance_clear``.

Historical name: post-sword path was titled “sword → Zelda”, but the
implemented continuous milestone is **secret-entrance clear** only
(stairs exit outdoors). Zelda rescue is a later planned segment
(``main_hall_to_zelda``).
"""

from __future__ import annotations

from alttp.opening_route.secret_entrance_clear import *  # noqa: F403
from alttp.opening_route.secret_entrance_clear import (  # noqa: F401
    STAIRS_ALIGN_TOLERANCE,
    STAIRS_ALIGN_X,
    STAIRS_ALIGN_Y,
    SOUTH_CHAMBER_Y_MAX,
    STAIRS_EXIT_MAX_FRAMES,
    SWORD_TO_SOUTH_CHAMBER_SCRIPT,
    SecretEntranceClearResult,
    SwordToZeldaResult,
    approach_south_chamber,
    ensure_sword_control,
    evaluate_acceptance,
    exit_secret_entrance_stairs,
    left_secret_entrance,
    run_from_state,
    run_from_sword,
)
