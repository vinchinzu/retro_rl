"""Shim: offline hop trim lives in ``super_metroid.human_tape.trim``.

Existing ``from super_metroid.human_tape_trim import …`` keeps working.
"""

from __future__ import annotations

from super_metroid.human_tape.trim import (  # noqa: F401
    BTN_A,
    BTN_B,
    BTN_DOWN,
    BTN_L,
    BTN_LEFT,
    BTN_R,
    BTN_RIGHT,
    BTN_SELECT,
    BTN_START,
    BTN_UP,
    BTN_X,
    BTN_Y,
    COMBAT_ROOM_IDS,
    TrimReport,
    export_trimmed_seed,
    find_mid_idle_cuts,
    find_retry_loop_cuts,
    holds_charge_safe_buttons,
    infer_mode,
    is_combat_room,
    is_idle_frame,
    progress_along_leave,
    trim_hop,
    trim_task_hop,
)

__all__ = [
    "BTN_A",
    "BTN_B",
    "BTN_X",
    "BTN_Y",
    "COMBAT_ROOM_IDS",
    "TrimReport",
    "export_trimmed_seed",
    "find_mid_idle_cuts",
    "find_retry_loop_cuts",
    "holds_charge_safe_buttons",
    "infer_mode",
    "is_combat_room",
    "is_idle_frame",
    "progress_along_leave",
    "trim_hop",
    "trim_task_hop",
]
