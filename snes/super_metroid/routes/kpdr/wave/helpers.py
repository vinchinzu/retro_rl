"""Shared Wave-branch helpers: knockback escape with room-specific stop ids.

Consolidation of escape_kb variants is bead rr-7sn.5 — keep local wrappers
behavior-identical for now.
"""

from __future__ import annotations

from super_metroid.routes.kpdr.rooms import (
    ROOM_DOUBLE_CHAMBER,
    ROOM_SINGLE_CHAMBER,
    ROOM_WAVE,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_knockback_spin


def escape_kb_bsc(session: ControllerSession, label: str, prefer: str) -> None:
    """Bubble → Single: stop on Single Chamber entry."""
    escape_knockback_spin(
        session,
        prefer_dir=prefer,
        run_frames=6,
        spin_frames=18,
        label=label,
        stop_room_id=ROOM_SINGLE_CHAMBER,
    )


def escape_kb_sc(session: ControllerSession, label: str, prefer: str) -> None:
    """Single → Double: stop on Double Chamber entry."""
    escape_knockback_spin(
        session,
        prefer_dir=prefer,
        run_frames=6,
        spin_frames=18,
        label=label,
        stop_room_id=ROOM_DOUBLE_CHAMBER,
    )


def escape_kb_dc(session: ControllerSession, label: str, prefer: str) -> None:
    """Double → Wave: stop on Wave Room entry."""
    escape_knockback_spin(
        session,
        prefer_dir=prefer,
        run_frames=6,
        spin_frames=18,
        label=label,
        stop_room_id=ROOM_WAVE,
    )


__all__ = [
    "escape_kb_bsc",
    "escape_kb_sc",
    "escape_kb_dc",
]
