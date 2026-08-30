"""Shared Red Tower → Hellway climb primitives."""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, is_morph, unmorph
from super_metroid.routes.kpdr.rooms import ROOM_HELLWAY, ROOM_RED_TOWER
from super_metroid.routes.rle import RleScript, load_rle_json
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_knockback_spin, is_knockback

_MORPH = frozenset({0x1D, 0x1E, 0x1F, 0x20, 29, 30, 39, 40, 41, 42, 81, 82})
_STAND = frozenset({1, 2, 9, 10, 12, 27, 28, 137, 138})

# Double-bomb IBJ cadence (probe-validated mid-shaft; peaks through tunnel).
_IBJ_WAIT1 = 18
_IBJ_WAIT2 = 30
# Right-wall re-catch past pocket ceiling (stable ~y1942 from pocket pin).
_RWJ_INTO = 2
_RWJ_WJ = 10
_RWJ_OUT = 18
_RWJ_BACK = 36

_DATA = Path(__file__).resolve().parents[1] / "data"
# Human ascent open-loop — first 850f dual-stable from live climb_mid floor
# pin ~(171,1606): peaks past temp floor to ~(122,1459) p81.
# Remainder of the tape desyncs from pure Bat→Red enemy/block state.
_HUMAN_ASCENT_FULL: RleScript = load_rle_json(
    _DATA / "red_to_hellway_human_ascent.json"
)
_HUMAN_FLOOR_FRAMES = 850


def _slice_rle(runs: RleScript, n_frames: int) -> RleScript:
    """Take the first ``n_frames`` of an RLE script."""
    out: list[tuple[int, tuple[str, ...]]] = []
    used = 0
    for n, buttons in runs:
        if used >= n_frames:
            break
        take = min(int(n), n_frames - used)
        if take > 0:
            out.append((take, tuple(buttons)))
            used += take
    return tuple(out)


_HUMAN_FLOOR_RLE: RleScript = _slice_rle(_HUMAN_ASCENT_FULL, _HUMAN_FLOOR_FRAMES)


def _in_red(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_RED_TOWER


def _in_hellway(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_HELLWAY


def _unmorph(session: ControllerSession, label: str) -> None:
    for _ in range(12):
        st = session.state
        if not (is_morph(st.pose) or int(st.pose) in _MORPH):
            return
        hold(session, 1, "UP", reason=f"{label}_unmorph")
    unmorph(session)


def _kb(session: ControllerSession, label: str, prefer: str = "LEFT") -> None:
    if is_knockback(session.state):
        escape_knockback_spin(
            session,
            prefer_dir=prefer,
            run_frames=2,
            spin_frames=8,
            label=f"{label}_kb",
            ensure_beam=True,
            break_on_motion_clear=True,
        )
