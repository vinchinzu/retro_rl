"""Red Tower → Hellway pure return (K5 hop 12).

Source: ``post_ice_bat_to_red_pure`` ~(206–216, 2443) after Bat→Red dual
**718f**. Climb reverse of ``play_red_tower_to_bat`` descent bands, then RIGHT
into top-right Hellway door ``0xA2F7``.

Hybrid pure (Hi-Jump held on K5 stack)::

  1. Accept Red bottom residual; clear Bat door lip (never RIGHT into 0xA3DD)
  2. Morph + double-bomb IBJ 18/30 centered x≈150 — dual tunnel peak ~y1820
     (do **not** climb_lower first — desyncs IBJ)
  3. Tunnel seat → midplat hop → midplat IBJ dual temporary floor ~y1606
  4. Human ascent RLE first 850f from floor → dual past temp floor ~(122,1459)
     p81 (mid-air peak — not solid; do not force-unmorph)
  5. Spin-left seat ~(37,1499) → alternating period WJ phases dual ~y420
  6. Ice-freeze ripper ladder (morph hop) → top door → RIGHT Hellway

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` f23078–29947 (~6869f Red).
Human mid "platforms" y2255/2159/2023 are **frozen rippers** (Ice held) —
not solid tiles. Temp floor is bombable from above (outbound); climb arrives
on/under lip via IBJ then uses human-matched open-loop + period WJ upper.
"""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    settle_hold,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k5.geometry import (
    RED_BOTTOM_Y,
    RED_FLOOR_Y,
    RED_LOWER_LIP_Y,
    RED_TO_HELLWAY_EXIT_HOLD,
    RED_TO_HELLWAY_EXIT_RUN,
    RED_TO_HELLWAY_EXIT_SETTLE,
    RED_TO_HELLWAY_EXIT_SHOOT,
    RED_TO_HELLWAY_EXIT_SPIN,
    RED_TOP_DOOR_X,
    RED_TOP_DOOR_Y,
    RED_TUNNEL_Y,
)
from super_metroid.routes.kpdr.rooms import ROOM_HELLWAY, ROOM_RED_TOWER
from super_metroid.routes.rle import RleScript, load_rle_json, play_script
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

# Upper shaft alternating period WJ (probe dual to ~y420). period/into/flip
# match spazer-style open-loop latch; short phases switch walls before stall.
_UPPER_WJ_PERIOD = 16
_UPPER_WJ_INTO = 6
_UPPER_WJ_FLIP = 8
# (side, frames, stop_y) — dual-stable D-chain from left seat y1499.
# Stop after RIGHT→~y420 peak (phase index 7). Phase LEFT stop=150 *falls*
# to dual end ~(171,687) and loses the peak — ice-ripper ladder takes over.
_UPPER_WJ_PHASES: tuple[tuple[str, int, int], ...] = (
    ("LEFT", 600, 1200),
    ("RIGHT", 800, 1050),
    ("LEFT", 800, 900),
    ("RIGHT", 800, 750),
    ("LEFT", 800, 600),
    ("RIGHT", 800, 450),
    ("LEFT", 800, 300),
    ("RIGHT", 800, 200),
)

# Ice-frozen upper rippers (0xD47F) at y≈520/416/320/232 — human top path.
# (enemy_y, land_y_lo, land_y_hi, min_dx) — min_dx keeps hop path clear of
# the ice underside (same-column freeze bonks from below).
_ICE_RIPPER_TIERS: tuple[tuple[int, int, int, int], ...] = (
    (520, 478, 515, 0),
    (416, 368, 410, 14),
    (320, 268, 320, 12),
    (232, 178, 230, 10),
)
_ENEMY_BASE = 0x0F78
_ENEMY_STRIDE = 0x40
_ENEMY_X_OFF = 0x02
_ENEMY_Y_OFF = 0x06
_ENEMY_HP_OFF = 0x14
_ENEMY_FR_OFF = 0x26  # freeze timer


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


def _bat_safe(session: ControllerSession, label: str) -> None:
    """Stay left of Bat door on the bottom floor / low spin."""
    st = session.state
    if not _in_red(st):
        return
    # Bat right door is lethal below y~2360 at x≳215.
    if int(st.samus_y) >= RED_BOTTOM_Y - 100 and int(st.samus_x) >= 210:
        hold(session, 1, "LEFT", reason=f"{label}_bat")


def _bat_abort(session: ControllerSession, label: str) -> bool:
    """True if we just corrected a Bat-door drift (caller should break RIGHT hold)."""
    st = session.state
    if not _in_red(st):
        return True
    if int(st.samus_y) >= RED_BOTTOM_Y - 90 and int(st.samus_x) >= 220:
        hold(session, 4, "LEFT", reason=f"{label}_bat_abort")
        return True
    return False


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


def _right_wall_wj_cycle(
    session: ControllerSession, label: str, *, from_floor: bool
) -> None:
    """One right-wall wall-jump cycle (human-validated lower-shaft primitive)."""
    if not _in_red(session.state):
        return
    _unmorph(session, label)
    _kb(session, label)

    if from_floor or int(session.state.samus_y) >= RED_BOTTOM_Y - 40:
        # Runway mid-left then jump into right wall — avoid Bat door x≥230 low.
        for _ in range(70):
            st = session.state
            if not _in_red(st):
                return
            _bat_safe(session, label)
            if int(st.pose) in _MORPH:
                hold(session, 1, "UP", reason=f"{label}_u")
                continue
            if int(st.samus_y) < RED_BOTTOM_Y - 40:
                break
            if 145 <= int(st.samus_x) <= 155 and int(st.velocity_y) == 0:
                break
            hold(
                session,
                1,
                "LEFT" if int(st.samus_x) > 155 else "RIGHT",
                "B",
                reason=f"{label}_runway",
            )
        settle_hold(session, 3, reason=f"{label}_runway_s")
        hold(session, 3, "RIGHT", "B", reason=f"{label}_run")
        for f in range(75):
            st = hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_jwall")
            if not _in_red(st):
                return
            if _bat_abort(session, label):
                return
            if int(st.samus_x) >= 218 and int(st.velocity_y) == 0 and f > 18:
                break
    else:
        for _ in range(35):
            st = session.state
            if not _in_red(st) or int(st.samus_y) >= RED_BOTTOM_Y - 20:
                return
            if int(st.pose) in _MORPH:
                hold(session, 1, "UP", reason=f"{label}_u")
                continue
            if _bat_abort(session, label):
                return
            if int(st.samus_x) >= 216:
                break
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_re")

    hold(session, 2, reason=f"{label}_contact")
    hold(session, 2, "LEFT", reason=f"{label}_face")
    for _ in range(18):
        st = hold(session, 1, "LEFT", "A", reason=f"{label}_wj")
        if not _in_red(st):
            return
    for f in range(36):
        st = hold(session, 1, "RIGHT", "A", reason=f"{label}_back")
        if not _in_red(st):
            return
        if int(st.samus_x) >= 217 and int(st.velocity_y) <= 1 and f > 8:
            hold(session, 1, "LEFT", reason=f"{label}_face2")
            for _ in range(15):
                hold(session, 1, "LEFT", "A", reason=f"{label}_wj2")
            for _ in range(22):
                hold(session, 1, "RIGHT", "A", reason=f"{label}_back2")
            break


def _right_wall_recatch(
    session: ControllerSession, label: str
) -> SuperMetroidState:
    """WJ off right wall, short outward LEFT, re-catch right (past pocket ceiling).

    Probe-stable from pocket ~(225,2091) to open shaft y≈1942.
    """
    if not _in_red(session.state):
        return session.state
    _unmorph(session, label)
    _kb(session, label)
    # Approach right wall if grounded mid-shaft.
    st = session.state
    if int(st.velocity_y) == 0 and int(st.samus_x) < 200 and int(st.samus_y) > 2000:
        hold(session, 8, "RIGHT", "B", reason=f"{label}_ap")
        hold(session, 16, "RIGHT", "B", "A", reason=f"{label}_jap")
    hold(session, _RWJ_INTO, "RIGHT", reason=f"{label}_into")
    for _ in range(_RWJ_WJ):
        st = hold(session, 1, "RIGHT", "A", reason=f"{label}_wj")
        if not _in_red(st):
            return st
    for _ in range(_RWJ_OUT):
        st = hold(session, 1, "LEFT", "A", reason=f"{label}_out")
        if not _in_red(st):
            return st
        if int(st.samus_x) <= 175:
            break
    for f in range(_RWJ_BACK):
        st = hold(session, 1, "RIGHT", "A", reason=f"{label}_back")
        if not _in_red(st):
            return st
        if int(st.samus_x) >= 218 and int(st.velocity_y) <= 1 and f > 6:
            break
    return session.state


def _ibj_double(
    session: ControllerSession,
    label: str,
    *,
    center_x: int = 150,
    stop_y: int = RED_FLOOR_Y - 40,
) -> SuperMetroidState:
    """One double-bomb IBJ cycle (morph X / wait / X / wait).

    Dual-stable lower climb: center_x≈150, waits 18/30 — peaks tunnel ~y1820
    from past-pocket morph fall; midplat center_x≈171 peaks temp floor ~y1600.
    """
    if not _in_red(session.state):
        return session.state
    _kb(session, label)
    if not is_morph(session.state.pose) and int(session.state.pose) not in _MORPH:
        ensure_morph(session)
    st = session.state
    x = int(st.samus_x)
    y = int(st.samus_y)
    # Low fall: bias LEFT hard so we never drift into Bat door.
    if y >= RED_BOTTOM_Y - 120 and x >= 200:
        hold(session, 4, "LEFT", reason=f"{label}_bat_bias")
    elif x > center_x + 18:
        hold(session, 2, "LEFT", reason=f"{label}_cL")
    elif x < center_x - 18:
        hold(session, 2, "RIGHT", reason=f"{label}_cR")
    hold(session, 2, "X", reason=f"{label}_b1")
    for _ in range(_IBJ_WAIT1):
        st = hold(session, 1, reason=f"{label}_w1")
        if not _in_red(st) or int(st.samus_y) <= stop_y:
            return st
        if int(st.samus_y) >= RED_BOTTOM_Y - 100 and int(st.samus_x) >= 200:
            hold(session, 1, "LEFT", reason=f"{label}_bat_w1")
    hold(session, 2, "X", reason=f"{label}_b2")
    for _ in range(_IBJ_WAIT2):
        st = hold(session, 1, reason=f"{label}_w2")
        if not _in_red(st) or int(st.samus_y) <= stop_y:
            return st
        if int(st.samus_y) >= RED_BOTTOM_Y - 100 and int(st.samus_x) >= 200:
            hold(session, 1, "LEFT", reason=f"{label}_bat_w2")
    return session.state


def _tunnel_to_midplat(session: ControllerSession, label: str) -> SuperMetroidState:
    """From tunnel peak ~y1820: seat y1883 x≈104 → hop to midplat ~y1720 x≈116.

    Human-matched UP+X then A / A+X / RIGHT+A+X into bomb-block mid shelf.
    Dual-stable from past-pocket IBJ peak pin.
    """
    if not _in_red(session.state):
        return session.state
    _unmorph(session, label)
    for _ in range(16):
        hold(session, 1, "UP", reason=f"{label}_stand")
    settle_hold(session, 5, reason=f"{label}_tun_s")
    # Drop / walk left onto solid tunnel floor y≈1883.
    for _ in range(35):
        st = session.state
        if not _in_red(st):
            return st
        if int(st.samus_y) >= RED_TUNNEL_Y:
            break
        hold(session, 1, "LEFT", reason=f"{label}_to_tun")
    for _ in range(40):
        st = session.state
        if not _in_red(st) or int(st.velocity_y) == 0:
            break
        hold(session, 1, reason=f"{label}_tun_land")
    for _ in range(50):
        st = session.state
        if not _in_red(st):
            return st
        if abs(int(st.samus_x) - 104) < 6 and int(st.velocity_y) == 0:
            break
        hold(
            session,
            1,
            "LEFT" if int(st.samus_x) > 104 else "RIGHT",
            reason=f"{label}_tun_x",
        )
    settle_hold(session, 5, reason=f"{label}_tun_seat")
    for _ in range(8):
        hold(session, 2, "UP", "X", reason=f"{label}_tun_shot")
    for i in range(50):
        st = session.state
        if not _in_red(st) or int(st.samus_y) <= RED_FLOOR_Y + 80:
            return st
        if i < 15:
            hold(session, 1, "A", reason=f"{label}_tun_j")
        elif i < 25:
            hold(session, 1, "A", "X", reason=f"{label}_tun_jx")
        else:
            hold(session, 1, "RIGHT", "A", "X", reason=f"{label}_tun_jrx")
    for _ in range(35):
        st = session.state
        if not _in_red(st) or int(st.velocity_y) == 0:
            break
        hold(session, 1, reason=f"{label}_mid_land")
    return session.state


def _pocket_spin_mid(session: ControllerSession, label: str) -> SuperMetroidState:
    """From right pocket ~(225,2091) or wall seat: B+LEFT+A spin into mid shaft.

    Human + probe: dual-stable peak/seat ~y1932–1942 at x≈170–185 (crouch under
    tunnel lip). Right-wall pure-A at y1942 gains 0 (hard ceiling) — must leave
    via spin, not recatch thrash. Still ~50px short of RED_TUNNEL_Y=1880.
    """
    if not _in_red(session.state):
        return session.state
    _unmorph(session, label)
    _kb(session, label)
    # Stand from crouch if needed (pocket often pose 1/9/137).
    for _ in range(14):
        st = session.state
        if int(st.pose) in _STAND:
            break
        hold(session, 1, "UP", reason=f"{label}_stand")
    settle_hold(session, 3, reason=f"{label}_spin_s")
    hold(session, 4, "LEFT", reason=f"{label}_face")
    for f in range(56):
        st = hold(session, 1, "B", "LEFT", "A", reason=f"{label}_spin")
        if not _in_red(st):
            return st
        y = int(st.samus_y)
        # Plateau on mid crouch seat / peak.
        if f > 32 and int(st.velocity_y) <= 1 and y <= 1955:
            break
        if y <= RED_TUNNEL_Y:
            break
    for _ in range(16):
        st = session.state
        if not _in_red(st) or int(st.velocity_y) == 0:
            break
        hold(session, 1, reason=f"{label}_spin_land")
    return session.state


def _on_pocket_seat(state: SuperMetroidState) -> bool:
    """Right-pocket dead-end ledge used as spin launch (not climb-done)."""
    return (
        _in_red(state)
        and int(state.samus_x) >= 210
        and RED_LOWER_LIP_Y - 40 <= int(state.samus_y) <= RED_LOWER_LIP_Y + 40
        and int(state.velocity_y) == 0
    )


def _on_right_wall_ceiling(state: SuperMetroidState) -> bool:
    """Hard right-wall ceiling seat ~y1942 (pure-A gain 0)."""
    return (
        _in_red(state)
        and int(state.samus_x) >= 210
        and 1925 <= int(state.samus_y) <= 1985
        and int(state.velocity_y) == 0
    )


def _climb_lower(session: ControllerSession, label: str) -> SuperMetroidState:
    """Bottom → past pocket ceiling into open shaft y≲1960.

    Do **not** accept right-pocket ~(225,2091) as done. Chain right-wall WJ
    to natural pocket land, then re-catch until y<=1960 (or tunnel).
    """
    target_y = 1960
    # Re-entry from mid fall: clear Bat door before any RIGHT runway.
    st0 = session.state
    if _in_red(st0) and int(st0.samus_y) >= RED_BOTTOM_Y - 80:
        for _ in range(60):
            st0 = session.state
            if not _in_red(st0):
                break
            if int(st0.samus_x) <= 165 and int(st0.velocity_y) == 0:
                break
            hold(session, 1, "LEFT", "B", reason=f"{label}_reentry_bat")
        settle_hold(session, 4, reason=f"{label}_reentry_s")

    for attempt in range(48):
        st = session.state
        if _in_hellway(st):
            return st
        if not _in_red(st):
            raise TimeoutError(
                f"{label}: left Red lower room=0x{int(st.room_id):04X} "
                f"xy=({st.samus_x},{st.samus_y})"
            )
        y = int(st.samus_y)
        x = int(st.samus_x)
        if y <= RED_TUNNEL_Y:
            return st
        if y <= target_y and x <= 230:
            return st

        _kb(session, f"{label}_a{attempt}")
        _bat_safe(session, f"{label}_a{attempt}")

        # Floor / deep lower: classic right-wall WJ toward pocket.
        if y >= RED_BOTTOM_Y - 100:
            _right_wall_wj_cycle(
                session,
                f"{label}_a{attempt}",
                from_floor=True,
            )
            continue

        # Between floor and pocket lip: keep right-wall WJ (not recatch yet).
        if y > RED_LOWER_LIP_Y + 30:
            _right_wall_wj_cycle(
                session,
                f"{label}_midwj{attempt}",
                from_floor=False,
            )
            continue

        # Pocket band y~2050–2120: wait for grounded right seat, then recatch.
        if RED_LOWER_LIP_Y - 50 <= y <= RED_LOWER_LIP_Y + 50:
            # Land / settle on pocket before recatch (mid-air recatch free-falls).
            if int(st.velocity_y) != 0 or int(st.pose) not in _STAND:
                for _ in range(40):
                    st2 = hold(session, 1, reason=f"{label}_pocket_land")
                    if not _in_red(st2):
                        return st2
                    if int(st2.velocity_y) == 0 and int(st2.pose) in _STAND:
                        break
                    if int(st2.samus_y) >= RED_BOTTOM_Y - 40:
                        break
                st = session.state
                y = int(st.samus_y)
                x = int(st.samus_x)
            # If we fell, floor WJ next loop.
            if y >= RED_BOTTOM_Y - 80:
                continue
            # Grounded on/near right pocket → recatch past ceiling.
            if x >= 200 and y <= RED_LOWER_LIP_Y + 40:
                _right_wall_recatch(session, f"{label}_rc{attempt}")
                continue
            # On pocket height but left of wall: approach right then recatch.
            if y <= RED_LOWER_LIP_Y + 40:
                hold(session, 10, "RIGHT", "B", reason=f"{label}_pocket_ap")
                _right_wall_recatch(session, f"{label}_rcap{attempt}")
                continue

        # Above pocket / open mid: recatch or re-approach right wall.
        if x >= 200 or y <= target_y + 40:
            _right_wall_recatch(session, f"{label}_hi{attempt}")
            continue

        hold(session, 12, "RIGHT", "B", "A", reason=f"{label}_mid_ap")
        _right_wall_recatch(session, f"{label}_rcm{attempt}")

    st = session.state
    if int(st.samus_y) > target_y + 80:
        raise TimeoutError(
            f"{label}: lower climb stalled y={st.samus_y} x={st.samus_x} p={st.pose}"
        )
    return st


def _climb_mid(session: ControllerSession, label: str) -> SuperMetroidState:
    """Bottom → tunnel peak → midplat → temporary floor (phased dual path).

    Probe path (rr-av5s night2)::

      1. Morph + IBJ 18/30 c150 from pure bottom — dual peak ~y1820
         (do **not** climb_lower first — desyncs IBJ)
      2. ``_tunnel_to_midplat``: tunnel seat → bomb-hop midplat ~y1720 x171
      3. Midplat IBJ 18/30 c171 dual peaks temp floor ~y1600 (hard lip)
      4. Bomb-floor reverse residual when y locks ≤RED_FLOOR_Y

    Human mid ledges y2255/2159/2023 are frozen rippers — not pure solid tiles.
    """
    best_y = int(session.state.samus_y)

    def _ensure_m() -> None:
        st = session.state
        if is_morph(st.pose) or int(st.pose) in _MORPH:
            return
        try:
            ensure_morph(session)
        except Exception:
            hold(session, 1, "DOWN", reason=f"{label}_morph_d")
            hold(session, 1, reason=f"{label}_morph_r")
            hold(session, 1, "DOWN", reason=f"{label}_morph_d2")

    # --- Phase 1: bottom IBJ → tunnel peak ---
    _ensure_m()
    for c in range(90):
        st = session.state
        if _in_hellway(st):
            return st
        if not _in_red(st):
            raise TimeoutError(
                f"{label}: left Red p1 room=0x{int(st.room_id):04X} "
                f"xy=({st.samus_x},{st.samus_y})"
            )
        y = int(st.samus_y)
        if y < best_y:
            best_y = y
        if y <= 1830 and int(st.velocity_y) == 0 and c > 20:
            break
        if y <= 1820:
            break
        _kb(session, f"{label}_p1_{c}")
        _ensure_m()
        _ibj_double(session, f"{label}_p1_{c}", center_x=150, stop_y=1820)
    st = session.state
    if int(st.samus_y) > 1900:
        raise TimeoutError(
            f"{label}: tunnel peak failed best_y={best_y} "
            f"y={st.samus_y} x={st.samus_x} p={st.pose}"
        )

    # --- Phase 2: tunnel → midplat ---
    if int(session.state.samus_y) > 1680:
        _tunnel_to_midplat(session, f"{label}_tun")
    st = session.state
    if not _in_red(st):
        if _in_hellway(st):
            return st
        raise TimeoutError(
            f"{label}: left Red tun room=0x{int(st.room_id):04X}"
        )
    if int(st.samus_y) > 1850:
        # Retry peak IBJ then tunnel once.
        _ensure_m()
        for c in range(40):
            _ibj_double(session, f"{label}_p1r_{c}", center_x=150, stop_y=1820)
            if int(session.state.samus_y) <= 1830:
                break
        _tunnel_to_midplat(session, f"{label}_tun2")
    st = session.state
    if int(st.samus_y) > 1850:
        raise TimeoutError(
            f"{label}: midplat failed best_y={best_y} "
            f"y={st.samus_y} x={st.samus_x} p={st.pose}"
        )

    # --- Phase 3: stand → x171 → morph IBJ 18/30 (dual peaks ~y1606) ---
    _unmorph(session, label)
    for _ in range(15):
        hold(session, 1, "UP", reason=f"{label}_mid_stand")
    settle_hold(session, 6, reason=f"{label}_mid_s")
    for _ in range(50):
        st = session.state
        if not _in_red(st) or int(st.samus_y) > 1820:
            break
        if int(st.samus_x) >= 168:
            break
        hold(session, 1, "RIGHT", reason=f"{label}_mid_r")
    settle_hold(session, 4, reason=f"{label}_mid_s2")
    _ensure_m()
    for c in range(45):
        st = session.state
        if _in_hellway(st):
            return st
        if not _in_red(st):
            raise TimeoutError(
                f"{label}: left Red p3 room=0x{int(st.room_id):04X} "
                f"xy=({st.samus_x},{st.samus_y})"
            )
        y = int(st.samus_y)
        if y < best_y:
            best_y = y
        if y <= RED_FLOOR_Y + 10:
            return st
        if y >= 1950:
            break
        _ensure_m()
        _ibj_double(
            session, f"{label}_p3_{c}", center_x=171, stop_y=RED_FLOOR_Y - 40
        )
        # At dual peak ~y1606, spray bombs against temp floor lip.
        st = session.state
        if _in_red(st) and int(st.samus_y) <= RED_FLOOR_Y + 30:
            for _ in range(6):
                hold(session, 2, "X", reason=f"{label}_lip")
                hold(session, 6, reason=f"{label}_lip_w")
            if int(session.state.samus_y) <= RED_FLOOR_Y + 10:
                return session.state

    st = session.state
    if int(st.samus_y) <= RED_FLOOR_Y + 20:
        return st
    # Progress residual: tunnel+midplat dual-stable; temp floor hard lip ~y1600.
    raise TimeoutError(
        f"{label}: floor peak failed best_y={best_y} "
        f"y={st.samus_y} x={st.samus_x} p={st.pose}"
    )


def _play_upper_rle(
    session: ControllerSession, runs: RleScript, label: str
) -> SuperMetroidState:
    """Play open-loop upper RLE; stop early on Hellway / leave Red."""
    if not runs:
        return session.state

    def _stop(st: SuperMetroidState) -> bool:
        return _in_hellway(st) or not _in_red(st)

    return play_script(
        session,
        runs,
        reason=label,
        room_id=ROOM_RED_TOWER,
        stop_when=_stop,
        on_lag="break",
    )


def _seat_left_after_handoff(
    session: ControllerSession, label: str
) -> SuperMetroidState:
    """From mid-air handoff ~(122,1459) p81: spin-left onto left ledge y1499.

    Dual pin is **not** solid — falls through unless immediately steered. Do
    **not** force UP-unmorph pose 81/82 (taller hitbox drops ~100px).
    """
    if not _in_red(session.state):
        return session.state
    for _ in range(90):
        st = session.state
        if not _in_red(st):
            return st
        if (
            int(st.velocity_y) == 0
            and int(st.samus_x) <= 50
            and 1480 <= int(st.samus_y) <= 1520
        ):
            break
        # True morph only.
        if int(st.pose) in (29, 30, 31, 32):
            hold(session, 1, "UP", reason=f"{label}_u")
            continue
        hold(session, 1, "LEFT", "B", "A", reason=f"{label}_seat_spin")
    settle_hold(session, 8, reason=f"{label}_seat_s")
    # Crouch 138 / turn → stand (not pose 81).
    for _ in range(20):
        st = session.state
        if not _in_red(st):
            return st
        if int(st.pose) in (1, 2):
            break
        if int(st.pose) in (29, 30, 31, 32, 137, 138, 9, 10):
            hold(session, 1, "UP", reason=f"{label}_stand")
        else:
            break
    settle_hold(session, 4, reason=f"{label}_seat_s2")
    return session.state


def _period_wj(
    session: ControllerSession,
    label: str,
    *,
    side: str,
    frames: int,
    stop_y: int | None = None,
    period: int = _UPPER_WJ_PERIOD,
    into: int = _UPPER_WJ_INTO,
    flip: int = _UPPER_WJ_FLIP,
) -> SuperMetroidState:
    """Open-loop period wall-jump on one wall (into / flip / spin)."""
    opp = "RIGHT" if side == "LEFT" else "LEFT"
    for i in range(frames):
        st = session.state
        if _in_hellway(st) or not _in_red(st):
            return st
        y = int(st.samus_y)
        if stop_y is not None and y <= stop_y:
            return st
        if y <= RED_TOP_DOOR_Y + 25:
            return st
        if y >= RED_BOTTOM_Y - 80:
            return st
        # True morph only — never force-unmorph 81/82 mid-climb.
        if int(st.pose) in (29, 30, 31, 32):
            hold(session, 1, "UP", reason=f"{label}_u")
            continue
        ph = i % period
        if ph < into:
            hold(session, 1, side, "A", reason=f"{label}_into")
        elif ph < into + flip:
            hold(session, 1, opp, "A", reason=f"{label}_flip")
        else:
            hold(session, 1, opp, "B", "A", reason=f"{label}_spin")
    return session.state


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def _session_env(session: ControllerSession):
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError("red_to_hellway ice ladder needs session.env")
    return env


def _list_upper_rippers(session: ControllerSession) -> list[dict]:
    """Live upper-shaft rippers (y<900) with freeze timer."""
    ram = _session_env(session).get_ram()
    out: list[dict] = []
    for i in range(12):
        base = _ENEMY_BASE + i * _ENEMY_STRIDE
        eid = _u16(ram, base)
        if eid == 0:
            continue
        x = _u16(ram, base + _ENEMY_X_OFF)
        y = _u16(ram, base + _ENEMY_Y_OFF)
        hp = _u16(ram, base + _ENEMY_HP_OFF)
        fr = _u16(ram, base + _ENEMY_FR_OFF)
        if x >= 0xFE00 or y >= 0xFE00 or y > 900:
            continue
        if x == 0 and y == 0:
            continue
        out.append({"i": i, "x": x, "y": y, "hp": hp, "fr": fr})
    return out


def _land_thin_seat(session: ControllerSession, label: str) -> SuperMetroidState:
    """From peak ~y420 / end ~y450: fall onto thin natural seat ~(85–95,587)."""
    if not _in_red(session.state):
        return session.state
    for _ in range(220):
        st = session.state
        if not _in_red(st):
            return st
        y = int(st.samus_y)
        x = int(st.samus_x)
        vy = int(st.velocity_y)
        if vy == 0 and 560 <= y <= 610 and 70 <= x <= 110:
            break
        if int(st.pose) in (29, 30, 31, 32):
            hold(session, 1, "UP", reason=f"{label}_u")
            continue
        if x > 95:
            hold(session, 1, "LEFT", reason=f"{label}_L")
        elif x < 75:
            hold(session, 1, "RIGHT", reason=f"{label}_R")
        else:
            hold(session, 1, reason=f"{label}_fall")
    settle_hold(session, 8, reason=f"{label}_s")
    for _ in range(30):
        st = session.state
        if int(st.pose) in (1, 2, 3, 4, 9, 10):
            break
        hold(session, 1, "UP", reason=f"{label}_stand")
    settle_hold(session, 4, reason=f"{label}_s2")
    return session.state


def _freeze_ripper_tier(
    session: ControllerSession,
    label: str,
    target_y: int,
    *,
    min_dx: int = 0,
    max_wait: int = 400,
) -> dict | None:
    """Ice-freeze the ripper whose y≈target_y; prefer |dx|≥min_dx from Samus."""
    if not _in_red(session.state):
        return None
    for _ in range(8):
        if int(session.state.pose) in (1, 3, 5, 7, 9):
            break
        hold(session, 1, "RIGHT", reason=f"{label}_face")
    settle_hold(session, 3, reason=f"{label}_fs")
    for _ in range(3):
        hold(session, 1, "UP", reason=f"{label}_aim")
    best: dict | None = None
    for wait in range(max_wait):
        ens = _list_upper_rippers(session)
        frozen = [
            e
            for e in ens
            if abs(e["y"] - target_y) <= 12 and e["fr"] > 40
        ]
        if frozen:
            dx = abs(frozen[0]["x"] - int(session.state.samus_x))
            if dx >= min_dx or wait > max_wait // 2:
                return frozen[0]
            best = frozen[0]
        cand = [
            e
            for e in ens
            if abs(e["y"] - target_y) <= 8
            and abs(e["x"] - int(session.state.samus_x)) <= 42
        ]
        if cand:
            e0 = cand[0]
            dx = abs(e0["x"] - int(session.state.samus_x))
            # Only shoot when offset is large enough for a clear vertical path
            # (or min_dx==0 / late fallback).
            if dx >= min_dx or (min_dx > 0 and wait > 120 and 6 <= dx <= 42):
                for _ in range(7):
                    hold(session, 1, "UP", "X", reason=f"{label}_shot")
                for _ in range(22):
                    hold(session, 1, "UP", reason=f"{label}_travel")
                hit = [
                    e
                    for e in _list_upper_rippers(session)
                    if abs(e["y"] - target_y) <= 14 and e["fr"] > 0
                ]
                if hit:
                    dxh = abs(hit[0]["x"] - int(session.state.samus_x))
                    if dxh >= min_dx or wait > max_wait // 2:
                        return hit[0]
                    best = hit[0]
            else:
                hold(session, 1, reason=f"{label}_wait_dx")
        else:
            hold(session, 1, reason=f"{label}_wait")
    if best is not None:
        return best
    frozen = [
        e
        for e in _list_upper_rippers(session)
        if abs(e["y"] - target_y) <= 12 and e["fr"] > 0
    ]
    return frozen[0] if frozen else None


def _morph_hop_ice(
    session: ControllerSession,
    label: str,
    enemy_y: int,
    land_lo: int,
    land_hi: int,
) -> bool:
    """High standing jump, drift onto frozen ripper top, stand (no ground-morph).

    Ground morph (pose 23) falls *through* frozen rippers; air spin / crouch
    land (pose 164/1) sticks. Peak ~enemy_y−55 then empty-fall onto top.
    """
    if not _in_red(session.state):
        return False
    frs = [
        e
        for e in _list_upper_rippers(session)
        if abs(e["y"] - enemy_y) <= 12 and e["fr"] > 25
    ]
    if not frs:
        return False
    ex = int(frs[0]["x"])
    for _ in range(12):
        if int(session.state.pose) in (1, 2, 3, 4, 9, 10):
            break
        hold(session, 1, "UP", reason=f"{label}_stand")
    settle_hold(session, 5, reason=f"{label}_hs")
    peak_tgt = enemy_y - 55
    hold(session, 1, "A", reason=f"{label}_j0")
    for f in range(40):
        y = int(session.state.samus_y)
        if y <= peak_tgt:
            break
        # Early bounce on underside of the ice — abort A thrash.
        if f > 10 and int(session.state.velocity_y) == 0 and y > enemy_y - 5:
            break
        hold(session, 1, "A", reason=f"{label}_j")
    for f in range(100):
        st = session.state
        if not _in_red(st):
            return False
        y = int(st.samus_y)
        x = int(st.samus_x)
        vy = int(st.velocity_y)
        # Never force ground-morph here — it falls through frozen enemies.
        if vy == 0 and land_lo <= y <= land_hi and f > 2:
            for _ in range(10):
                hold(session, 1, reason=f"{label}_land")
            for _ in range(30):
                if int(session.state.pose) in (1, 2, 3, 4, 9, 10):
                    break
                hold(session, 1, "UP", reason=f"{label}_stand2")
            ys: list[int] = []
            for _ in range(14):
                hold(session, 1, reason=f"{label}_stick")
                ys.append(int(session.state.samus_y))
            return all(land_lo - 12 <= yy <= land_hi + 18 for yy in ys)
        if y < enemy_y - 12:
            # High enough: drift to ice x then empty-fall.
            if abs(x - ex) > 5:
                btn = "RIGHT" if x < ex else "LEFT"
                if f < 12:
                    hold(session, 1, btn, "A", reason=f"{label}_drift")
                else:
                    hold(session, 1, btn, reason=f"{label}_drift2")
            else:
                hold(session, 1, reason=f"{label}_fall")
        else:
            if f < 12:
                hold(session, 1, "A", reason=f"{label}_up")
            elif x < ex - 3:
                hold(session, 1, "RIGHT", "A", reason=f"{label}_r")
            elif x > ex + 3:
                hold(session, 1, "LEFT", "A", reason=f"{label}_l")
            else:
                hold(session, 1, "A", reason=f"{label}_up2")
    return False


def _ice_ripper_ladder(
    session: ControllerSession, label: str
) -> SuperMetroidState:
    """From thin seat / peak ~y420: Ice-freeze ripper ladder → top door band.

    Human top path (tape f29304–29947) stands on frozen rippers at y495/391/
    295/207 then walks RIGHT into Hellway. Pure WJ stalls ~y390–420 without
    these platforms.
    """
    if not _in_red(session.state):
        return session.state
    y0 = int(session.state.samus_y)
    # Land thin seat if still above it (post peak / WJ).
    if y0 < 560 and y0 > 400:
        _land_thin_seat(session, f"{label}_seat")
    elif y0 < 400:
        # Already mid-ladder (retry entry); continue tiers below us.
        pass
    elif y0 > 620:
        return session.state

    for enemy_y, land_lo, land_hi, min_dx in _ICE_RIPPER_TIERS:
        if not _in_red(session.state):
            return session.state
        if int(session.state.samus_y) <= land_lo - 5:
            continue
        for _ in range(25):
            if int(session.state.velocity_y) == 0:
                break
            hold(session, 1, reason=f"{label}_settle")
        if int(session.state.samus_y) > 650:
            return session.state
        fr = _freeze_ripper_tier(
            session, f"{label}_fz{enemy_y}", enemy_y, min_dx=min_dx
        )
        if fr is None:
            fr = _freeze_ripper_tier(
                session,
                f"{label}_fz2{enemy_y}",
                enemy_y,
                min_dx=max(0, min_dx - 6),
                max_wait=250,
            )
        if fr is None:
            continue
        ok = _morph_hop_ice(
            session, f"{label}_hop{enemy_y}", enemy_y, land_lo, land_hi
        )
        if not ok:
            for attempt in range(2):
                for _ in range(25):
                    if int(session.state.velocity_y) == 0:
                        break
                    hold(session, 1, reason=f"{label}_rs")
                if int(session.state.samus_y) > 650:
                    break
                fr = _freeze_ripper_tier(
                    session,
                    f"{label}_fzr{attempt}",
                    enemy_y,
                    min_dx=max(0, min_dx - 4),
                    max_wait=280,
                )
                if fr is None:
                    continue
                ok = _morph_hop_ice(
                    session,
                    f"{label}_hopr{attempt}",
                    enemy_y,
                    land_lo,
                    land_hi,
                )
                if ok:
                    break
        if not ok:
            # Progress residual: keep whatever height we have.
            continue
        for _ in range(6):
            hold(session, 1, "UP", "X", reason=f"{label}_prep")
        settle_hold(session, 4, reason=f"{label}_tier_s")

    return session.state


def _climb_upper(session: ControllerSession, label: str) -> SuperMetroidState:
    """Temporary floor ~y1600 → top door band.

    Dual path (rr-av5s)::

      1. Human ascent RLE first 850f from live climb_mid floor ~(171,1606)
         → dual peak past temp floor ~(122,1459) p81 (mid-air, not solid)
      2. Spin-left seat onto left ledge ~(37,1499) — no force-unmorph p81
      3. Alternating period WJ phases (D-chain) dual-stable to ~y420
      4. Ice-freeze ripper ladder (morph hop) y495→391→295→207 → door

    Do **not** bomb-open the temp floor from below. Do **not** force-unmorph
    pose 81/82 at the dual handoff. Human RLE past 850 desyncs from pure pin.
    Do **not** continue period WJ past the y420 peak (phase 8 falls to y687).
    """
    st0 = session.state
    if not _in_red(st0):
        return st0

    # --- Phase A: dual human RLE past temp floor ---
    if int(st0.samus_y) >= RED_FLOOR_Y - 120:
        _play_upper_rle(session, _HUMAN_FLOOR_RLE, f"{label}_human850")
        if _in_hellway(session.state):
            return session.state
        if not _in_red(session.state):
            return session.state

    # --- Phase B: left ledge seat (handoff is mid-air peak) ---
    y_h = int(session.state.samus_y)
    if y_h <= 1550 and y_h >= 1300:
        _seat_left_after_handoff(session, f"{label}_seat")
        if _in_hellway(session.state) or not _in_red(session.state):
            return session.state

    # --- Phase C: alternating period WJ (dual ~y420 peak) ---
    if int(session.state.samus_y) > 500:
        # Launch into left wall from seat.
        hold(session, 3, "LEFT", "B", reason=f"{label}_wj_run")
        for _ in range(12):
            st = hold(session, 1, "LEFT", "B", "A", reason=f"{label}_wj_j")
            if _in_hellway(st) or not _in_red(st):
                return st
            if int(st.samus_y) <= RED_TOP_DOOR_Y + 40:
                return st
        for i, (side, frames, stop_y) in enumerate(_UPPER_WJ_PHASES):
            _period_wj(
                session,
                f"{label}_pwj{i}",
                side=side,
                frames=frames,
                stop_y=stop_y,
            )
            st = session.state
            if _in_hellway(st) or not _in_red(st):
                return st
            if int(st.samus_y) <= RED_TOP_DOOR_Y + 40:
                return st

    # --- Phase D: Ice-ripper ladder (y420 peak → top door) ---
    if (
        _in_red(session.state)
        and int(session.state.samus_y) > RED_TOP_DOOR_Y + 40
        and int(session.state.samus_y) < 900
    ):
        _ice_ripper_ladder(session, f"{label}_ice")
        if _in_hellway(session.state) or not _in_red(session.state):
            return session.state

    # --- Phase E: residual only when already near top door band ---
    # Adaptive thrash after dual mid pins loses height — only when y≤~280.
    if int(session.state.samus_y) > RED_TOP_DOOR_Y + 100:
        return session.state

    best_y = int(session.state.samus_y)
    last_best = 0
    side = "LEFT" if int(session.state.samus_x) > 128 else "RIGHT"
    for frame in range(1200):
        st = session.state
        if _in_hellway(st) or not _in_red(st):
            return st
        y = int(st.samus_y)
        x = int(st.samus_x)
        if y < best_y:
            best_y = y
            last_best = frame
        if y <= RED_TOP_DOOR_Y + 30:
            return st
        if y >= RED_BOTTOM_Y - 100 and x >= 210:
            hold(session, 8, "LEFT", reason=f"{label}_bat")
            continue
        if int(st.pose) in (29, 30, 31, 32):
            hold(session, 1, "UP", reason=f"{label}_u")
            continue
        if frame - last_best > 350:
            side = "RIGHT" if side == "LEFT" else "LEFT"
            last_best = frame
        opp = "RIGHT" if side == "LEFT" else "LEFT"
        ph = frame % _UPPER_WJ_PERIOD
        if ph < _UPPER_WJ_INTO:
            hold(session, 1, side, "A", reason=f"{label}_res_i")
        elif ph < _UPPER_WJ_INTO + _UPPER_WJ_FLIP:
            hold(session, 1, opp, "A", reason=f"{label}_res_f")
        else:
            hold(session, 1, opp, "B", "A", reason=f"{label}_res_s")

    return session.state


def play_red_to_hellway(session: ControllerSession) -> SuperMetroidState:
    """Red Tower bottom → ordinary Hellway left (K5 hop 12).

    Product body: ``warehouse_to_red_human`` hop 6, dual-green from
    ``post_ice_bat_to_red_pure`` ~(216,2443) as well as its live enter pin.
    Ice-ladder RAM rewrite stays in this module as residual research.
    """
    label = "red_to_hellway"
    require_room(session, ROOM_RED_TOWER, label)
    from super_metroid.routes.rle import load_rle_json, play_script

    body_path = _DATA / "red_to_hellway_human_hop.json"
    play_script(
        session,
        load_rle_json(body_path),
        reason=label,
        room_id=ROOM_RED_TOWER,
        stop_when=lambda s: int(s.room_id) != ROOM_RED_TOWER,
    )
    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )
    if int(session.state.room_id) != ROOM_RED_TOWER:
        st = session.state
        raise TimeoutError(
            f"{label}: hop body left Red to 0x{int(st.room_id):04X} "
            f"xy=({st.samus_x},{st.samus_y}) p={st.pose}"
        )
    unmorph(session)
    select_weapon(session, 0)
    hold(session, 6, reason=f"{label}_entry_glide")

    # Clear Bat door lip before any RIGHT spin.
    for _ in range(100):
        st = session.state
        if not _in_red(st):
            break
        if int(st.samus_x) <= 165 and int(st.velocity_y) == 0:
            break
        hold(session, 1, "LEFT", "B", reason=f"{label}_clear_bat")
    settle_hold(session, 6, reason=f"{label}_bottom_settle")

    # Mid IBJ 18/30 c150 is dual-stable from the pure bottom pin itself.
    # Running climb_lower first desyncs enemy/block state and kills the IBJ
    # climb (probe: bottom→peak y1820 dual; post-lower→IBJ stalls ~y1977).
    # Keep lower as recovery inside _climb_mid only.
    _climb_mid(session, f"{label}_mid")
    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )

    _climb_upper(session, f"{label}_upper")
    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )
    if not _in_red(session.state):
        raise TimeoutError(f"{label}: left Red unexpectedly: {session.state}")

    # True morph only before exit — never force-unmorph pose 81/82 residual.
    if int(session.state.pose) in (29, 30, 31, 32):
        _unmorph(session, label)
    # Only reclimb when already in the upper door band (avoid full RLE thrash).
    if RED_TOP_DOOR_Y < int(session.state.samus_y) <= RED_TOP_DOOR_Y + 100:
        _climb_upper(session, f"{label}_reclimb")
        if _in_hellway(session.state):
            return wait_ordinary_room(
                session,
                ROOM_HELLWAY,
                settle_frames=RED_TO_HELLWAY_EXIT_SETTLE,
                label=label,
            )

    # Exit only from the top door band — mid-shaft RIGHT thrash desyncs residual.
    if int(session.state.samus_y) > RED_TOP_DOOR_Y + 120:
        st = session.state
        raise TimeoutError(
            f"{label}: upper residual room=0x{int(st.room_id):04X} "
            f"xy=({st.samus_x},{st.samus_y}) p={st.pose} "
            f"(need y≤{RED_TOP_DOOR_Y + 120} for Hellway exit)"
        )

    for _ in range(100):
        st = session.state
        if _in_hellway(st) or not _in_red(st):
            break
        if int(st.samus_y) > RED_TOP_DOOR_Y + 50:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_top_hop")
            continue
        if int(st.samus_x) < RED_TOP_DOOR_X - 30:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_top_run")
            continue
        break

    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )

    return play_run_shoot_exit(
        session,
        from_room=ROOM_RED_TOWER,
        to_room=ROOM_HELLWAY,
        direction="RIGHT",
        label=label,
        run_frames=RED_TO_HELLWAY_EXIT_RUN,
        shoot_frames=RED_TO_HELLWAY_EXIT_SHOOT,
        spin_frames=RED_TO_HELLWAY_EXIT_SPIN,
        hold_frames=RED_TO_HELLWAY_EXIT_HOLD,
        settle_frames=RED_TO_HELLWAY_EXIT_SETTLE,
    )


__all__ = ["play_red_to_hellway"]
