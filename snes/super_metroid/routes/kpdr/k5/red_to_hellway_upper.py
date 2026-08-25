"""Red Tower upper-shaft climb: human RLE, period WJ, Ice-ripper ladder."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, settle_hold
from super_metroid.routes.kpdr.k5.geometry import (
    RED_BOTTOM_Y,
    RED_FLOOR_Y,
    RED_TOP_DOOR_Y,
)
from super_metroid.routes.kpdr.k5.red_to_hellway_common import (
    _HUMAN_FLOOR_RLE,
    _in_hellway,
    _in_red,
)
from super_metroid.routes.kpdr.rooms import ROOM_RED_TOWER
from super_metroid.routes.rle import RleScript, play_script
from super_metroid.routes.runtime import ControllerSession

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
