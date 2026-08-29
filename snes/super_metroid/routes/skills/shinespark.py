"""Reusable shinespark charge / crouch-store / activate helpers.

Room-agnostic dual-track skill surface for Moat, Landing Site practice, and
later elevated sparks. Controllers pass a :class:`ControllerSession` (or any
object with ``state`` + ``step``); probes may wrap a minimal session.

Harness buttons (never swap for VOD A/B labels)
----------------------------------------------
| Role                 | Harness | VOD / SM default |
|----------------------|---------|------------------|
| Dash / speed charge  | **B**   | A                |
| Jump / shine activate| **A**   | B                |
| Store                | DOWN    | DOWN             |

Verified facts (Moat residual + Landing Site practice)
------------------------------------------------------
* Full charge: ``speed_echoes ≥ ECHOES_FULL`` (hi byte of ``$0B3E``) while
  grounded; typical pure ``RIGHT+B`` on flat runway ≈90f to pose 9.
* Crouch-store from pose 9: arms ``$0A68`` ≈ ``TYPICAL_ARM_TIMER`` (179) on
  first DOWN → pose 53; settles near pose 39 with timer draining.
* Spin jump after store does **not** consume store (wiki-aligned); mid-air
  unspin then A activates.
* Horizontal spark poses ~199–202; diagonal uses aim (UP / RIGHT+UP / …) + A
  while armed.
* Store-from-pose: pose 9 ok; spin poses 25/166 + DOWN wipe echoes / fail arm.

Short charge (boost-counter vs velocity dual-track)
---------------------------------------------------
Speed Booster tracks (1) horizontal velocity from continuous dash+forward and
(2) a **boost counter** that only increments on run-animation "magic frames"
while dash+forward are held. Echoes appear at boost-counter 4; a stored
shinespark then travels at **full** spark speed even if velocity was near
walking.

* NTSC magic dash frames (forward held from 0): 25, 50, 70, 85.
  On 85, dash+DOWN can store in the same frame.
* PAL magic dash frames: 20, 40, 60, 70 (store on 70).
* Stutter-walk before the first magic frame shortens runway further
  (NTSC min ≈163.2 px; PAL ≈157.7 px) — see :func:`short_charge_plan`.

Addresses: :data:`super_metroid.ram.ADDR_SHINESPARK_TIMER` (``$0A68``),
:data:`super_metroid.ram.ADDR_SPEED_COUNTER` (``$0B3E``),
:data:`super_metroid.ram.ADDR_SPEED_FLAG` (``$0B3C``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from super_metroid.ram import (
    ADDR_SHINESPARK_TIMER,
    ADDR_SPEED_COUNTER,
    ADDR_SPEED_FLAG,
    parse_env_state,
)
from super_metroid.routes.runtime import hold
from super_metroid.routes.skills.shinespark_plans import (
    Direction,
    NTSC_MAGIC_DASH_FRAMES,
    NTSC_SHORT_CHARGE_FRAMES,
    NTSC_STUTTER_FULL_STOP_PX,
    NTSC_STUTTER_MIN_PX,
    PAL_MAGIC_DASH_FRAMES,
    PAL_SHORT_CHARGE_FRAMES,
    PAL_STUTTER_MIN_PX,
    Region,
    magic_dash_frames,
    short_charge_plan,
    stutter_dash_mask,
    stutter_forward_mask,
)

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession

ChargeMode = Literal["full", "short", "stutter"]

# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------

# Hi-byte of $0B3E at full speed-booster charge (echoes / blue suit gate).
ECHOES_FULL: int = 4

# First-frame $0A68 after crouch-store from grounded pose 9 (measured Moat /
# Landing Site). Drains ~1/frame while armed; exact arm can be 178–180.
TYPICAL_ARM_TIMER: int = 179

# Approx pure-run frames to full charge on a flat continuous runway.
TYPICAL_CHARGE_FRAMES: int = 90

# Crouch-store usually arms within a few frames of first DOWN from pose 9.
DEFAULT_STORE_MAX_FRAMES: int = 24

# Shinespark travel poses (horizontal ~199–202; diagonal/vertical ~203–206).
SPARK_POSES: frozenset[int] = frozenset({199, 200, 201, 202, 203, 204, 205, 206})

# Frames of UP (or neutral) after crouch-store before A will actually spark.
# Immediate RIGHT+A from pose 53 only walks; UP≈4 or neutral≈5 then A works.
DEFAULT_PRE_STAND_FRAMES: int = 4

# Grounded running pose that reliably crouch-stores (facing right/left run).
STORE_OK_POSES: frozenset[int] = frozenset({1, 2, 5, 6, 9, 10})

# Spin / fall poses where DOWN does not arm (echoes wipe or wrong transition).
STORE_WIPE_POSES: frozenset[int] = frozenset({25, 26, 27, 28, 166, 167})

# Knockback — do not treat as charge/store success.
KNOCKBACK_POSES: frozenset[int] = frozenset({137, 138})

# Short-charge constants / plan builders: see shinespark_plans (re-exported).

# ---------------------------------------------------------------------------
# WRAM read helpers
# ---------------------------------------------------------------------------


def _u16(ram: Any, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def read_spark_wram(env: Any) -> dict[str, int]:
    """Raw shine/speed words from WRAM (not just the parsed hi-byte).

    Keys
    ----
    spark_timer:
        ``$0A68`` — store / spark charge timer (>0 while armed or sparking).
    speed_flag:
        ``$0B3C`` — speed-boost / blue-suit related flag.
    speed_counter_word:
        full ``$0B3E`` word (lo = anim tick, hi = echoes).
    speed_echoes:
        hi byte of ``$0B3E`` (0–4+).
    speed_anim:
        lo byte of ``$0B3E``.
    """
    ram = env.get_ram()
    sc_word = _u16(ram, ADDR_SPEED_COUNTER)
    return {
        "spark_timer": _u16(ram, ADDR_SHINESPARK_TIMER),
        "speed_flag": _u16(ram, ADDR_SPEED_FLAG),
        "speed_counter_word": sc_word,
        "speed_echoes": (sc_word >> 8) & 0xFF,
        "speed_anim": sc_word & 0xFF,
    }


def spark_snapshot(env: Any, frame: int = 0) -> dict[str, Any]:
    """Nav parse + spark WRAM in one dict (probe-friendly)."""
    st = parse_env_state(env, frame=frame, mode="nav")
    w = read_spark_wram(env)
    return {
        "frame": frame,
        "room": st.room_id,
        "room_hex": f"0x{st.room_id:04X}",
        "x": st.samus_x,
        "y": st.samus_y,
        "pose": st.pose,
        "facing": st.facing,
        "vx": st.velocity_x,
        "vy": st.velocity_y,
        "gs": st.game_state,
        "door_trans": st.door_transition,
        "health": st.health,
        **w,
        "speed_boosting": st.speed_boosting,
        "shinesparking": st.shinesparking,
    }


def _env_of(session: Any) -> Any:
    env = getattr(session, "env", None)
    if env is None:
        raise AttributeError(
            "session has no .env; read_spark_wram/spark_snapshot need env access"
        )
    return env


def session_spark_wram(session: Any) -> dict[str, int]:
    """``read_spark_wram`` via ``session.env``."""
    return read_spark_wram(_env_of(session))


def is_spark_pose(pose: int) -> bool:
    return int(pose) in SPARK_POSES


def store_pose_ok(pose: int) -> bool:
    """True when crouch-store is expected to arm (not spin-wipe)."""
    p = int(pose)
    if p in STORE_WIPE_POSES or p in KNOCKBACK_POSES:
        return False
    return p in STORE_OK_POSES or p in (39, 40, 53, 54)  # crouch family


# ---------------------------------------------------------------------------
# Short-charge execution (plans from shinespark_plans)
# ---------------------------------------------------------------------------


def charge_by_plan(
    session: ControllerSession,
    plan: Sequence[Sequence[str]],
    *,
    require_grounded: bool = True,
    label: str = "short_charge",
    stop_on_boost: bool = True,
) -> dict[str, Any]:
    """Execute a frame-perfect button plan; optionally stop at echoes≥4.

    Returns the same shape as :func:`charge_until_boost` plus ``plan_len``,
    ``mode`` tags, ``start_x`` / ``end_x`` / ``delta_x``, and ``buttons_log``
    (compact: only frames that pressed dash or DOWN).
    """
    first_echo: dict[str, Any] | None = None
    boost_row: dict[str, Any] | None = None
    start_frame = int(session.frame)
    start_x = int(session.state.samus_x)
    buttons_log: list[dict[str, Any]] = []
    dash_frames: list[int] = []

    for i, btns in enumerate(plan):
        st = session.state
        try:
            echoes = session_spark_wram(session)["speed_echoes"]
        except AttributeError:
            echoes = int(getattr(st, "speed_counter", 0))

        if first_echo is None and echoes >= 1:
            first_echo = _snap(session)

        grounded = int(st.velocity_y) == 0 if require_grounded else True
        boosting = bool(getattr(st, "speed_boosting", echoes >= ECHOES_FULL))
        if stop_on_boost and boosting and grounded and int(st.pose) not in KNOCKBACK_POSES:
            boost_row = _snap(session)
            end_x = int(st.samus_x)
            return {
                "ok": True,
                "frames": i,
                "elapsed": int(session.frame) - start_frame,
                "direction": "RIGHT" if "RIGHT" in (plan[0] if plan else ()) else "LEFT",
                "first_echo": first_echo,
                "boost": boost_row,
                "plan_len": len(plan),
                "start_x": start_x,
                "end_x": end_x,
                "delta_x": end_x - start_x,
                "dash_frames": dash_frames,
                "buttons_log": buttons_log,
                "early_stop": True,
            }

        btn_t = tuple(btns)
        hold(session, 1, *btn_t, reason=f"{label}_{i}")
        # Log interesting presses (dash / store)
        interesting = [b for b in btn_t if b in ("B", "Y", "DOWN") or b == "A"]
        # Prefer tracking harness dash B
        if any(b in btn_t for b in ("B", "Y")):
            dash_frames.append(i)
        if interesting or i in (0, len(plan) - 1):
            buttons_log.append({"f": i, "buttons": list(btn_t)})

    # Final observation after last step
    st = session.state
    try:
        echoes = session_spark_wram(session)["speed_echoes"]
    except AttributeError:
        echoes = int(getattr(st, "speed_counter", 0))
    grounded = int(st.velocity_y) == 0 if require_grounded else True
    boosting = bool(getattr(st, "speed_boosting", echoes >= ECHOES_FULL))
    end_x = int(st.samus_x)
    if first_echo is None and echoes >= 1:
        first_echo = _snap(session)
    ok = boosting and grounded and int(st.pose) not in KNOCKBACK_POSES
    out: dict[str, Any] = {
        "ok": ok,
        "frames": len(plan),
        "elapsed": int(session.frame) - start_frame,
        "direction": next(
            (b for b in (plan[0] if plan else ()) if b in ("LEFT", "RIGHT")),
            "RIGHT",
        ),
        "first_echo": first_echo,
        "boost": _snap(session) if ok else _snap(session),
        "plan_len": len(plan),
        "start_x": start_x,
        "end_x": end_x,
        "delta_x": end_x - start_x,
        "dash_frames": dash_frames,
        "buttons_log": buttons_log,
        "early_stop": False,
    }
    if not ok:
        out["error"] = (
            f"short charge plan finished without echoes≥4 "
            f"(echoes={echoes} grounded={grounded} pose={int(st.pose)})"
        )
    return out


def short_charge_until_boost(
    session: ControllerSession,
    direction: Direction = "RIGHT",
    *,
    region: Region = "NTSC",
    stutter: bool = False,
    store_on_last: bool = False,
    dash_button: str = "B",
    require_grounded: bool = True,
    stop_on_boost: bool = True,
    label: str = "short_charge",
) -> dict[str, Any]:
    """Magic-frame short charge (optional stutter prefix).

    Holds forward continuously (or stutter pattern) and presses dash only on
    boost-counter check frames. Full echoes unlock a full-speed shinespark
    store even at near-walking velocity — key for short Moat / West Ocean
    runways.

    See module docstring and :func:`short_charge_plan`.
    """
    plan = short_charge_plan(
        region,
        stutter=stutter,
        store_on_last=store_on_last,
        direction=direction,
        dash_button=dash_button,
    )
    # When storing on the final magic frame, run the full plan (do not early-stop
    # after the 3rd boost tick or DOWN never fires).
    effective_stop = False if store_on_last else stop_on_boost
    report = charge_by_plan(
        session,
        plan,
        require_grounded=require_grounded,
        label=label,
        stop_on_boost=effective_stop,
    )
    report["region"] = region
    report["stutter"] = stutter
    report["store_on_last"] = store_on_last
    report["mode"] = "stutter" if stutter else "short"
    report["magic_frames"] = list(magic_dash_frames(region))
    if store_on_last:
        try:
            w = session_spark_wram(session)
            timer = w["spark_timer"]
            echoes = w["speed_echoes"]
        except AttributeError:
            timer = int(getattr(session.state, "shinespark_timer", 0))
            echoes = int(getattr(session.state, "speed_counter", 0))
        report["store_armed"] = timer > 0
        report["spark_timer"] = timer
        # Store often replaces the blue-suit flag; treat armed timer as success.
        if timer > 0:
            report["ok"] = True
            report.pop("error", None)
        elif echoes >= ECHOES_FULL:
            report["ok"] = True
            report.pop("error", None)
    return report


def charge_until_boost(
    session: ControllerSession,
    direction: Direction = "RIGHT",
    *,
    budget: int = 500,
    dash_button: str = "B",
    require_grounded: bool = True,
    extra_buttons: Sequence[str] = (),
    label: str = "charge",
    mode: ChargeMode = "full",
    region: Region = "NTSC",
    store_on_last: bool = False,
) -> dict[str, Any]:
    """Charge until grounded full echoes (``speed_boosting``).

    ``mode``:
      * ``full`` — continuous ``direction``+dash (classic ~90f runway).
      * ``short`` — magic-frame dash only (NTSC 25/50/70/85).
      * ``stutter`` — stutter-walk prefix + short charge (min ~163 px NTSC).

    Returns a report with ``ok``, ``frames``, ``first_echo``, ``boost``
    snapshots (via ``session.env`` when present; otherwise thin state dicts).
    """
    if mode in ("short", "stutter"):
        return short_charge_until_boost(
            session,
            direction,
            region=region,
            stutter=(mode == "stutter"),
            store_on_last=store_on_last,
            dash_button=dash_button,
            require_grounded=require_grounded,
            label=label,
        )

    dir_btn = "LEFT" if direction == "LEFT" else "RIGHT"
    first_echo: dict[str, Any] | None = None
    boost_row: dict[str, Any] | None = None
    start_frame = int(session.frame)
    start_x = int(session.state.samus_x)

    for i in range(budget):
        st = session.state
        w: dict[str, int] | None = None
        try:
            w = session_spark_wram(session)
            echoes = w["speed_echoes"]
        except AttributeError:
            echoes = int(getattr(st, "speed_counter", 0))

        if first_echo is None and echoes >= 1:
            first_echo = _snap(session)

        grounded = int(st.velocity_y) == 0 if require_grounded else True
        boosting = bool(getattr(st, "speed_boosting", echoes >= ECHOES_FULL))
        if boosting and grounded and int(st.pose) not in KNOCKBACK_POSES:
            boost_row = _snap(session)
            end_x = int(st.samus_x)
            return {
                "ok": True,
                "frames": i,
                "elapsed": int(session.frame) - start_frame,
                "direction": dir_btn,
                "first_echo": first_echo,
                "boost": boost_row,
                "mode": "full",
                "start_x": start_x,
                "end_x": end_x,
                "delta_x": end_x - start_x,
            }

        hold(
            session,
            1,
            dir_btn,
            dash_button,
            *extra_buttons,
            reason=f"{label}_run",
        )

    end_x = int(session.state.samus_x)
    return {
        "ok": False,
        "frames": budget,
        "elapsed": int(session.frame) - start_frame,
        "direction": dir_btn,
        "first_echo": first_echo,
        "boost": _snap(session),
        "mode": "full",
        "start_x": start_x,
        "end_x": end_x,
        "delta_x": end_x - start_x,
        "error": "never reached speed_boosting (echoes≥4)",
    }


def crouch_store(
    session: ControllerSession,
    *,
    max_frames: int = DEFAULT_STORE_MAX_FRAMES,
    label: str = "store",
) -> dict[str, Any]:
    """Hold DOWN until ``$0A68 > 0`` (first armed frame) or budget ends.

    Call when grounded with full echoes (pose 9 preferred). Returns
    ``armed`` snap, ``peak_timer``, and ``store_frame_index`` (0-based).
    """
    armed: dict[str, Any] | None = None
    peak = 0
    start_pose = int(session.state.pose)

    for i in range(max_frames):
        hold(session, 1, "DOWN", reason=f"{label}_{i}")
        try:
            w = session_spark_wram(session)
            timer = w["spark_timer"]
        except AttributeError:
            timer = int(getattr(session.state, "shinespark_timer", 0))
        peak = max(peak, timer)
        if armed is None and timer > 0:
            armed = _snap(session)
            armed["store_frame_index"] = i
            armed["start_pose"] = start_pose
            break

    return {
        "ok": armed is not None,
        "armed": armed,
        "peak_timer_during_store": peak,
        "after": _snap(session),
        "start_pose": start_pose,
        "error": None
        if armed is not None
        else f"store never armed $0A68 (peak={peak}, start_pose={start_pose})",
    }


def wait_store_window(
    session: ControllerSession,
    frames: int,
    *,
    hold_down: bool = False,
    label: str = "idle",
) -> dict[str, Any]:
    """Idle after store; track ``$0A68`` drain for ``frames`` (or until zero).

    ``hold_down=True`` keeps crouch (timer still drains). Default releases to
    neutral so the controller can hop / re-aim before activate.
    """
    series: list[dict[str, Any]] = []
    alive = 0
    for i in range(max(0, frames)):
        if hold_down:
            hold(session, 1, "DOWN", reason=f"{label}_store_{i}")
        else:
            hold(session, 1, reason=f"{label}_{i}")
        row = _snap(session)
        series.append(row)
        timer = int(row.get("spark_timer", row.get("shinespark_timer", 0)))
        if timer > 0:
            alive += 1
        elif i > 0 and series and int(
            series[0].get("spark_timer", series[0].get("shinespark_timer", 0))
        ) > 0:
            break

    def _t(row: dict[str, Any]) -> int:
        return int(row.get("spark_timer", row.get("shinespark_timer", 0)))

    return {
        "requested_frames": frames,
        "hold_down": hold_down,
        "frames_timer_gt0": alive,
        "timer_start": _t(series[0]) if series else 0,
        "timer_end": _t(series[-1]) if series else 0,
        "first_zero_frame": next(
            (r.get("frame") for r in series if _t(r) == 0), None
        ),
        "samples": series if len(series) <= 120 else series[:: max(1, len(series) // 60)],
        "sample_count": len(series),
        "drain_per_frame": (
            (_t(series[0]) - _t(series[-1])) / max(1, len(series) - 1)
            if len(series) > 1
            else None
        ),
    }


def activate_shinespark(
    session: ControllerSession,
    *aim_buttons: str,
    activate_button: str = "A",
    hold_frames: int = 12,
    travel_budget: int = 0,
    travel_hold: Sequence[str] | None = None,
    pre_stand_frames: int = DEFAULT_PRE_STAND_FRAMES,
    pre_stand_buttons: Sequence[str] = ("UP",),
    label: str = "activate",
) -> dict[str, Any]:
    """Press aim + jump (A) to fire the shinespark.

    Examples
    --------
    * Horizontal right: ``activate_shinespark(session, "RIGHT")``
    * Diagonal up-right: ``activate_shinespark(session, "RIGHT", "UP")``
    * Vertical up: ``activate_shinespark(session, "UP")``

    Harness: **A** is jump / shine-activate; **B** only walks after store.

    After crouch-store, Samus is often pose 53 — immediate aim+A usually
    walks instead of sparking. Default ``pre_stand_frames`` holds UP (or
    neutral if ``pre_stand_buttons`` is empty) so the crystal flash arms
    cleanly before A.
    """
    aim = tuple(aim_buttons)
    pre_snap: dict[str, Any] | None = None
    if pre_stand_frames > 0:
        for i in range(pre_stand_frames):
            if pre_stand_buttons:
                hold(
                    session,
                    1,
                    *pre_stand_buttons,
                    reason=f"{label}_pre_{i}",
                )
            else:
                hold(session, 1, reason=f"{label}_pre_{i}")
        pre_snap = _snap(session)

    activate_snap: dict[str, Any] | None = None
    spark_seen = False
    min_y = int(session.state.samus_y)
    max_x = int(session.state.samus_x)
    min_x = int(session.state.samus_x)
    start_room = int(session.state.room_id)

    for i in range(hold_frames):
        hold(
            session,
            1,
            *aim,
            activate_button,
            reason=f"{label}_{i}",
        )
        st = session.state
        min_y = min(min_y, int(st.samus_y))
        max_x = max(max_x, int(st.samus_x) if int(st.samus_x) < 60000 else max_x)
        min_x = min(min_x, int(st.samus_x) if int(st.samus_x) < 60000 else min_x)
        if is_spark_pose(int(st.pose)):
            spark_seen = True
            if activate_snap is None or not is_spark_pose(
                int(activate_snap.get("pose", -1))
            ):
                activate_snap = _snap(session)
                activate_snap["activate_frame_index"] = i
        try:
            timer = session_spark_wram(session)["spark_timer"]
        except AttributeError:
            timer = int(getattr(st, "shinespark_timer", 0))
        # Do not latch on store timer alone (already >0 before A); wait for pose.

    travel_rows: list[dict[str, Any]] = []
    if travel_budget > 0:
        hold_btns = (
            tuple(travel_hold)
            if travel_hold is not None
            else (*aim, activate_button)
        )
        for i in range(travel_budget):
            hold(session, 1, *hold_btns, reason=f"{label}_travel_{i}")
            st = session.state
            min_y = min(min_y, int(st.samus_y))
            if int(st.samus_x) < 60000:
                max_x = max(max_x, int(st.samus_x))
                min_x = min(min_x, int(st.samus_x))
            if i % 10 == 0 or is_spark_pose(int(st.pose)):
                travel_rows.append(_snap(session))
            # Stop if timer died and no longer sparking
            try:
                timer = session_spark_wram(session)["spark_timer"]
            except AttributeError:
                timer = int(getattr(st, "shinespark_timer", 0))
            if (
                timer == 0
                and not is_spark_pose(int(st.pose))
                and i > 8
            ):
                break

    final = _snap(session)
    return {
        "ok": spark_seen or is_spark_pose(int(session.state.pose)),
        "aim": aim,
        "pre_stand": pre_snap,
        "pre_stand_frames": pre_stand_frames,
        "pre_stand_buttons": list(pre_stand_buttons),
        "activate": activate_snap or final,
        "spark_pose_seen": spark_seen,
        "final": final,
        "min_y": min_y,
        "max_x": max_x,
        "min_x": min_x,
        "start_room": start_room,
        "end_room": int(session.state.room_id),
        "room_changed": int(session.state.room_id) != start_room,
        "travel_samples": travel_rows,
    }


def store_then_spin_unspin_activate(
    session: ControllerSession,
    *,
    stand_frames: int = 8,
    hop_frames: int = 13,
    hop_direction: Direction = "RIGHT",
    unspin_frames: int = 4,
    unspin_buttons: Sequence[str] = ("UP",),
    aim_buttons: Sequence[str] = ("RIGHT",),
    micro_run_frames: int = 0,
    activate_hold: int = 12,
    travel_budget: int = 300,
    label: str = "hop_carry",
) -> dict[str, Any]:
    """Recipe: after store, stand → optional micro-run → spin hop → unspin → A.

    Wiki-aligned hop-carry: spin does not consume store; unspin mid-air then
    jump-activate. Call **after** :func:`crouch_store` has armed ``$0A68``.
    """
    dir_btn = "LEFT" if hop_direction == "LEFT" else "RIGHT"
    report: dict[str, Any] = {
        "params": {
            "stand_frames": stand_frames,
            "hop_frames": hop_frames,
            "hop_direction": dir_btn,
            "unspin_frames": unspin_frames,
            "unspin_buttons": list(unspin_buttons),
            "aim_buttons": list(aim_buttons),
            "micro_run_frames": micro_run_frames,
        }
    }

    if stand_frames > 0:
        hold(session, stand_frames, reason=f"{label}_stand")
    report["after_stand"] = _snap(session)

    if micro_run_frames > 0:
        hold(
            session,
            micro_run_frames,
            dir_btn,
            "B",
            reason=f"{label}_micro_run",
        )
        report["after_micro_run"] = _snap(session)

    # Spin hop: direction + dash + jump
    hold(
        session,
        hop_frames,
        dir_btn,
        "B",
        "A",
        reason=f"{label}_hop",
    )
    report["after_hop"] = _snap(session)

    if unspin_frames > 0 and unspin_buttons:
        hold(
            session,
            unspin_frames,
            *unspin_buttons,
            reason=f"{label}_unspin",
        )
    report["after_unspin"] = _snap(session)

    report["activate"] = activate_shinespark(
        session,
        *aim_buttons,
        hold_frames=activate_hold,
        travel_budget=travel_budget,
        pre_stand_frames=0,  # hop recipe already stood/unspun
        label=f"{label}_act",
    )
    report["ok"] = bool(report["activate"].get("ok"))
    return report


def charge_store_activate(
    session: ControllerSession,
    *,
    direction: Direction = "RIGHT",
    charge_budget: int = 500,
    store_max_frames: int = DEFAULT_STORE_MAX_FRAMES,
    idle_after_store: int = 0,
    hold_down_idle: bool = False,
    aim_buttons: Sequence[str] = ("RIGHT",),
    travel_budget: int = 200,
    label: str = "spark",
    charge_mode: ChargeMode = "full",
    region: Region = "NTSC",
    store_on_last_magic: bool = False,
) -> dict[str, Any]:
    """Full pipeline: charge → crouch-store → optional idle → activate+travel.

    ``charge_mode`` ``short`` / ``stutter`` uses magic-frame dash (see
    :func:`short_charge_until_boost`). When ``store_on_last_magic`` is True with
    a short mode, DOWN is pressed on the final magic frame; a separate
    crouch-store is skipped if ``$0A68`` is already armed.
    """
    report: dict[str, Any] = {"label": label, "charge_mode": charge_mode}
    report["charge"] = charge_until_boost(
        session,
        direction,
        budget=charge_budget,
        label=f"{label}_charge",
        mode=charge_mode,
        region=region,
        store_on_last=store_on_last_magic and charge_mode in ("short", "stutter"),
    )
    if not report["charge"].get("ok"):
        report["ok"] = False
        report["error"] = report["charge"].get("error")
        return report

    already_armed = bool(report["charge"].get("store_armed"))
    if already_armed:
        report["store"] = {
            "ok": True,
            "armed": report["charge"].get("boost"),
            "peak_timer_during_store": int(report["charge"].get("spark_timer") or 0),
            "after": _snap(session),
            "start_pose": int(session.state.pose),
            "via": "store_on_last_magic",
        }
    else:
        report["store"] = crouch_store(
            session, max_frames=store_max_frames, label=f"{label}_store"
        )
        if not report["store"].get("ok"):
            report["ok"] = False
            report["error"] = report["store"].get("error")
            return report

    if idle_after_store > 0:
        report["window"] = wait_store_window(
            session,
            idle_after_store,
            hold_down=hold_down_idle,
            label=f"{label}_idle",
        )
    else:
        armed = report["store"]["armed"] or {}
        report["window"] = {
            "requested_frames": 0,
            "frames_timer_gt0": 1 if int(armed.get("spark_timer", 0)) > 0 else 0,
            "timer_start": armed.get("spark_timer"),
            "timer_end": armed.get("spark_timer"),
        }

    report["activate"] = activate_shinespark(
        session,
        *aim_buttons,
        travel_budget=travel_budget,
        label=f"{label}_act",
    )
    report["ok"] = bool(report["activate"].get("ok"))
    if not report["ok"]:
        report["error"] = "activate did not enter spark pose"
    return report


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _snap(session: Any) -> dict[str, Any]:
    try:
        return spark_snapshot(_env_of(session), frame=int(session.frame))
    except AttributeError:
        st = session.state
        return {
            "frame": int(getattr(session, "frame", 0)),
            "room": int(st.room_id),
            "room_hex": f"0x{int(st.room_id):04X}",
            "x": int(st.samus_x),
            "y": int(st.samus_y),
            "pose": int(st.pose),
            "facing": int(getattr(st, "facing", 0)),
            "vx": int(st.velocity_x),
            "vy": int(st.velocity_y),
            "spark_timer": int(getattr(st, "shinespark_timer", 0)),
            "speed_echoes": int(getattr(st, "speed_counter", 0)),
            "speed_boosting": bool(getattr(st, "speed_boosting", False)),
            "shinesparking": bool(getattr(st, "shinesparking", False)),
        }


def diagnose_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify a shine attempt from per-frame WRAM/nav rows."""
    empty = {
        "ok": False, "grade": "EMPTY", "failures": ["no frames recorded"],
        "cues": [], "peaks": {}, "milestones": {},
    }
    if not trace:
        return empty

    def btns(row: dict[str, Any]) -> set[str]:
        return {str(b).upper() for b in (row.get("buttons") or [])}

    peak_e = spark_n = down_n = 0
    first_store = first_spark = start_i = None
    wins: list[tuple[int, int]] = []
    for i, row in enumerate(trace):
        e = int(row.get("speed_echoes") or 0)
        t = int(row.get("spark_timer") or 0)
        pose = int(row.get("pose") or 0)
        peak_e = max(peak_e, e)
        if first_store is None and t > 0:
            first_store = int(row.get("frame") or i)
        if is_spark_pose(pose):
            spark_n += 1
            if first_spark is None:
                first_spark = int(row.get("frame") or i)
        if "DOWN" in btns(row):
            down_n += 1
        if e >= ECHOES_FULL and start_i is None:
            start_i = i
        elif e < ECHOES_FULL and start_i is not None:
            wins.append((start_i, i - 1))
            start_i = None
    if start_i is not None:
        wins.append((start_i, len(trace) - 1))

    missed = 0
    kill_dir = False
    for s, e_i in wins:
        if any("DOWN" in btns(r) for r in trace[s : e_i + 1]):
            continue
        if any("DOWN" in btns(r) for r in trace[e_i + 1 :]):
            missed += 1
            nxt = btns(trace[e_i + 1]) if e_i + 1 < len(trace) else set()
            if "B" in nxt and "RIGHT" not in nxt and "LEFT" not in nxt:
                kill_dir = True
    late = first_store is None and peak_e >= ECHOES_FULL and missed > 0 and down_n > 0

    crouch_walk = False
    if first_store is not None and first_spark is None:
        a_hold = 0
        for r in trace:
            if int(r.get("spark_timer") or 0) <= 0:
                continue
            b = btns(r)
            pose = int(r.get("pose") or 0)
            if "A" in b and ({"RIGHT", "LEFT", "UP"} & b):
                a_hold += 1
            if pose in (39, 40, 53, 54) and "A" in b and "RIGHT" in b:
                crouch_walk = True
        crouch_walk = crouch_walk or a_hold >= 8

    ok = first_spark is not None and spark_n >= 3
    failures: list[str] = []
    cues: list[str] = []
    if peak_e < ECHOES_FULL:
        failures.append(f"charge incomplete (peak echoes={peak_e}, need ≥4)")
    elif first_store is None:
        failures.append("never crouch-stored ($0A68 stayed 0)")
        if late:
            failures.append("late crouch: charged but DOWN never during echoes=4")
            if kill_dir:
                failures.append("boost killed by releasing LEFT/RIGHT while keeping B")
            cues.append("CRITICAL: ALSO press DOWN while still holding RIGHT+B")
    elif first_spark is None:
        failures.append("stored but never entered spark pose")
    if ok:
        grade = "GREEN"
    elif peak_e >= ECHOES_FULL and first_store is not None:
        grade = "YELLOW"
    elif peak_e >= ECHOES_FULL:
        grade = "ORANGE"
    else:
        grade = "RED"
    return {
        "ok": ok, "grade": grade, "failures": failures, "cues": cues,
        "peaks": {"echoes": peak_e, "spark_travel_frames": spark_n,
                  "missed_store_windows": missed},
        "milestones": {"late_store_after_charge_died": late,
                       "activate_from_crouch_walk": crouch_walk},
    }


__all__ = [
    "ECHOES_FULL",
    "TYPICAL_ARM_TIMER",
    "TYPICAL_CHARGE_FRAMES",
    "DEFAULT_STORE_MAX_FRAMES",
    "DEFAULT_PRE_STAND_FRAMES",
    "SPARK_POSES",
    "STORE_OK_POSES",
    "STORE_WIPE_POSES",
    "KNOCKBACK_POSES",
    "NTSC_MAGIC_DASH_FRAMES",
    "PAL_MAGIC_DASH_FRAMES",
    "NTSC_STUTTER_MIN_PX",
    "PAL_STUTTER_MIN_PX",
    "NTSC_STUTTER_FULL_STOP_PX",
    "NTSC_SHORT_CHARGE_FRAMES",
    "PAL_SHORT_CHARGE_FRAMES",
    "read_spark_wram",
    "spark_snapshot",
    "session_spark_wram",
    "is_spark_pose",
    "store_pose_ok",
    "magic_dash_frames",
    "stutter_forward_mask",
    "stutter_dash_mask",
    "short_charge_plan",
    "charge_by_plan",
    "short_charge_until_boost",
    "charge_until_boost",
    "crouch_store",
    "wait_store_window",
    "activate_shinespark",
    "store_then_spin_unspin_activate",
    "charge_store_activate",
    "diagnose_trace",
]
