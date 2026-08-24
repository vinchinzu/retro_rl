"""KPDR Charge Plus Missiles Phantoon (MassHesteria).

https://wiki.supermetroid.run/Phantoon#Any.25_KPDR_.28Charge_Plus_Missiles.29

Round: 2 missiles, wait, 2 missiles, Charge Shot (4 rounds, ~700 each).
Super only if HP ≤ 600 (else enrage). Default Super off. Not product
charge-only (20537f ×2, 9×300). No X-factor. Seat left; fire on measured
open-eye with product skip list (right x≥155; rain unless (48,96) y 88–104).
Hits by ammo/charge delta + enemy0_hp, not X. Boss bit ``$7E:D82B`` via
``read_bank7e_wram``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from super_metroid.combat.features import boss_defeated_in_state, phantoon_catalog
from super_metroid.combat.phantoon import (
    ADDR_WS_BOSS_BITS,
    HURT_POSES,
    PHANTOON_BOSS_BIT,
    ROOM_PHANTOON,
    WEAPON_BEAM,
    WEAPON_MISSILES,
    WEAPON_SUPERS,
    PhantoonStrategy,
    _dead,
    _env_of,
    _fire_window,
    _go_to_seat,
    _hittable,
    _rain_corner_wait,
    charge_window_ok,
    enemy_extra,
    eye_open,
    in_release_band,
    rain_phase,
    right_park,
    seated,
)
from super_metroid.combat.primitives import ensure_weapon
from super_metroid.ram import SuperMetroidState, read_bank7e_wram
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.runtime import ControllerSession, hold

ROUND_RECIPE: tuple[str, str, str] = ("missiles", "missiles", "charge")
MISSILES_PER_BARRAGE = 2
CHARGE_SHOTS_PER_ROUND = 1
MISSILE_SPACING = 10
BARRAGE_GAP_FRAMES = 24
SUPER_KILL_HP = 600
PRODUCT_BENCH_FRAMES = 20537
WIKI_URL = (
    "https://wiki.supermetroid.run/Phantoon"
    "#Any.25_KPDR_.28Charge_Plus_Missiles.29"
)
go_to_seat = _go_to_seat


@dataclass(frozen=True)
class ChargeMissilesStrategy:
    """Left-seat 2+2+charge rounds. Super gated at HP ≤ 600, default off."""

    allow_super: bool = False
    missiles_per_barrage: int = MISSILES_PER_BARRAGE
    charge_shots: int = CHARGE_SHOTS_PER_ROUND
    missile_spacing: int = MISSILE_SPACING
    barrage_gap_frames: int = BARRAGE_GAP_FRAMES
    super_kill_hp: int = SUPER_KILL_HP
    max_fight_frames: int = 40_000
    window_timeout: int = 480
    boss_bit_grace_frames: int = 1_200
    first_open_timeout: int = 2_400

    def phantoon(self, *, weapon: int, shots: int) -> PhantoonStrategy:
        return PhantoonStrategy(
            weapon=weapon,
            shots_per_window=shots,
            max_fight_frames=self.max_fight_frames,
            window_timeout=self.window_timeout,
            boss_bit_grace_frames=self.boss_bit_grace_frames,
        )


@dataclass
class ChargeMissilesEvidence:
    """Measured Charge Plus Missiles fight (or first-window probe)."""

    start_frame: int
    body_zero_frame: int | None
    boss_bit_frame: int | None
    end_frame: int
    peak_body_hp: int
    min_body_hp: int
    action_frames: int
    final_body_hp: int
    boss_bit_set: bool
    outcome: str
    rounds: int = 0
    windows: int = 0
    missiles_spent: int = 0
    charges_spent: int = 0
    supers_spent: int = 0
    round_log: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame, "body_zero_frame": self.body_zero_frame,
            "boss_bit_frame": self.boss_bit_frame, "end_frame": self.end_frame,
            "peak_body_hp": self.peak_body_hp, "min_body_hp": self.min_body_hp,
            "action_frames": self.action_frames, "final_body_hp": self.final_body_hp,
            "boss_bit_set": self.boss_bit_set, "outcome": self.outcome,
            "rounds": self.rounds, "windows": self.windows,
            "shots": {
                "missiles": self.missiles_spent, "charges": self.charges_spent,
                "supers": self.supers_spent,
            },
            "round_log": list(self.round_log),
            **format_segment_time(self.action_frames),
        }


def should_fire_super(
    hp: int, *, allow_super: bool = False, kill_hp: int = SUPER_KILL_HP
) -> bool:
    """True only when Super is armed and remaining HP is a guaranteed kill."""
    return bool(allow_super) and 0 < int(hp) <= int(kill_hp)


def round_recipe(*, allow_super: bool = False, hp: int = 2500) -> tuple[str, ...]:
    """Per-round weapon order. Super replaces charge only on a kill HP."""
    if should_fire_super(hp, allow_super=allow_super):
        return ("missiles", "missiles", "super")
    return ROUND_RECIPE

def fight_charge_missiles_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: ChargeMissilesStrategy | None = None,
    *,
    round_step: str = "missiles",
) -> tuple[str, ...]:
    """One-frame seat / fire hint. Play loop owns counted shots."""
    del frame_index
    strat = strategy or ChargeMissilesStrategy()
    p = PhantoonStrategy()
    if int(state.enemy0_hp) == 0:
        return ()
    if not seated(state):
        if int(state.samus_y) < p.floor_y_min:
            return ()
        if int(state.samus_x) > p.seat_x_max:
            return ("LEFT",)
        return ("RIGHT",) if int(state.samus_x) < p.seat_x_min else ()
    if not eye_open(state) or right_park(int(state.enemy0_x)):
        return ()
    if round_step == "super":
        ok = should_fire_super(
            int(state.enemy0_hp), allow_super=strat.allow_super, kill_hp=strat.super_kill_hp
        )
        return ("X",) if ok else ()
    if round_step == "missiles" and int(state.missiles) <= 0:
        return ()
    return ("X",) if round_step in ("missiles", "charge") else ()


def _body_func(session: ControllerSession) -> int:
    extra = enemy_extra(_env_of(session))
    try:
        return int(str(extra.get("func") or "0"), 16)
    except ValueError:
        return 0


def _boss_bit(session: ControllerSession) -> bool:
    env = _env_of(session)
    if env is not None:
        try:
            return bool(int(read_bank7e_wram(env)[ADDR_WS_BOSS_BITS]) & PHANTOON_BOSS_BIT)
        except Exception:
            pass
    return boss_defeated_in_state(session.state, phantoon_catalog())


def _park_ok(session: ControllerSession, park_x: int) -> bool:
    return charge_window_ok(_body_func(session), int(park_x), session.state.enemy0_y)


def _open_ok(session: ControllerSession, park_x: int) -> bool:
    st = session.state
    return eye_open(st, _env_of(session)) and _park_ok(session, park_x)


def _wait_tick(session: ControllerSession, pstrat: PhantoonStrategy, park_x: int) -> None:
    """Product wait: snipe skip parks, LEFT-to-seat, hold X on beam (RNG)."""
    func = _body_func(session)
    skip = rain_phase(func) or right_park(int(park_x)) or (0 < int(park_x) < 100)
    if skip:
        _rain_corner_wait(session, pstrat)
        return
    names: list[str] = []
    if int(session.state.samus_x) > pstrat.seat_x_max:
        names.append("LEFT")
    if int(session.state.selected_item) == WEAPON_BEAM:
        names.append("X")
    hold(session, 1, *tuple(names), reason="phan_cm_wait")


def _ammo_or_hp_hit(session: ControllerSession, *, ammo_before: int, hp_before: int, ammo_attr: str) -> bool:
    st = session.state
    return int(getattr(st, ammo_attr)) < ammo_before or int(st.enemy0_hp) < hp_before


def _fast_missiles(session: ControllerSession) -> None:
    if int(session.state.selected_item) == WEAPON_MISSILES or int(session.state.missiles) <= 0:
        return
    hold(session, 1, "SELECT", reason="phan_cm_ms_sel")
    for _ in range(20):
        if int(session.state.selected_item) == WEAPON_MISSILES:
            return
        hold(session, 1, reason="phan_cm_ms_sel")


def _fire_missile_barrage(session: ControllerSession, pstrat: PhantoonStrategy) -> int:
    """Jump-fire up to ``shots_per_window`` missiles at 10f from the left seat."""
    if int(session.state.missiles) <= 0:
        return 0
    _fast_missiles(session)
    if int(session.state.selected_item) != WEAPON_MISSILES:
        try:
            ensure_weapon(session, WEAPON_MISSILES)
        except RuntimeError:
            return 0
    hits = 0
    last_spend = -99
    seen_open = False
    pending_hp: int | None = None
    n = int(pstrat.shots_per_window)
    for _ in range(min(180, pstrat.window_timeout)):
        st = session.state
        if _dead(session) or int(st.enemy0_hp) == 0 or hits >= n or int(st.missiles) <= 0:
            break
        if pending_hp is not None and int(st.enemy0_hp) < pending_hp:
            hits += 1
            pending_hp = None
            if hits >= n:
                break
        hittable = _hittable(session)
        seen_open = seen_open or hittable
        if not hittable and seen_open:
            break
        if int(st.pose) in HURT_POSES:
            hold(session, 1, reason="phan_cm_hurt")
            continue
        dx = int(st.enemy0_x) - int(st.samus_x)
        close = abs(dx) <= max(pstrat.fire_close_x, 24)
        on_floor = int(st.samus_y) >= pstrat.floor_y_min - 6
        names: list[str] = []
        if int(st.facing) != 8:
            names.append("RIGHT")
        elif dx > 16:
            names.append("RIGHT")
            if on_floor:
                names.append("B")
        elif dx < -16:
            names.append("LEFT")
        if hittable and close and on_floor:
            names.append("A")
        if int(st.samus_y) > int(st.enemy0_y) + 10:
            names.append("UP")
        fire = (
            hittable and close and (not on_floor) and int(st.missiles) > 0
            and hits < n and (session.frame - last_spend) >= MISSILE_SPACING
            and in_release_band(st, pstrat)
        )
        hp_before, ms_before = int(st.enemy0_hp), int(st.missiles)
        if fire:
            names = ["UP", "X"] if int(st.samus_y) > int(st.enemy0_y) + 10 else ["X"]
        hold(session, 1, *tuple(dict.fromkeys(names)), reason="phan_cm_ms")
        if not fire:
            continue
        last_spend = session.frame
        if int(session.state.enemy0_hp) < hp_before:
            hits += 1
            pending_hp = None
        else:
            pending_hp = hp_before
    return hits


def _fire_super_kill(session: ControllerSession, pstrat: PhantoonStrategy, *, kill_hp: int) -> int:
    hp = int(session.state.enemy0_hp)
    if not (0 < hp <= kill_hp) or int(session.state.super_missiles) <= 0:
        return 0
    try:
        ensure_weapon(session, WEAPON_SUPERS)
    except RuntimeError:
        return 0
    for _ in range(min(90, pstrat.window_timeout)):
        st = session.state
        if _dead(session) or int(st.enemy0_hp) == 0 or not _hittable(session):
            return 0
        if int(st.pose) in HURT_POSES:
            hold(session, 1, reason="phan_cm_hurt")
            continue
        hp_before, sm_before = int(st.enemy0_hp), int(st.super_missiles)
        hold(session, 1, "X", reason="phan_cm_super")
        if _ammo_or_hp_hit(
            session, ammo_before=sm_before, hp_before=hp_before, ammo_attr="super_missiles"
        ):
            return 1
    return 0


def _gap_wait(session: ControllerSession, pstrat: PhantoonStrategy, frames: int) -> None:
    for _ in range(max(0, frames)):
        st = session.state
        if _dead(session) or int(st.enemy0_hp) == 0 or not _hittable(session):
            return
        hold(session, 1, *(("LEFT",) if int(st.samus_x) > pstrat.kite_x_max else ()), reason="phan_cm_gap")


def play_round(
    session: ControllerSession, strategy: ChargeMissilesStrategy | None = None
) -> dict[str, int]:
    """One wiki round: 2 missiles, gap, 2 missiles, charge (optional Super)."""
    strat = strategy or ChargeMissilesStrategy()
    ms_p = strat.phantoon(weapon=WEAPON_MISSILES, shots=strat.missiles_per_barrage)
    beam_p = strat.phantoon(weapon=WEAPON_BEAM, shots=strat.charge_shots)
    hp0 = int(session.state.enemy0_hp)
    a = _fire_missile_barrage(session, ms_p)
    _gap_wait(session, ms_p, strat.barrage_gap_frames)
    b = 0
    if _hittable(session) and int(session.state.enemy0_hp) > 0:
        b = _fire_missile_barrage(session, ms_p)
    charges = supers = 0
    hp_mid = int(session.state.enemy0_hp)
    if should_fire_super(hp_mid, allow_super=strat.allow_super, kill_hp=strat.super_kill_hp) and _hittable(session):
        supers = _fire_super_kill(session, ms_p, kill_hp=strat.super_kill_hp)
    elif _hittable(session) and hp_mid > 0:
        charges = int(_fire_window(session, beam_p))
        hp_after = int(session.state.enemy0_hp)
        if should_fire_super(hp_after, allow_super=strat.allow_super, kill_hp=strat.super_kill_hp):
            supers = _fire_super_kill(session, beam_p, kill_hp=strat.super_kill_hp)
    if not _dead(session) and int(session.state.enemy0_hp) > 0:
        _go_to_seat(session, ms_p)
    return {
        "missiles": a + b, "missiles_a": a, "missiles_b": b,
        "charges": charges, "supers": supers,
        "hp_drop": hp0 - int(session.state.enemy0_hp),
        "hp_after": int(session.state.enemy0_hp),
    }


def _wait_open(session: ControllerSession, pstrat: PhantoonStrategy, *, timeout: int) -> bool:
    park_x = int(session.state.enemy0_x)
    last_func: int | None = None
    for _ in range(timeout):
        if _dead(session) or int(session.state.enemy0_hp) == 0:
            return False
        func_now = _body_func(session)
        if func_now != last_func:
            park_x = int(session.state.enemy0_x)
            last_func = func_now
        if _open_ok(session, park_x):
            return True
        _wait_tick(session, pstrat, park_x)
    return _open_ok(session, park_x)


def play_first_missile_window(
    session: ControllerSession, strategy: ChargeMissilesStrategy | None = None
) -> dict[str, object]:
    """Seat, wait first charge_window_ok open, fire the 2-missile opener."""
    strat = strategy or ChargeMissilesStrategy()
    pstrat = strat.phantoon(weapon=WEAPON_MISSILES, shots=strat.missiles_per_barrage)
    beam_p = strat.phantoon(weapon=WEAPON_BEAM, shots=strat.charge_shots)
    start = session.frame
    if session.state.room_id != ROOM_PHANTOON:
        raise RuntimeError(
            f"Charge+Missiles expected room 0x{ROOM_PHANTOON:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )
    _go_to_seat(session, beam_p)
    opened = _wait_open(session, beam_p, timeout=strat.first_open_timeout)
    pre_ms, pre_hp = int(session.state.missiles), int(session.state.enemy0_hp)
    extra = enemy_extra(_env_of(session))
    hits = 0
    if opened and pre_ms > 0:
        hits = _fire_missile_barrage(session, pstrat)
    if not _dead(session) and int(session.state.enemy0_hp) > 0:
        _go_to_seat(session, pstrat)
    hp_drop = pre_hp - int(session.state.enemy0_hp)
    return {
        "opened": opened, "seated": seated(session.state, pstrat),
        "missiles_before": pre_ms, "missiles_after": int(session.state.missiles),
        "missiles_delta": pre_ms - int(session.state.missiles), "missiles_hits": hits,
        "enemy0_hp_before": pre_hp, "enemy0_hp_after": int(session.state.enemy0_hp),
        "hp_drop": hp_drop,
        "func": extra.get("func"),
        "eye_ilist": extra.get("eye_ilist"),
        "enemy_xy": [int(session.state.enemy0_x), int(session.state.enemy0_y)],
        "success": opened and hp_drop > 0 and hits >= 1,
        **format_segment_time(session.frame - start),
    }


def play_charge_missiles_fight(
    session: ControllerSession,
    *,
    strategy: ChargeMissilesStrategy | None = None,
    require_boss_bit: bool = True,
) -> ChargeMissilesEvidence:
    """Fight Phantoon with 2+2+charge rounds until HP 0 + optional boss bit."""
    strat = strategy or ChargeMissilesStrategy()
    ms_p = strat.phantoon(weapon=WEAPON_MISSILES, shots=strat.missiles_per_barrage)
    beam_p = strat.phantoon(weapon=WEAPON_BEAM, shots=strat.charge_shots)
    start = session.frame
    if session.state.room_id != ROOM_PHANTOON:
        raise RuntimeError(
            f"Charge+Missiles expected room 0x{ROOM_PHANTOON:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )
    try:
        ensure_weapon(session, WEAPON_BEAM)
    except RuntimeError:
        pass
    _go_to_seat(session, beam_p)
    peak_hp = min_hp = prev_hp = int(session.state.enemy0_hp)
    body_zero_frame: int | None = start if peak_hp == 0 else None
    boss_bit_frame: int | None = None
    rounds = windows = missiles_spent = charges_spent = supers_spent = 0
    round_log: list[dict[str, object]] = []
    park_x = int(session.state.enemy0_x)
    last_func: int | None = None
    for _ in range(strat.max_fight_frames):
        state = session.state
        if state.room_id != ROOM_PHANTOON or _dead(session):
            break
        peak_hp = max(peak_hp, int(state.enemy0_hp))
        min_hp = min(min_hp, int(state.enemy0_hp))
        if body_zero_frame is None and int(state.enemy0_hp) == 0 and prev_hp > 0:
            body_zero_frame = session.frame
            min_hp = 0
        prev_hp = int(state.enemy0_hp)
        if body_zero_frame is not None:
            if _boss_bit(session):
                boss_bit_frame = session.frame
                break
            if (not require_boss_bit) or session.frame - body_zero_frame >= strat.boss_bit_grace_frames:
                break
            hold(session, 1, reason="phantoon_death_anim")
            continue
        func_now = _body_func(session)
        if func_now != last_func:
            park_x = int(state.enemy0_x)
            last_func = func_now
        if _open_ok(session, park_x):
            row = play_round(session, strat)
            landed = int(row["missiles"]) + int(row["charges"]) + int(row["supers"])
            if landed:
                rounds += 1
                windows += 1
                missiles_spent += int(row["missiles"])
                charges_spent += int(row["charges"])
                supers_spent += int(row["supers"])
                round_log.append({"frame": session.frame, "round": rounds, **row})
            else:
                hold(session, 1, reason="phan_cm_wait")
            continue
        _wait_tick(session, beam_p, park_x)
    boss_set = _boss_bit(session)
    if _dead(session):
        outcome = "died"
    elif boss_set:
        outcome = "phantoon_defeated"
    elif body_zero_frame is not None:
        outcome = "phantoon_body_zero_no_boss_bit"
    else:
        outcome = "timeout"
    return ChargeMissilesEvidence(
        start_frame=start, body_zero_frame=body_zero_frame,
        boss_bit_frame=boss_bit_frame, end_frame=session.frame,
        peak_body_hp=peak_hp, min_body_hp=min_hp,
        action_frames=session.frame - start,
        final_body_hp=int(session.state.enemy0_hp),
        boss_bit_set=boss_set, outcome=outcome, rounds=rounds, windows=windows,
        missiles_spent=missiles_spent, charges_spent=charges_spent,
        supers_spent=supers_spent, round_log=round_log,
    )


__all__ = [
    "BARRAGE_GAP_FRAMES", "CHARGE_SHOTS_PER_ROUND", "MISSILE_SPACING",
    "MISSILES_PER_BARRAGE", "PRODUCT_BENCH_FRAMES", "ROUND_RECIPE",
    "SUPER_KILL_HP", "WIKI_URL", "ChargeMissilesEvidence",
    "ChargeMissilesStrategy", "fight_charge_missiles_action", "go_to_seat",
    "play_charge_missiles_fight", "play_first_missile_window", "play_round",
    "round_recipe", "seated", "should_fire_super",
]
