"""Wiki missile-doppler Phantoon (PRKD recipe on KPDR inventory).

Public policy: https://wiki.supermetroid.run/Phantoon
  https://wiki.supermetroid.run/Phantoon#Any.25_PRKD
  https://wiki.supermetroid.run/Phantoon#Any.25_KPDR_.28Charge_Plus_Missiles.29
  (KPDR Room Strategies: “Dopplers” / Charge+Missiles mini-doppler)

Vanish-damage still needs ~10f to close the eye; missile cooldown is 9f;
wiki spacing is **10 frames**. Super 600 **enrages** if he lives —
Super-only-if-kill (HP ≤ 600). Recipe 2-2-N: optional uncharged beam tap
(do not dump a charge); two missiles, wait ~30–60f; two missiles, let him
gain distance; walk in, mash at 10f (cap ~6 extra); retreat seat. Skip
x≥155 right park; skip rain unless (48, 96) y 88–104. Rain reuses product
corner snipe — do not jump under the body. Hits: missiles decreasing +
enemy0_hp delta. Boss bit: ``read_bank7e_wram`` ``$7E:D82B`` bit 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from super_metroid.combat.features import boss_defeated_in_state, phantoon_catalog
from super_metroid.combat.phantoon import (
    ADDR_WS_BOSS_BITS, HURT_POSES, PHANTOON_BOSS_BIT, ROOM_PHANTOON,
    WEAPON_BEAM, WEAPON_MISSILES, WEAPON_SUPERS, PhantoonStrategy,
    _go_to_seat, _rain_corner_wait, charge_window_ok, enemy_extra, eye_open,
    rain_phase, right_park, seated,
)
from super_metroid.combat.primitives import ensure_weapon
from super_metroid.ram import GameplayPhase, SuperMetroidState, read_bank7e_wram
from super_metroid.routes.controller_common import is_morph, unmorph
from super_metroid.routes.runtime import ControllerSession, hold

PAIR_SIZE = 2
MAX_DOPPLER_EXTRA = 6
PAIR_WAIT_FRAMES = 45
GAP_FRAMES = 40
DOPPLER_SPACING = 10
SUPER_KILL_HP = 600
KITE_X_MAX = 130
FIRE_CLOSE_X = 32


@dataclass(frozen=True)
class DopplerStrategy:
    """2-2-N missile doppler. Super only when HP ≤ 600."""

    seat: PhantoonStrategy = field(default_factory=PhantoonStrategy)
    flame_eat: bool = True
    fire_close_x: int = FIRE_CLOSE_X
    kite_x_max: int = KITE_X_MAX
    max_fight_frames: int = 20_000
    window_timeout: int = 720
    boss_bit_grace_frames: int = 1_200


@dataclass(frozen=True)
class DopplerWindow:
    """One 2-2-N barrage (or Super finisher)."""

    missiles_spent: int = 0
    super_spent: int = 0
    hp_drop: int = 0
    pair1: int = 0
    pair2: int = 0
    extra: int = 0
    close_eye_extra: int = 0
    halt_miss: bool = False
    pre_hp: int = 0
    post_hp: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(shots=self.missiles_spent + self.super_spent, super_used=self.super_spent > 0,
                       recipe=f"{self.pair1}-{self.pair2}-{self.extra}")
        return payload


@dataclass(frozen=True)
class DopplerEvidence:
    """Measured result of one wiki-doppler fight attempt."""

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
    missiles_spent: int = 0
    super_spent: int = 0
    max_barrage: int = 0
    close_eye_extra: int = 0
    rounds: int = 0
    hp_drop: int = 0
    window_results: tuple[DopplerWindow, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["window_results"] = [w.to_dict() for w in self.window_results]
        payload.update(shots_fired=self.missiles_spent + self.super_spent,
                       windows=self.rounds, super_used=self.super_spent > 0)
        return payload


def missile_spacing_ok(frames_since_fire: int, spacing: int = DOPPLER_SPACING) -> bool:
    """True when the wiki 10f missile tempo has elapsed."""
    return int(frames_since_fire) >= int(spacing)


def should_fire_missile(
    missiles: int, *, hittable: bool, frames_since_fire: int, spacing: int = DOPPLER_SPACING,
) -> bool:
    """Fire a missile only with ammo, an open eye, and 10f spacing."""
    return int(missiles) > 0 and hittable and missile_spacing_ok(frames_since_fire, spacing)


def should_fire_super(hp: int, supers: int, *, kill_hp: int = SUPER_KILL_HP) -> bool:
    """Super only if it would kill (HP ≤ 600). Enrage if he lives."""
    return int(supers) > 0 and 0 < int(hp) <= int(kill_hp)


def barrage_phase(
    pair1: int, pair2: int, extra: int, wait_frames: int, gap_frames: int = 0,
) -> str:
    """2-2-N recipe: pair1 → wait → pair2 → gap → doppler extras."""
    if int(pair1) < PAIR_SIZE:
        return "pair1"
    if int(wait_frames) < PAIR_WAIT_FRAMES:
        return "wait1"
    if int(pair2) < PAIR_SIZE:
        return "pair2"
    if int(gap_frames) < GAP_FRAMES:
        return "gap"
    return "done" if int(extra) >= MAX_DOPPLER_EXTRA else "doppler"


def fight_phantoon_doppler_action(
    state: SuperMetroidState, *, frames_since_fire: int = 99, pair1: int = 0,
    pair2: int = 0, extra: int = 0, wait_frames: int = 0, gap_frames: int = 0,
    hittable: bool = True,
) -> tuple[str, ...]:
    """One-frame fire hint (tests). Play loop owns counted spends."""
    if int(state.enemy0_hp) == 0:
        return ()
    if should_fire_super(int(state.enemy0_hp), int(state.super_missiles)):
        return ("X",) if hittable and missile_spacing_ok(frames_since_fire) else ()
    phase = barrage_phase(pair1, pair2, extra, wait_frames, gap_frames)
    if phase in {"wait1", "gap", "done"}:
        return ()
    fire = should_fire_missile(
        int(state.missiles), hittable=hittable, frames_since_fire=frames_since_fire,
    )
    return ("X",) if fire else ()


def _env_of(session: ControllerSession):
    return getattr(session, "env", None)


def _dead(session: ControllerSession) -> bool:
    st = session.state
    return int(st.health) == 0 or st.phase is GameplayPhase.DEATH_OR_GAME_OVER


def _phantoon_boss_bit(session: ControllerSession) -> bool:
    env = _env_of(session)
    if env is not None:
        try:
            return bool(int(read_bank7e_wram(env)[ADDR_WS_BOSS_BITS]) & PHANTOON_BOSS_BIT)
        except Exception:
            pass
    return boss_defeated_in_state(session.state, phantoon_catalog())


def _body_func(session: ControllerSession) -> int:
    try:
        return int(str(enemy_extra(_env_of(session)).get("func") or "0"), 16)
    except ValueError:
        return 0


def _hittable(session: ControllerSession) -> bool:
    extra = enemy_extra(_env_of(session))
    return bool(extra.get("func_vuln") or extra.get("eye_il_open") or extra.get("body_eye_hit")) or (
        _env_of(session) is None and eye_open(session.state, None)
    )


def _park_ok(session: ControllerSession, park_x: int, strategy: DopplerStrategy) -> bool:
    skip = int(strategy.seat.skip_enemy_x)
    return charge_window_ok(_body_func(session), park_x, session.state.enemy0_y, skip_x=skip)


def _skip_park(session: ControllerSession, park_x: int, strategy: DopplerStrategy) -> bool:
    skip = int(strategy.seat.skip_enemy_x)
    return not _park_ok(session, park_x, strategy) and (
        rain_phase(_body_func(session)) or right_park(park_x, skip_x=skip) or 0 < int(park_x) < 100
    )


def _window_ready(session: ControllerSession, park_x: int, strategy: DopplerStrategy) -> bool:
    st = session.state
    return _hittable(session) and int(st.samus_y) >= int(strategy.seat.floor_y_min) and _park_ok(
        session, park_x, strategy
    )


def _face(state: SuperMetroidState) -> str:
    return "RIGHT" if int(state.enemy0_x) >= int(state.samus_x) else "LEFT"


def _close_enough(state: SuperMetroidState, strategy: DopplerStrategy) -> bool:
    return abs(int(state.samus_x) - int(state.enemy0_x)) <= int(strategy.fire_close_x)


def _fire_names(state: SuperMetroidState) -> list[str]:
    return ["UP", "X"] if int(state.samus_y) > int(state.enemy0_y) + 10 else ["X"]


def _chase_names(state: SuperMetroidState, strategy: DopplerStrategy) -> list[str]:
    """Walk toward Phantoon; never under a rain body or past kite_x_max."""
    x, ex = int(state.samus_x), int(state.enemy0_x)
    dx = ex - x
    names: list[str] = []
    if ex <= 56:
        if x > int(strategy.seat.seat_x_max):
            names.append("LEFT")
        elif int(state.facing) != 8:
            names.append("RIGHT")
    elif x >= int(strategy.kite_x_max) and dx > 0:
        return ["LEFT"]
    elif abs(dx) > int(strategy.fire_close_x):
        names = ["RIGHT" if dx > 0 else "LEFT"]
        if int(state.samus_y) >= int(strategy.seat.floor_y_min) - 4:
            names.append("B")
        return names
    else:
        names.append(_face(state))
    if int(state.samus_y) - int(state.enemy0_y) > 40 and int(state.pose) not in HURT_POSES:
        names.append("A")
    if int(state.samus_y) > int(state.enemy0_y) + 10:
        names.append("UP")
    return names


def _select(session: ControllerSession, weapon: int) -> None:
    if int(session.state.selected_item) != weapon:
        try:
            ensure_weapon(session, weapon)
        except RuntimeError:
            pass


def _flame_eat_tap(session: ControllerSession, strategy: DopplerStrategy) -> None:
    st = session.state
    if not strategy.flame_eat or int(st.selected_item) != WEAPON_BEAM:
        return
    if int(st.pose) in HURT_POSES or is_morph(int(st.pose)):
        return
    hold(session, 2, "X", reason="phan_doppler_flame_eat")
    hold(session, 2, reason="phan_doppler_flame_eat_release")


def _retreat(session: ControllerSession, strategy: DopplerStrategy) -> None:
    if _dead(session) or int(session.state.enemy0_hp) <= 0:
        return
    for _ in range(24):
        if int(session.state.pose) not in HURT_POSES:
            break
        hold(session, 1, reason="phan_doppler_land")
    _go_to_seat(session, strategy.seat)


def _fire_doppler_window(session: ControllerSession, strategy: DopplerStrategy) -> DopplerWindow:
    """One 2-2-N barrage (or Super finisher). Halt when a spend does not chip HP."""
    if is_morph(session.state.pose):
        try:
            unmorph(session)
        except Exception:
            hp = int(session.state.enemy0_hp)
            return DopplerWindow(halt_miss=True, pre_hp=hp, post_hp=hp)

    pre_hp = int(session.state.enemy0_hp)
    want_super = should_fire_super(pre_hp, int(session.state.super_missiles))
    if not want_super:
        _flame_eat_tap(session, strategy)
    _select(session, WEAPON_SUPERS if want_super else WEAPON_MISSILES)

    pair1 = pair2 = extra = wait_frames = gap_frames = 0
    missiles_spent = super_spent = close_eye_extra = 0
    last_spend, last_hp = -99, pre_hp
    pending: int | None = None
    halt_miss = seen_open = False

    for _ in range(strategy.window_timeout):
        st = session.state
        if _dead(session) or int(st.enemy0_hp) == 0 or int(st.health) <= 20:
            break
        hittable = _hittable(session)
        seen_open = seen_open or hittable
        if pending is not None:
            if int(st.enemy0_hp) < last_hp:
                pending, last_hp = None, int(st.enemy0_hp)
            elif session.frame - pending >= 48:
                halt_miss = True
                break
        spent = missiles_spent + super_spent
        if (not hittable) and seen_open and spent >= 1 and pending is None:
            break
        if rain_phase(_body_func(session)) and not _park_ok(session, int(st.enemy0_x), strategy):
            break
        if int(st.pose) in HURT_POSES:
            hold(session, 1, reason="phan_doppler_hurt")
            continue

        use_super = want_super or should_fire_super(int(st.enemy0_hp), int(st.super_missiles))
        phase = "super" if use_super else barrage_phase(pair1, pair2, extra, wait_frames, gap_frames)
        if phase == "done" or (phase == "super" and super_spent >= 1 and pending is None):
            break
        if phase in {"wait1", "gap"}:
            wait_frames += phase == "wait1"
            gap_frames += phase == "gap"
            hold(session, 1, reason="phan_doppler_wait" if phase == "wait1" else "phan_doppler_gap")
            continue
        if phase == "super":
            want_super = True
            _select(session, WEAPON_SUPERS)
            ammo, attr, reason = int(st.super_missiles), "super_missiles", "phan_doppler_super"
            fire = hittable and ammo > 0 and missile_spacing_ok(session.frame - last_spend) and _close_enough(st, strategy)
        else:
            ammo, attr, reason = int(st.missiles), "missiles", "phan_doppler_shot"
            fire = should_fire_missile(
                ammo, hittable=hittable, frames_since_fire=session.frame - last_spend,
            ) and _close_enough(st, strategy)
        names = _fire_names(st) if fire else (_chase_names(st, strategy) or [_face(st)])
        hold(session, 1, *tuple(dict.fromkeys(names)), reason=reason)
        if int(getattr(session.state, attr)) < ammo:
            last_spend = pending = session.frame
            if phase == "super":
                super_spent += 1
            elif phase == "pair1":
                pair1 += 1
                missiles_spent += 1
            elif phase == "pair2":
                pair2 += 1
                missiles_spent += 1
            else:
                extra += 1
                missiles_spent += 1
                close_eye_extra += extra > 3

    post_hp = int(session.state.enemy0_hp)
    hp_drop = max(0, pre_hp - post_hp)
    if missiles_spent + super_spent >= 1 and hp_drop <= 0:
        halt_miss = True
    _retreat(session, strategy)
    return DopplerWindow(
        missiles_spent=missiles_spent, super_spent=super_spent, hp_drop=hp_drop,
        pair1=pair1, pair2=pair2, extra=extra, close_eye_extra=close_eye_extra,
        halt_miss=halt_miss, pre_hp=pre_hp, post_hp=post_hp,
    )


def _wait_tick(session: ControllerSession, park_x: int, strategy: DopplerStrategy) -> None:
    """Rain snipe, or pre-select Super when HP ≤ 600 so SELECT does not eat the open."""
    st = session.state
    if should_fire_super(int(st.enemy0_hp), int(st.super_missiles)):
        _select(session, WEAPON_SUPERS)
        hold(session, 1, "DOWN", reason="phan_doppler_wait_super")
        return
    if _skip_park(session, park_x, strategy):
        _rain_corner_wait(session, strategy.seat)
        return
    if not seated(st, strategy.seat):
        _go_to_seat(session, strategy.seat)
        return
    if is_morph(int(st.pose)):
        try:
            unmorph(session)
        except Exception:
            hold(session, 1, reason="phan_doppler_unmorph")
        return
    _select(session, WEAPON_MISSILES)
    hold(session, 1, "DOWN", reason="phan_doppler_wait_eye")


def wait_doppler_window(
    session: ControllerSession, strategy: DopplerStrategy, *, timeout: int = 2400,
) -> bool:
    """Left-corner wait for ``charge_window_ok`` open. Rain uses product snipe."""
    _go_to_seat(session, strategy.seat)
    _select(session, WEAPON_MISSILES)
    park_x = int(session.state.enemy0_x)
    last_func: int | None = None
    for _ in range(timeout):
        st = session.state
        if _dead(session) or int(st.enemy0_hp) == 0:
            return False
        func_now = _body_func(session)
        if func_now != last_func:
            park_x, last_func = int(st.enemy0_x), func_now
        if _window_ready(session, park_x, strategy):
            return True
        _wait_tick(session, park_x, strategy)
    return _window_ready(session, park_x, strategy)


def _require_room(session: ControllerSession) -> None:
    if int(session.state.room_id) != ROOM_PHANTOON:
        raise RuntimeError(
            f"Phantoon doppler expected room 0x{ROOM_PHANTOON:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )


def play_phantoon_doppler_window(
    session: ControllerSession, *, strategy: DopplerStrategy | None = None, wait: int = 2400,
) -> DopplerWindow:
    """Seat + one barrage. Halt at first miss (spend without HP chip)."""
    strategy = strategy or DopplerStrategy()
    _require_room(session)
    if not wait_doppler_window(session, strategy, timeout=wait):
        hp = int(session.state.enemy0_hp)
        return DopplerWindow(halt_miss=True, pre_hp=hp, post_hp=hp)
    return _fire_doppler_window(session, strategy)


def play_phantoon_doppler_fight(
    session: ControllerSession, *, strategy: DopplerStrategy | None = None, require_boss_bit: bool = True,
) -> DopplerEvidence:
    """Fight Phantoon from room ``0xCD13`` with wiki 2-2-N doppler."""
    strategy = strategy or DopplerStrategy()
    start = session.frame
    _require_room(session)
    _select(session, WEAPON_MISSILES)
    _go_to_seat(session, strategy.seat)

    peak_hp = min_hp = start_hp = prev_hp = int(session.state.enemy0_hp)
    body_zero_frame: int | None = start if peak_hp == 0 else None
    boss_bit_frame: int | None = None
    missiles = supers = extra = max_barrage = 0
    windows: list[DopplerWindow] = []
    park_x = int(session.state.enemy0_x)
    last_func: int | None = None
    halt = False

    for _ in range(strategy.max_fight_frames):
        state = session.state
        if int(state.room_id) != ROOM_PHANTOON or _dead(session):
            break
        hp = int(state.enemy0_hp)
        peak_hp, min_hp = max(peak_hp, hp), min(min_hp, hp)
        if body_zero_frame is None and hp == 0 and prev_hp > 0:
            body_zero_frame, min_hp = session.frame, 0
        prev_hp = hp
        if body_zero_frame is not None:
            if _phantoon_boss_bit(session):
                boss_bit_frame = session.frame
                break
            if not require_boss_bit or session.frame - body_zero_frame >= strategy.boss_bit_grace_frames:
                break
            hold(session, 1, reason="phantoon_death_anim")
            continue
        func_now = _body_func(session)
        if func_now != last_func:
            park_x, last_func = int(state.enemy0_x), func_now
        if not _window_ready(session, park_x, strategy):
            _wait_tick(session, park_x, strategy)
            continue
        got = _fire_doppler_window(session, strategy)
        windows.append(got)
        missiles += got.missiles_spent
        supers += got.super_spent
        extra += got.close_eye_extra
        max_barrage = max(max_barrage, got.missiles_spent + got.super_spent)
        if got.halt_miss and got.hp_drop <= 0 and len(windows) == 1:
            halt = True
            break
        if got.missiles_spent + got.super_spent <= 0:
            hold(session, 1, reason="phan_doppler_wait_eye")

    boss_set = _phantoon_boss_bit(session)
    outcome = (
        "died" if _dead(session) else "halt_miss" if halt else
        "phantoon_defeated" if boss_set else
        "phantoon_body_zero_no_boss_bit" if body_zero_frame is not None else "timeout"
    )
    return DopplerEvidence(
        start_frame=start, body_zero_frame=body_zero_frame, boss_bit_frame=boss_bit_frame,
        end_frame=session.frame, peak_body_hp=peak_hp, min_body_hp=min_hp,
        action_frames=session.frame - start, final_body_hp=int(session.state.enemy0_hp),
        boss_bit_set=boss_set, outcome=outcome, missiles_spent=missiles,
        super_spent=supers, max_barrage=max_barrage, close_eye_extra=extra,
        rounds=len(windows), hp_drop=max(0, start_hp - int(session.state.enemy0_hp)),
        window_results=tuple(windows),
    )
