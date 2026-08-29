"""Phantoon seat/window helpers.

Product spine fight is ``phantoon_doppler.play_phantoon_doppler_fight``.
This file is shared seat + open-eye geometry used by that recipe.

Public policy (https://wiki.supermetroid.run/Phantoon): eye opens after
1 / 6 / 11 s (fast/mid/slow), left or right. Round-1 has 6 positions;
later rounds another 6.

Traps:
- Super Missile enrages Phantoon (flame-wave counter). Super-spray is a
  scaffold, not a hit.
- Ice is equipped (0x1007). Ice charge can change hide behavior. Measure.
- Invisible / closed eye = no damage. Stock RAM may lack a dedicated eye
  flag — log enemy0 x/y/hp/spritemap every distinct map.
- Flame droplets: do not stand under the body.
- Jump into a wall can climb out (Spore lesson).

HP 2500. Missile 100; Super 600 but enrages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from super_metroid.combat.primitives import ensure_weapon, settle_standing
from super_metroid.ram import GameplayPhase, SuperMetroidState
from super_metroid.routes.controller_common import is_morph, unmorph
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_PHANTOON = 0xCD13
WEAPON_BEAM = 0
WEAPON_MISSILES = 1
WEAPON_SUPERS = 2
# Wrecked Ship boss bits ($7E:D82B); bit 0 = Phantoon. Low-WRAM
# env.get_ram() is 8 KiB and never contains this byte.
ADDR_WS_BOSS_BITS = 0xD82B
PHANTOON_BOSS_BIT = 0x01

# Left-corner floor seat (door-enter pin lands ~(39,124) p81 and falls in).
SEAT_X = 32
SEAT_X_MIN = 16
SEAT_X_MAX = 56
FLOOR_Y_MIN = 160
FLOOR_Y_MAX = 210

# Measured open-eye body maps (dump v3). 0xDEDD is intro/figure-8 closed.
VULNERABLE_SPRITEMAPS: frozenset[int] = frozenset({0xDEF1, 0xDEE7})

# Beam charge counter ($7E:0CD0). Full charge is 60; glow continues to 120.
ADDR_BEAM_CHARGE = 0x0CD0
CHARGE_FULL = 60

ADDR_ENEMY0_ID = 0x0F78
ADDR_ENEMY0_ILIST = 0x0F92
ADDR_ENEMY0_ITIMER = 0x0F94
ADDR_ENEMY0_PALETTE = 0x0F96
ADDR_ENEMY0_AI0 = 0x0FA8
ADDR_ENEMY0_AI1 = 0x0FAA
ADDR_ENEMY0_AI2 = 0x0FAC
ADDR_ENEMY0_FUNC = 0x0FB2
ENEMY_SLOT = 0x40
# Eye / tentacles / mouth are slots 1–3 (PJBoy bank $A7).
ADDR_EYE_ID = ADDR_ENEMY0_ID + ENEMY_SLOT
ADDR_EYE_X = 0x0F7A + ENEMY_SLOT
ADDR_EYE_Y = 0x0F7E + ENEMY_SLOT
ADDR_EYE_HP = 0x0F8C + ENEMY_SLOT
ADDR_EYE_SPRITEMAP = 0x0F8E + ENEMY_SLOT
ADDR_EYE_ILIST = 0x0F92 + ENEMY_SLOT

# Body IL ranges (current pointer walks through the list).
IL_BODY_INVULN = (0xCC41, 0xCC47)
IL_BODY_FULL = (0xCC47, 0xCC4D)
IL_BODY_EYE_HIT = (0xCC4D, 0xCC53)
# Eye slot IL: open anim + looking-around eyeball = damage window.
IL_EYE_OPEN = (0xCC53, 0xCC7B)
IL_EYE_LOOK = (0xCC9D, 0xCCD7)

# Body function ($0FB2) — figure-8 / flame-rain vulnerable windows.
FUNC_FIG8_VULN = 0xD60D
FUNC_FIG8_SWOOP_TRIG = 0xD65C
FUNC_SWOOP_OPAQUE = 0xD678
FUNC_RAIN_MAKE_VULN = 0xD767
FUNC_RAIN_VULN = 0xD788
VULNERABLE_FUNCS = frozenset(
    {
        FUNC_FIG8_VULN,
        FUNC_FIG8_SWOOP_TRIG,
        FUNC_SWOOP_OPAQUE,
        FUNC_RAIN_MAKE_VULN,
        FUNC_RAIN_VULN,
    }
)
RAIN_VULN_FUNCS = frozenset({FUNC_RAIN_MAKE_VULN, FUNC_RAIN_VULN})
# Flame-rain cycle after a fig-8 (skip-rain dump). Standing through D82A died.
FUNC_RAIN_START = 0xD82A
FUNC_RAIN_MOVE = 0xD73F
FUNC_RAIN_HIDE = 0xD7D5
FUNC_RAIN_NEXT = 0xD7F7
RAIN_PHASE_FUNCS = RAIN_VULN_FUNCS | frozenset(
    {FUNC_RAIN_START, FUNC_RAIN_MOVE, FUNC_RAIN_HIDE, FUNC_RAIN_NEXT}
)

# Knockback / hit-stun only. Pose 81 is ordinary falling — keep jump held.
HURT_POSES = frozenset({83, 84, 109, 143, 158, 159, 160})


@dataclass(frozen=True)
class PhantoonStrategy:
    """Left-corner seat and open-eye geometry. Seat fields are the living ones."""

    seat_x: int = SEAT_X
    seat_x_min: int = SEAT_X_MIN
    seat_x_max: int = SEAT_X_MAX
    floor_y_min: int = FLOOR_Y_MIN
    floor_y_max: int = FLOOR_Y_MAX
    weapon: int = WEAPON_MISSILES
    shots_per_window: int = 2
    min_shots_to_fire: int = 1
    charge_full: int = CHARGE_FULL
    fire_standoff: int = 48
    fire_close_x: int = 16
    kite_x_max: int = 130
    skip_enemy_x: int = 155
    right_seat_x_min: int = 180
    right_seat_x: int = 220
    # Room wall is x≈219 — x≥230 is not reachable. Jump from the wall.
    right_jump_x: int = 216
    jump_hold_frames: int = 36
    charge_hold_frames: int = 70
    fire_release_frames: int = 2
    height_slop: int = 80
    # W1 charge chip was dy=41 (y=149 vs eye 108). W2 miss was dy=78 (y=174
    # vs eye 96) — still the floor hop. Release only in this band.
    release_dy_min: int = 28
    release_dy_max: int = 56
    max_fight_frames: int = 20_000
    boss_bit_grace_frames: int = 1_200
    window_timeout: int = 480


def _ram(env: Any) -> Any:
    if env is None:
        return None
    try:
        return env.get_ram()
    except Exception:
        return None


def _u16(ram: Any, address: int) -> int:
    if ram is None or len(ram) < address + 2:
        return 0
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def _in_il(value: int, span: tuple[int, int]) -> bool:
    return span[0] <= value < span[1]


def enemy_extra(env: Any) -> dict[str, object]:
    """Phantoon body + eye-slot words (instruction list / function / AI)."""
    ram = _ram(env)
    func = _u16(ram, ADDR_ENEMY0_FUNC)
    body_il = _u16(ram, ADDR_ENEMY0_ILIST)
    eye_il = _u16(ram, ADDR_EYE_ILIST)
    return {
        "id": f"0x{_u16(ram, ADDR_ENEMY0_ID):04X}",
        "ilist": f"0x{body_il:04X}",
        "timer": _u16(ram, ADDR_ENEMY0_ITIMER),
        "palette": f"0x{_u16(ram, ADDR_ENEMY0_PALETTE):04X}",
        "ai0": f"0x{_u16(ram, ADDR_ENEMY0_AI0):04X}",
        "ai1": f"0x{_u16(ram, ADDR_ENEMY0_AI1):04X}",
        "ai2": f"0x{_u16(ram, ADDR_ENEMY0_AI2):04X}",
        "func": f"0x{func:04X}",
        "charge": _u16(ram, ADDR_BEAM_CHARGE),
        "eye_id": f"0x{_u16(ram, ADDR_EYE_ID):04X}",
        "eye_x": _u16(ram, ADDR_EYE_X),
        "eye_y": _u16(ram, ADDR_EYE_Y),
        "eye_hp": _u16(ram, ADDR_EYE_HP),
        "eye_spritemap": f"0x{_u16(ram, ADDR_EYE_SPRITEMAP):04X}",
        "eye_ilist": f"0x{eye_il:04X}",
        "func_vuln": func in VULNERABLE_FUNCS,
        "eye_il_open": _in_il(eye_il, IL_EYE_OPEN) or _in_il(eye_il, IL_EYE_LOOK),
        "body_eye_hit": _in_il(body_il, IL_BODY_EYE_HIT),
    }


def eye_open(state: SuperMetroidState, env: Any = None) -> bool:
    """True during a measured open-eye damage window.

    Stock enemy0 spritemap is the *body* (often 0xDEDD through intro). The
    eye is slot 1; function $0FB2 and the eye instruction list are the
    live flags. ``env`` is required for the live check; without it, fall
    back to the measured body spritemap set.
    """
    if env is not None:
        extra = enemy_extra(env)
        if extra["func_vuln"] or extra["eye_il_open"] or extra["body_eye_hit"]:
            return True
        return False
    return int(state.enemy0_spritemap) in VULNERABLE_SPRITEMAPS


def seated(state: SuperMetroidState, strategy: PhantoonStrategy | None = None) -> bool:
    """Standing/crouching in the left-corner floor seat (not morph, not air)."""
    strat = strategy or PhantoonStrategy()
    if is_morph(int(state.pose)):
        return False
    if int(state.pose) in (81, 82, 164):
        return False
    return (
        strat.seat_x_min <= int(state.samus_x) <= strat.seat_x_max
        and strat.floor_y_min <= int(state.samus_y) <= strat.floor_y_max
    )


def eye_ilist_open(ilist: int) -> bool:
    """True when the eye-slot instruction list is the open/look window."""
    return _in_il(int(ilist), IL_EYE_OPEN) or _in_il(int(ilist), IL_EYE_LOOK)


def func_vulnerable(func: int) -> bool:
    """True when body function $0FB2 is a damage window."""
    return int(func) in VULNERABLE_FUNCS


def rain_vulnerable(func: int) -> bool:
    """True on flame-rain vuln ($D767 / $D788). Do not jump under the body."""
    return int(func) in RAIN_VULN_FUNCS


def rain_phase(func: int) -> bool:
    """True on the flame-rain cycle (D82A…). Morph in a corner; do not stand."""
    return int(func) in RAIN_PHASE_FUNCS


# Only rain (48, 96). (56, 113) jumped p83; (88, 64) crossed the body.
RAIN_FIRE_X_MAX = 56
RAIN_FIRE_Y_MIN = 88
RAIN_FIRE_Y_MAX = 104


def rain_charge_ok(
    enemy_x: int,
    enemy_y: int = 96,
    *,
    max_x: int = RAIN_FIRE_X_MAX,
) -> bool:
    """True only for rain park ≈(48, 96). Skip (56, 113) and (53, 82)."""
    return (
        0 < int(enemy_x) <= max_x
        and RAIN_FIRE_Y_MIN <= int(enemy_y) <= RAIN_FIRE_Y_MAX
    )


def charge_window_ok(
    func: int,
    enemy_x: int,
    enemy_y: int = 96,
    *,
    skip_x: int = 155,
) -> bool:
    """W1 left fig-8 (~120) or rain (48, 96). Skip (53, 82) and x=219."""
    if right_park(enemy_x, skip_x=skip_x):
        return False
    if rain_phase(func):
        return rain_vulnerable(func) and rain_charge_ok(enemy_x, enemy_y)
    # W1 fig-8 only (~120). Skip (53, 82) and (83, 64) from the left seat.
    return 100 <= int(enemy_x) < skip_x


def right_park(enemy_x: int, *, skip_x: int = 155) -> bool:
    """True when the eye is on the right half (measured W2 fig-8 (203, 83))."""
    return int(enemy_x) >= skip_x


def _dead(session: ControllerSession) -> bool:
    st = session.state
    return int(st.health) == 0 or st.phase is GameplayPhase.DEATH_OR_GAME_OVER


def _face_right(session: ControllerSession) -> None:
    if session.state.facing == 8:
        return
    hold(session, 1, "RIGHT", reason="phan_face")


def _flame_snipe_tap(session: ControllerSession, strategy: PhantoonStrategy) -> None:
    """Uncharged UP taps from the living seat (the policy that lived at 99)."""
    st = session.state
    if is_morph(int(st.pose)):
        try:
            unmorph(session)
        except Exception:
            hold(session, 1, reason="phan_farm_idle")
        return
    if int(st.selected_item) != WEAPON_BEAM:
        try:
            ensure_weapon(session, WEAPON_BEAM)
        except RuntimeError:
            hold(session, 1, reason="phan_farm_idle")
        return
    if int(st.samus_y) < strategy.floor_y_min:
        hold(session, 1, reason="phan_fall_in")
        return
    if int(st.pose) in HURT_POSES:
        hold(session, 1, reason="phan_hurt")
        return
    # Never A on skip — leftover jump poses idle until they land.
    if int(st.pose) in (21, 22, 25, 43, 44, 81, 82):
        hold(session, 1, reason="phan_farm_land")
        return
    if int(st.samus_x) > strategy.seat_x_max:
        hold(session, 1, "LEFT", reason="phan_farm_left")
        return
    names = ["RIGHT"] if int(st.facing) != 8 else ["UP"]
    if session.frame % 8 < 2:
        names.append("X")
    hold(session, 1, *names, reason="phan_farm_snipe")


def _rain_corner_wait(session: ControllerSession, strategy: PhantoonStrategy) -> None:
    """Skip-right / rain: pose-3 UP+tap X (living seat). Do not sit-charge."""
    _flame_snipe_tap(session, strategy)


def _go_to_seat(session: ControllerSession, strategy: PhantoonStrategy) -> None:
    """Land from the door-fall pin and park in the left-corner floor seat."""
    if session.state.samus_y < strategy.floor_y_min:
        for _ in range(180):
            if session.state.samus_y >= strategy.floor_y_min or _dead(session):
                break
            hold(session, 1, reason="phan_fall_in")
    settle_standing(
        session,
        min_y=strategy.floor_y_min,
        max_frames=80,
        reason="phan_land",
    )
    if is_morph(session.state.pose):
        try:
            unmorph(session)
        except Exception:
            return
    if seated(session.state, strategy) or _dead(session) or session.state.enemy0_hp == 0:
        if seated(session.state, strategy):
            _face_right(session)
            try:
                ensure_weapon(session, strategy.weapon)
            except RuntimeError:
                pass
        return

    for _ in range(90):
        st = session.state
        if _dead(session) or seated(st, strategy) or st.enemy0_hp == 0:
            break
        if st.samus_y < strategy.floor_y_min:
            hold(session, 1, reason="phan_fall_in")
            continue
        if is_morph(st.pose):
            try:
                unmorph(session)
            except Exception:
                return
            continue
        if st.samus_x < strategy.seat_x_min:
            hold(session, 1, "RIGHT", reason="phan_seat_right")
        elif st.samus_x > strategy.seat_x_max:
            hold(session, 1, "LEFT", reason="phan_seat_left")
        else:
            hold(session, 1, reason="phan_seat_idle")
    if not _dead(session) and session.state.enemy0_hp > 0:
        _face_right(session)
        try:
            ensure_weapon(session, strategy.weapon)
        except RuntimeError:
            pass


__all__ = [
    "CHARGE_FULL",
    "FLOOR_Y_MAX",
    "FLOOR_Y_MIN",
    "ADDR_WS_BOSS_BITS",
    "PHANTOON_BOSS_BIT",
    "ROOM_PHANTOON",
    "SEAT_X",
    "SEAT_X_MAX",
    "SEAT_X_MIN",
    "VULNERABLE_SPRITEMAPS",
    "WEAPON_BEAM",
    "WEAPON_MISSILES",
    "WEAPON_SUPERS",
    "PhantoonStrategy",
    "charge_window_ok",
    "enemy_extra",
    "eye_ilist_open",
    "eye_open",
    "func_vulnerable",
    "rain_charge_ok",
    "rain_phase",
    "rain_vulnerable",
    "right_park",
    "seated",
]
