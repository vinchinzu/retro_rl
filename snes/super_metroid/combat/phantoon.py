"""No-assist Phantoon policy: left-corner seat + open-eye charge/missiles.

Public policy (https://wiki.supermetroid.run/Phantoon,
https://wiki.supermetroid.run/Phantoon#Phantoon_First):

KPDR Room Strategies: eye opens after 1 / 6 / 11 s (fast/mid/slow), left or
right. Round-1 has 6 positions; later rounds another 6.

Passable KPDR (Charge+Wave+Spazer+Ice): beginner 4-round — charge shot when
the eye opens, then two more, he disappears; repeat. Charge+Missiles variant:
2 missiles + 2 missiles + charge × 4 rounds. Do not invent X-factor / 2-round
TAS unless a measured window makes it free.

Traps:
- Super Missile enrages Phantoon (flame-wave counter). Super-spray is a
  scaffold, not a hit. Pin boots with supers selected — select beam or
  missiles before the first open.
- Ice is equipped (0x1007). Ice charge can change hide behavior. Measure.
- Invisible / closed eye = no damage. Stock RAM may lack a dedicated eye
  flag — log enemy0 x/y/hp/spritemap every distinct map.
- Flame droplets: shoot for drops when farming; do not stand under the body.
- ``select_weapon`` at 0 ammo raises (game forces beam). Farm first.
- Jump into a wall can climb out (Spore lesson).

HP 2500. Charged Spazer/Wave/Ice ~300; missile 100; Super 600 but enrages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from super_metroid.combat.features import (
    boss_defeated_in_state,
    features_from_state,
    phantoon_catalog,
)
from super_metroid.combat.primitives import ensure_weapon, settle_standing
from super_metroid.combat.spore_spawn import Pickup, list_pickups
from super_metroid.ram import GameplayPhase, SuperMetroidState
from super_metroid.routes.controller_common import is_morph, unmorph
from super_metroid.routes.runtime import ControllerSession, hold

ROOM_PHANTOON = 0xCD13
WEAPON_BEAM = 0
WEAPON_MISSILES = 1
WEAPON_SUPERS = 2
PHANTOON_INVISIBLE = "invisible"
PHANTOON_VULNERABLE = "vulnerable"
PHANTOON_DEFEATED = "defeated"

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
# Standing / crouch / turn — W2 rain floor release. Not jump (21/25) or hurt.
FLOOR_RELEASE_POSES = frozenset({1, 2, 3, 4, 5, 11})

# Knockback / hit-stun only. Pose 81 is ordinary falling — keep jump held.
HURT_POSES = frozenset({83, 84, 109, 143, 158, 159, 160})


@dataclass(frozen=True)
class PhantoonStrategy:
    """Left-corner seat + open-eye charge (optional missiles). Never Super."""

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


@dataclass(frozen=True)
class PhantoonEvidence:
    """Measured result of one no-assist Phantoon fight attempt."""

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
    phase_transitions: tuple[tuple[str, int], ...] = ()
    shots_fired: int = 0
    windows: int = 0
    vulnerable_spritemaps: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "body_zero_frame": self.body_zero_frame,
            "boss_bit_frame": self.boss_bit_frame,
            "end_frame": self.end_frame,
            "peak_body_hp": self.peak_body_hp,
            "min_body_hp": self.min_body_hp,
            "action_frames": self.action_frames,
            "final_body_hp": self.final_body_hp,
            "boss_bit_set": self.boss_bit_set,
            "outcome": self.outcome,
            "phase_transitions": [
                {"phase": phase, "frame": frame}
                for phase, frame in self.phase_transitions
            ],
            "shots_fired": self.shots_fired,
            "windows": self.windows,
            "vulnerable_spritemaps": list(self.vulnerable_spritemaps),
        }


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


def beam_charge(env: Any) -> int:
    """Live beam-charge counter ($0CD0). 0 if RAM is missing."""
    return _u16(_ram(env), ADDR_BEAM_CHARGE)


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


def phantoon_phase(state: SuperMetroidState) -> str:
    """Classify observable phase: open-eye vulnerable vs closed/invisible."""
    if state.enemy0_hp == 0:
        return PHANTOON_DEFEATED
    if eye_open(state):
        return PHANTOON_VULNERABLE
    features = features_from_state(state, phantoon_catalog())
    if not features.enemy_active:
        return PHANTOON_INVISIBLE
    return PHANTOON_INVISIBLE


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


# Only (48, 96) from the living seat. (88, 64) crossed the body (p83 at 101,127).
RAIN_FIRE_X_MAX = 64


def rain_charge_ok(enemy_x: int, *, max_x: int = RAIN_FIRE_X_MAX) -> bool:
    """True for left-ish rain parks. Skip (128, 96) and anything right."""
    return 0 < int(enemy_x) <= max_x


def charge_window_ok(
    func: int, enemy_x: int, *, skip_x: int = 155
) -> bool:
    """Left fig-8, or left-ish rain vuln. Skip right wall and (128, 96)."""
    if right_park(enemy_x, skip_x=skip_x):
        return False
    if rain_phase(func):
        return rain_vulnerable(func) and rain_charge_ok(enemy_x)
    return True


def right_park(enemy_x: int, *, skip_x: int = 155) -> bool:
    """True when the eye is on the right half (measured W2 fig-8 (203, 83))."""
    return int(enemy_x) >= skip_x


def rain_corner_morph(
    state: SuperMetroidState, strategy: PhantoonStrategy | None = None
) -> bool:
    """Morph in a bottom corner — rain seat, not standing, not under the body."""
    strat = strategy or PhantoonStrategy()
    if not is_morph(int(state.pose)):
        return False
    if not (strat.floor_y_min <= int(state.samus_y) <= strat.floor_y_max):
        return False
    x = int(state.samus_x)
    return x <= strat.seat_x_max or x >= strat.right_seat_x_min


def floor_release_ok(
    state: SuperMetroidState, strategy: PhantoonStrategy | None = None
) -> bool:
    """Standing/crouch on the floor — rain charge release, no jump."""
    strat = strategy or PhantoonStrategy()
    return (
        int(state.pose) in FLOOR_RELEASE_POSES
        and int(state.samus_y) >= strat.floor_y_min - 6
    )


def fight_phantoon_action(
    state: SuperMetroidState,
    frame_index: int,
    strategy: PhantoonStrategy = PhantoonStrategy(),
) -> tuple[str, ...]:
    """One-frame seat / charge / fire hint (tests + AP). Play loop owns shots."""
    del frame_index
    if state.enemy0_hp == 0:
        return ()
    if seated(state, strategy):
        if not eye_open(state):
            return ("X",) if strategy.weapon == WEAPON_BEAM else ()
        if int(state.enemy0_x) >= strategy.skip_enemy_x:
            return ()
        if strategy.weapon == WEAPON_MISSILES and state.missiles <= 0:
            return ()
        if strategy.weapon == WEAPON_MISSILES:
            return ("X",)
        return ()
    if int(state.samus_y) < strategy.floor_y_min:
        return ()
    if int(state.samus_x) > strategy.seat_x_max:
        return ("LEFT",)
    if int(state.samus_x) < strategy.seat_x_min:
        return ("RIGHT",)
    return ()


def _dead(session: ControllerSession) -> bool:
    st = session.state
    return int(st.health) == 0 or st.phase is GameplayPhase.DEATH_OR_GAME_OVER


def _face_right(session: ControllerSession) -> None:
    if session.state.facing == 8:
        return
    hold(session, 1, "RIGHT", reason="phan_face")


def _right_seat_names(state: SuperMetroidState, strategy: PhantoonStrategy) -> list[str]:
    """Floor RIGHT+B, keep charge. Do not jump under the body."""
    names: list[str] = ["RIGHT"]
    if int(state.samus_y) >= strategy.floor_y_min - 4:
        names.append("B")
    if strategy.weapon == WEAPON_BEAM:
        names.append("X")
    return names


def _go_to_right_seat(session: ControllerSession, strategy: PhantoonStrategy) -> None:
    """Floor dash to the right seat (x≥180). No jump under the body."""
    if is_morph(session.state.pose):
        try:
            unmorph(session)
        except Exception:
            return
    for _ in range(180):
        st = session.state
        if _dead(session) or st.enemy0_hp == 0:
            return
        if (
            int(st.samus_x) >= strategy.right_seat_x
            and int(st.samus_y) >= strategy.floor_y_min - 4
            and st.pose not in (81, 82, 164)
        ):
            return
        if int(st.samus_y) < strategy.floor_y_min:
            hold(session, 1, reason="phan_fall_in")
            continue
        hold(session, 1, *_right_seat_names(st, strategy), reason="phan_right_seat")


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
    if int(st.samus_x) > strategy.seat_x_max:
        hold(session, 1, "LEFT", reason="phan_farm_left")
        return
    names = ["RIGHT"] if int(st.facing) != 8 else ["UP"]
    if session.frame % 8 < 2:
        names.append("X")
    hold(session, 1, *names, reason="phan_farm_snipe")


def _rain_snipe(session: ControllerSession, strategy: PhantoonStrategy) -> None:
    """Stand the living left seat and hold charge. Do not morph-tank or chase."""
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
    if int(st.samus_x) > strategy.seat_x_max:
        hold(session, 1, "LEFT", "X", reason="phan_farm_left")
        return
    hold(session, 1, "X", reason="phan_farm_charge")


def _rain_corner_wait(session: ControllerSession, strategy: PhantoonStrategy) -> None:
    """Skip-right / rain: beam-snipe flames from the left. Do not morph-tank."""
    _rain_snipe(session, strategy)


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


def _keep_seat(session: ControllerSession, strategy: PhantoonStrategy) -> None:
    if seated(session.state, strategy):
        return
    _go_to_seat(session, strategy)


def _env_of(session: ControllerSession) -> Any:
    return getattr(session, "env", None)


def _session_eye_open(session: ControllerSession) -> bool:
    return eye_open(session.state, _env_of(session))


def _charged(session: ControllerSession, strategy: PhantoonStrategy) -> bool:
    return beam_charge(_env_of(session)) >= strategy.charge_full


def _hold_charge(session: ControllerSession, frames: int = 1) -> None:
    hold(session, frames, "X", reason="phan_charge")


def _hittable(session: ControllerSession) -> bool:
    extra = enemy_extra(_env_of(session))
    return bool(extra.get("func_vuln") or extra.get("eye_il_open") or extra.get("body_eye_hit"))


def _aim_names(
    state: SuperMetroidState,
    strategy: PhantoonStrategy,
    *,
    rain: bool = False,
) -> list[str]:
    names: list[str] = []
    dx = int(state.enemy0_x) - int(state.samus_x)
    close = abs(dx) <= strategy.fire_close_x
    if rain:
        # Stay left of the body. Dash-to-|dx|≤16 walks under (128,96) and
        # a floor UP from (116,187) dumped charge with no HP chip.
        if int(state.samus_x) > strategy.seat_x_max:
            names.append("LEFT")
            return names
        if int(state.facing) != 8:
            names.append("RIGHT")
            return names
        names.append("UP")
        return names
    # Right fig-8: stay on the floor until the right seat (x≥180). No jump
    # under the body while crossing from the left. Keep charge (X).
    # Once there, face LEFT then jump in place — LEFT+A drifts under the body.
    if right_park(state.enemy0_x, skip_x=strategy.skip_enemy_x):
        if int(state.samus_x) < strategy.right_jump_x:
            return _right_seat_names(state, strategy)
        if int(state.facing) != 4:
            return ["LEFT"]
        if int(state.samus_x) >= 22 and _need_height(state, strategy):
            names.append("A")
        if int(state.samus_y) > int(state.enemy0_y) + 10:
            names.append("UP")
        return names
    # (48, 96) sits on the living seat — jump in place, do not walk into it.
    if rain_charge_ok(state.enemy0_x) and int(state.enemy0_x) <= strategy.seat_x_max + 16:
        if close and _need_height(state, strategy):
            names.append("A")
        if close and int(state.samus_y) > int(state.enemy0_y) + 10:
            names.append("UP")
        return names
    if (
        (not right_park(state.enemy0_x, skip_x=strategy.skip_enemy_x))
        and int(state.samus_x) >= strategy.kite_x_max
    ):
        names.append("LEFT")
    elif state.samus_x < 22:
        names.append("RIGHT")
    elif dx > 10:
        names.append("RIGHT")
    elif dx < -10:
        names.append("LEFT")
    on_floor = int(state.samus_y) >= strategy.floor_y_min - 4
    if (not close) and on_floor and names:
        names.append("B")
    # Rain: never A. Fig-8: jump only once close — A on the approach dumps p77.
    if (
        (not rain)
        and close
        and int(state.samus_x) >= 22
        and _need_height(state, strategy)
    ):
        names.append("A")
    if close and int(state.samus_y) > int(state.enemy0_y) + 10:
        names.append("UP")
    return names


def _need_height(state: SuperMetroidState, strategy: PhantoonStrategy) -> bool:
    """True when Samus is below the charge-release band (needs more jump)."""
    dy = int(state.samus_y) - int(state.enemy0_y)
    return dy > strategy.release_dy_max


def in_release_band(
    state: SuperMetroidState, strategy: PhantoonStrategy | None = None
) -> bool:
    """True when Samus is the measured W1 charge-release height below the eye.

    W1 chip: samus (104, 149) vs eye (120, 108) → dy=41.
    W2 miss at y=148 vs eye 83 → dy=65 (outside 28–56). Target y=111–139.
    """
    strat = strategy or PhantoonStrategy()
    dy = int(state.samus_y) - int(state.enemy0_y)
    return strat.release_dy_min <= dy <= strat.release_dy_max


def _height_ok(state: SuperMetroidState, strategy: PhantoonStrategy) -> bool:
    return int(state.samus_y) <= int(state.enemy0_y) + strategy.height_slop


def _fire_window(session: ControllerSession, strategy: PhantoonStrategy) -> int:
    """Spend N counted shots (ammo/charge delta) while the eye is open."""
    if is_morph(session.state.pose):
        try:
            unmorph(session)
        except Exception:
            return 0
    if session.state.selected_item != strategy.weapon:
        try:
            ensure_weapon(session, strategy.weapon)
        except RuntimeError:
            return 0

    shots = 0
    last_spend = -99
    seen_open = False
    func0 = _u16(_ram(_env_of(session)), ADDR_ENEMY0_FUNC)
    if rain_phase(func0) and not rain_charge_ok(session.state.enemy0_x):
        return 0
    if right_park(session.state.enemy0_x, skip_x=strategy.skip_enemy_x):
        if int(session.state.samus_x) < strategy.right_jump_x:
            _go_to_right_seat(session, strategy)
    for _ in range(strategy.window_timeout):
        st = session.state
        if _dead(session) or st.enemy0_hp == 0:
            break
        hittable = _hittable(session)
        if hittable:
            seen_open = True
        if not hittable and (shots >= strategy.min_shots_to_fire or seen_open):
            break
        if strategy.weapon == WEAPON_MISSILES and st.missiles <= 0:
            break
        if shots >= strategy.shots_per_window:
            break

        func_now = _u16(_ram(_env_of(session)), ADDR_ENEMY0_FUNC)
        if rain_phase(func_now) and not rain_charge_ok(st.enemy0_x):
            break
        names = _aim_names(st, strategy, rain=False)
        if st.pose in HURT_POSES:
            hold(session, 1, reason="phan_hurt")
            continue

        parked_right = right_park(st.enemy0_x, skip_x=strategy.skip_enemy_x)
        still_left = int(st.samus_x) < strategy.right_seat_x_min
        if strategy.weapon == WEAPON_MISSILES:
            close = abs(int(st.samus_x) - int(st.enemy0_x)) <= strategy.fire_close_x
            on_floor = int(st.samus_y) >= strategy.floor_y_min - 6
            if parked_right and still_left:
                break
            if int(st.samus_x) >= strategy.kite_x_max and shots >= 1:
                break
            if shots >= strategy.min_shots_to_fire and not close:
                break
            fire = (
                hittable
                and close
                and (not on_floor)
                and st.missiles > 0
                and st.pose not in HURT_POSES
                and shots < strategy.shots_per_window
                and (session.frame - last_spend) >= 10
            )
            if fire:
                # No LEFT/RIGHT on the spend — diagonal walks the shot off the eye.
                fire_names = ["X"]
                if int(st.samus_y) > int(st.enemy0_y) + 10:
                    fire_names.insert(0, "UP")
                names = fire_names
            ms_before = st.missiles
            hold(session, 1, *tuple(dict.fromkeys(names)), reason="phan_eye_shot")
            if session.state.missiles < ms_before:
                shots += 1
                last_spend = session.frame
            continue

        close = abs(int(st.samus_x) - int(st.enemy0_x)) <= strategy.fire_close_x
        if parked_right:
            close = int(st.samus_x) >= strategy.right_jump_x
        if parked_right and still_left:
            break
        if int(st.samus_x) >= strategy.kite_x_max and shots >= 1:
            break
        if not _charged(session, strategy):
            names.append("X")
            hold(session, 1, *tuple(dict.fromkeys(names)), reason="phan_charge")
            continue
        face_ok = (not parked_right) or int(st.facing) == 4
        fire = (
            hittable
            and close
            and face_ok
            and in_release_band(st, strategy)
            and (not _need_height(st, strategy))
            and st.pose not in HURT_POSES
            and shots < strategy.shots_per_window
        )
        if not fire:
            names.append("X")
            hold(session, 1, *tuple(dict.fromkeys(names)), reason="phan_charge")
            continue
        fire_names: list[str] = []
        if int(st.samus_y) > int(st.enemy0_y) + 10:
            fire_names.append("UP")
        ch_before = beam_charge(_env_of(session))
        hold(
            session,
            strategy.fire_release_frames,
            *tuple(fire_names),
            reason="phan_fire",
        )
        ch_after = beam_charge(_env_of(session))
        if ch_after < ch_before and ch_before >= strategy.charge_full:
            shots += 1
            last_spend = session.frame

    if not _dead(session) and session.state.enemy0_hp > 0:
        for _ in range(24):
            if session.state.pose not in HURT_POSES:
                break
            hold(session, 1, reason="phan_land")
        _go_to_seat(session, strategy)
    return shots


def play_phantoon_fight(
    session: ControllerSession,
    *,
    strategy: PhantoonStrategy = PhantoonStrategy(),
    require_boss_bit: bool = True,
) -> PhantoonEvidence:
    """Fight Phantoon from room ``0xCD13`` until HP 0 (+ optional boss bit).

    Seat left, wait for a measured open-eye spritemap, spend charge/missiles
    counted by ammo or charge actually decreasing, retreat. Never Super.
    """
    catalog = phantoon_catalog()
    start = session.frame
    if session.state.room_id != ROOM_PHANTOON:
        raise RuntimeError(
            f"Phantoon fight expected room 0x{ROOM_PHANTOON:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    try:
        ensure_weapon(session, strategy.weapon)
    except RuntimeError:
        pass
    _go_to_seat(session, strategy)

    peak_hp = session.state.enemy0_hp
    min_hp = session.state.enemy0_hp
    body_zero_frame: int | None = start if peak_hp == 0 else None
    boss_bit_frame: int | None = None
    prev_hp = session.state.enemy0_hp
    current_phase = phantoon_phase(session.state)
    phase_transitions: list[tuple[str, int]] = [(current_phase, start)]
    shots_fired = 0
    windows = 0
    seen: set[int] = set()
    park_x = int(session.state.enemy0_x)
    last_func: int | None = None

    for _ in range(strategy.max_fight_frames):
        state = session.state
        if state.room_id != ROOM_PHANTOON:
            break
        if _dead(session):
            break
        peak_hp = max(peak_hp, state.enemy0_hp)
        min_hp = min(min_hp, state.enemy0_hp)
        if _session_eye_open(session):
            seen.add(int(state.enemy0_spritemap))

        next_phase = phantoon_phase(state)
        if next_phase != current_phase:
            phase_transitions.append((next_phase, session.frame))
            current_phase = next_phase

        if body_zero_frame is None and state.enemy0_hp == 0 and prev_hp > 0:
            body_zero_frame = session.frame
            min_hp = 0
        prev_hp = state.enemy0_hp

        if body_zero_frame is not None:
            if boss_defeated_in_state(session.state, catalog):
                boss_bit_frame = session.frame
                break
            if not require_boss_bit:
                break
            if session.frame - body_zero_frame >= strategy.boss_bit_grace_frames:
                break
            hold(session, 1, reason="phantoon_death_anim")
            continue

        func_now = _u16(_ram(_env_of(session)), ADDR_ENEMY0_FUNC)
        if func_now != last_func:
            park_x = int(state.enemy0_x)
            last_func = func_now
        if not charge_window_ok(func_now, park_x) and (
            rain_phase(func_now) or right_park(park_x, skip_x=strategy.skip_enemy_x)
        ):
            _rain_corner_wait(session, strategy)
            continue
        ready = (
            _session_eye_open(session)
            and state.samus_y >= strategy.floor_y_min
            and charge_window_ok(func_now, park_x)
        )
        if strategy.weapon == WEAPON_MISSILES and state.missiles < strategy.min_shots_to_fire:
            try:
                ensure_weapon(session, WEAPON_BEAM)
            except RuntimeError:
                pass
            strategy = PhantoonStrategy(
                weapon=WEAPON_BEAM,
                shots_per_window=3,
                max_fight_frames=strategy.max_fight_frames,
                boss_bit_grace_frames=strategy.boss_bit_grace_frames,
            )
            continue
        if strategy.weapon == WEAPON_MISSILES:
            ready = ready and state.missiles >= strategy.min_shots_to_fire
        if ready:
            got = _fire_window(session, strategy)
            if got:
                windows += 1
                shots_fired += got
            else:
                hold(session, 1, reason="phan_wait_eye")
            continue

        if not seated(state, strategy):
            _keep_seat(session, strategy)
        elif strategy.weapon == WEAPON_BEAM:
            _hold_charge(session, 1)
        else:
            hold(session, 1, "DOWN", reason="phan_wait_eye")

    final_hp = session.state.enemy0_hp
    boss_set = boss_defeated_in_state(session.state, catalog)
    if _dead(session):
        outcome = "died"
    elif boss_set:
        outcome = "phantoon_defeated"
    elif body_zero_frame is not None:
        outcome = "phantoon_body_zero_no_boss_bit"
    else:
        outcome = "timeout"

    return PhantoonEvidence(
        start_frame=start,
        body_zero_frame=body_zero_frame,
        boss_bit_frame=boss_bit_frame,
        end_frame=session.frame,
        peak_body_hp=peak_hp,
        min_body_hp=min_hp,
        action_frames=session.frame - start,
        final_body_hp=final_hp,
        boss_bit_set=boss_set,
        outcome=outcome,
        phase_transitions=tuple(phase_transitions),
        shots_fired=shots_fired,
        windows=windows,
        vulnerable_spritemaps=tuple(sorted(seen)),
    )


__all__ = [
    "CHARGE_FULL",
    "FLOOR_Y_MAX",
    "FLOOR_Y_MIN",
    "PHANTOON_DEFEATED",
    "PHANTOON_INVISIBLE",
    "PHANTOON_VULNERABLE",
    "Pickup",
    "ROOM_PHANTOON",
    "SEAT_X",
    "SEAT_X_MAX",
    "SEAT_X_MIN",
    "VULNERABLE_SPRITEMAPS",
    "WEAPON_BEAM",
    "WEAPON_MISSILES",
    "WEAPON_SUPERS",
    "PhantoonEvidence",
    "PhantoonStrategy",
    "beam_charge",
    "charge_window_ok",
    "enemy_extra",
    "eye_open",
    "fight_phantoon_action",
    "floor_release_ok",
    "in_release_band",
    "list_pickups",
    "phantoon_phase",
    "play_phantoon_fight",
    "rain_charge_ok",
    "rain_corner_morph",
    "rain_phase",
    "rain_vulnerable",
    "right_park",
    "seated",
]
