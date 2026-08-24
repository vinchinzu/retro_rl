"""Wiki 2-round X-Factor / popTOON Phantoon (Ice-on caveat).

https://wiki.supermetroid.run/Phantoon#Any.25_KPDR_.28popTOON_-_X-Factor_Plus_Missiles.29
https://wiki.supermetroid.run/Phantoon#Any.25_KPDR_.28X-Factor_Only.29
https://wiki.supermetroid.run/Charge_Beam_Combos#Wave_Shield

Wave Shield (X-Factor) = Charge + Wave + Power Bomb, charge 120.
Pin beams 0x1007 = Charge+Ice+Wave+Spazer. Ice 0x0002 is equipped, so the
combo is Ice Shield / Ice+Wave particles, not true X-Factor. No pause-menu
beam-toggle helper exists — do not invent one.

popTOON: 2+2+XF, then 2+2+S. Never Super unless HP ≤ 600 (kill).
Product baseline is left-corner charge-only assist 20537f ×2 — do not replace it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from super_metroid.combat.phantoon import (
    ADDR_ENEMY0_FUNC,
    ADDR_WS_BOSS_BITS,
    PHANTOON_BOSS_BIT,
    PhantoonStrategy,
    _env_of,
    _go_to_seat,
    _rain_corner_wait,
    beam_charge,
    charge_window_ok,
    enemy_extra,
    eye_open,
    rain_phase,
    right_park,
    seated,
)
from super_metroid.combat.primitives import ensure_weapon
from super_metroid.ram import GameplayPhase, SuperMetroidState, read_bank7e_wram
from super_metroid.routes.controller_common import is_morph, unmorph
from super_metroid.routes.runtime import ControllerSession, hold

WEAPON_POWER_BOMBS = 3
BEAM_WAVE, BEAM_ICE, BEAM_SPAZER, BEAM_CHARGE = 0x0001, 0x0002, 0x0004, 0x1000
PIN_BEAMS, PIN_ITEMS = 0x1007, 0x3105
SUPER_KILL_HP = 600
CHARGE_COMBO = 120
N_SAMUS_PROJECTILES = 10
ADDR_SAMUS_PROJ_TYPE, ADDR_SAMUS_PROJ_X, ADDR_SAMUS_PROJ_Y = 0x0C04, 0x0C18, 0x0C4A
PROJ_ICE_SBA, PROJ_WAVE_SBA = 0x001C, 0x001D
PROJ_SPAZER_SBA, PROJ_PLASMA_SBA = 0x001E, 0x001F
MISSILE_COOLDOWN, WATCH_FRAMES = 10, 90
_MISS_NOTE = (
    "Ice-on SBA (Charge+Ice+Wave+Spazer, Ice=0x0002) did not chip enemy0_hp. "
    "True Wave Shield needs Ice off; no pause-menu unequip helper. "
    "Do not wire popTOON. Product stays 20537f charge-only."
)


class PoptoonStep(str, Enum):
    FIRE_MISSILE = "fire_missile"
    CHARGE_XFACTOR = "charge_xfactor"
    FIRE_SUPER = "fire_super"
    DONE = "done"
    BLOCKED_NO_PB = "blocked_no_pb"
    BLOCKED_ICE = "blocked_ice"


@dataclass
class PoptoonProgress:
    """2+2+XF then 2+2+S (Super only if HP ≤ 600)."""

    round_index: int = 1
    missiles_this_round: int = 0
    xfactor_fired: bool = False
    super_fired: bool = False


@dataclass
class XFactorWindowEvidence:
    """One Ice-on Wave-Shield / SBA attempt on a measured open."""

    opened: bool
    hp_before: int
    hp_after: int
    hp_drop: int
    pb_before: int
    pb_after: int
    pb_spent: int
    charge_peak: int
    charge_at_release: int
    combo_charge: bool
    ice_equipped: bool
    true_wave_shield: bool
    equipped_beams: int
    projectile_types: tuple[int, ...]
    combo_class: str
    chips: bool
    outcome: str
    frames: int
    notes: str = ""
    projectiles: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["equipped_beams"] = f"0x{self.equipped_beams:04X}"
        d["beams"] = decode_beams(self.equipped_beams)
        d["projectile_types"] = [f"0x{t:04X}" for t in self.projectile_types]
        return d


def _u16(ram: Any, address: int) -> int:
    if ram is None or len(ram) < address + 2:
        return 0
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def _ram(session: ControllerSession) -> Any:
    env = _env_of(session)
    if env is None:
        return None
    try:
        return env.get_ram()
    except Exception:
        return None


def decode_beams(beams: int) -> dict[str, bool]:
    b = int(beams)
    return {
        "charge": bool(b & BEAM_CHARGE),
        "ice": bool(b & BEAM_ICE),
        "wave": bool(b & BEAM_WAVE),
        "spazer": bool(b & BEAM_SPAZER),
    }


def ice_equipped(beams: int) -> bool:
    return bool(int(beams) & BEAM_ICE)


def true_wave_shield(beams: int) -> bool:
    """Charge+Wave, Ice and Spazer off — otherwise not X-Factor."""
    b = int(beams)
    return bool(b & BEAM_CHARGE) and bool(b & BEAM_WAVE) and not (b & (BEAM_ICE | BEAM_SPAZER))


def super_ok(hp: int) -> bool:
    """Super only as a kill — HP ≤ 600. Never Super-spray / enrage."""
    return 0 < int(hp) <= SUPER_KILL_HP


def xfactor_ready(state: SuperMetroidState) -> bool:
    """PB selected, ammo > 0, not morph. 0 PB does not fire."""
    return (
        not is_morph(int(state.pose))
        and int(state.selected_item) == WEAPON_POWER_BOMBS
        and int(state.power_bombs) > 0
    )


def xfactor_fire_action(state: SuperMetroidState) -> tuple[str, ...]:
    return ("X",) if xfactor_ready(state) else ()


def next_poptoon_step(
    progress: PoptoonProgress,
    *,
    hp: int,
    power_bombs: int,
    ice_on: bool = True,
    combo_chips: bool | None = None,
) -> PoptoonStep:
    """Round 1: 2+2+XF. Round 2: 2+2+S iff ``super_ok``."""
    if int(hp) <= 0:
        return PoptoonStep.DONE
    if progress.round_index <= 1:
        if progress.missiles_this_round < 4:
            return PoptoonStep.FIRE_MISSILE
        if not progress.xfactor_fired:
            return _xf_or_blocked(power_bombs, ice_on, combo_chips)
        return PoptoonStep.DONE
    if progress.missiles_this_round < 4:
        return PoptoonStep.FIRE_MISSILE
    if super_ok(hp) and not progress.super_fired:
        return PoptoonStep.FIRE_SUPER
    if progress.super_fired:
        return PoptoonStep.DONE
    if not progress.xfactor_fired:
        return _xf_or_blocked(power_bombs, ice_on, combo_chips)
    return PoptoonStep.DONE


def _xf_or_blocked(power_bombs: int, ice_on: bool, combo_chips: bool | None) -> PoptoonStep:
    if int(power_bombs) <= 0:
        return PoptoonStep.BLOCKED_NO_PB
    if ice_on and combo_chips is False:
        return PoptoonStep.BLOCKED_ICE
    return PoptoonStep.CHARGE_XFACTOR


def classify_combo(types: list[int] | tuple[int, ...]) -> str:
    seen = {int(t) & 0xFFFF for t in types if int(t)}
    if not seen:
        return "none"
    ice, wave = PROJ_ICE_SBA in seen, PROJ_WAVE_SBA in seen
    if wave and not ice:
        return "wave_shield"
    if ice and wave:
        return "ice_wave_shield"
    if ice:
        return "ice_shield"
    if PROJ_SPAZER_SBA in seen:
        return "spazer_shield"
    if PROJ_PLASMA_SBA in seen:
        return "plasma_shield"
    return "unknown:" + ",".join(f"0x{t:04X}" for t in sorted(seen))


def samus_projectiles(env: Any) -> list[dict[str, object]]:
    ram = None
    if env is not None:
        try:
            ram = env.get_ram()
        except Exception:
            ram = None
    if ram is None:
        return []
    out: list[dict[str, object]] = []
    for slot in range(N_SAMUS_PROJECTILES):
        kind = _u16(ram, ADDR_SAMUS_PROJ_TYPE + slot * 2)
        if kind == 0:
            continue
        out.append(
            {
                "slot": slot,
                "type": f"0x{kind:04X}",
                "type_u16": kind,
                "x": _u16(ram, ADDR_SAMUS_PROJ_X + slot * 2),
                "y": _u16(ram, ADDR_SAMUS_PROJ_Y + slot * 2),
            }
        )
    return out


def _dead(session: ControllerSession) -> bool:
    st = session.state
    return int(st.health) == 0 or st.phase is GameplayPhase.DEATH_OR_GAME_OVER


def _body_func(session: ControllerSession) -> int:
    return _u16(_ram(session), ADDR_ENEMY0_FUNC)


def _boss_bit(session: ControllerSession) -> bool:
    env = _env_of(session)
    if env is None:
        return False
    try:
        return bool(int(read_bank7e_wram(env)[ADDR_WS_BOSS_BITS]) & PHANTOON_BOSS_BIT)
    except Exception:
        return False


def _note_projs(
    session: ControllerSession,
    types: list[int],
    log: list[dict[str, object]],
    charge: int,
) -> None:
    for p in samus_projectiles(_env_of(session)):
        t = int(p["type_u16"])
        if t not in types:
            types.append(t)
        if len(log) < 80:
            log.append(
                {
                    "frame": session.frame,
                    "charge": charge,
                    "hp": session.state.enemy0_hp,
                    "pb": session.state.power_bombs,
                    "slot": p["slot"],
                    "type": p["type"],
                    "x": p["x"],
                    "y": p["y"],
                }
            )


def xfactor_snapshot(session: ControllerSession) -> dict[str, object]:
    st = session.state
    extra = enemy_extra(_env_of(session))
    beams = int(st.equipped_beams)
    return {
        "frame": int(getattr(session, "frame", st.frame)),
        "room_id_hex": f"0x{st.room_id:04X}",
        "samus_xy": [st.samus_x, st.samus_y],
        "pose": st.pose,
        "facing": st.facing,
        "health": st.health,
        "game_state": st.game_state,
        "missiles": st.missiles,
        "supers": st.super_missiles,
        "power_bombs": st.power_bombs,
        "max_power_bombs": st.max_power_bombs,
        "selected_item": st.selected_item,
        "equipped_items": f"0x{st.equipped_items:04X}",
        "equipped_beams": f"0x{beams:04X}",
        "beams": decode_beams(beams),
        "ice_equipped": ice_equipped(beams),
        "true_wave_shield": true_wave_shield(beams),
        "charge": beam_charge(_env_of(session)),
        "enemy0_hp": st.enemy0_hp,
        "enemy_xy": [st.enemy0_x, st.enemy0_y],
        "enemy0_spritemap": f"0x{st.enemy0_spritemap:04X}",
        "func": extra.get("func"),
        "eye_ilist": extra.get("eye_ilist"),
        "eye_xy": [extra.get("eye_x"), extra.get("eye_y")],
        "eye_open": eye_open(st, _env_of(session)),
        "seated": seated(st),
        "xfactor_ready": xfactor_ready(st),
        "projectiles": samus_projectiles(_env_of(session)),
        "boss_bit": _boss_bit(session),
    }


def wait_charge_window(
    session: ControllerSession,
    *,
    timeout: int = 2400,
    func_log: list[dict[str, object]] | None = None,
) -> bool:
    """Seat left; skip rain/right parks with product corner snipe."""
    strat = PhantoonStrategy()
    if not seated(session.state, strat):
        _go_to_seat(session, strat)
    park_x = int(session.state.enemy0_x)
    last_func: int | None = None
    for _ in range(timeout):
        st = session.state
        if _dead(session) or int(st.enemy0_hp) == 0:
            return False
        func = _body_func(session)
        if func != last_func:
            park_x = int(st.enemy0_x)
            last_func = func
            if func_log is not None:
                extra = enemy_extra(_env_of(session))
                func_log.append(
                    {
                        "frame": session.frame,
                        "func": extra.get("func"),
                        "eye_ilist": extra.get("eye_ilist"),
                        "enemy_xy": [st.enemy0_x, st.enemy0_y],
                        "park_x": park_x,
                        "samus_xy": [st.samus_x, st.samus_y],
                        "charge": beam_charge(_env_of(session)),
                        "hp": st.enemy0_hp,
                    }
                )
        if charge_window_ok(func, park_x, st.enemy0_y) and eye_open(st, _env_of(session)):
            return True
        skip = rain_phase(func) or right_park(park_x) or (0 < int(park_x) < 100)
        if skip:
            _rain_corner_wait(session, strat)
            continue
        if is_morph(int(st.pose)):
            try:
                unmorph(session)
            except Exception:
                hold(session, 1, reason="xf_unmorph")
            continue
        if not seated(st, strat):
            _go_to_seat(session, strat)
            continue
        hold(session, 1, reason="xf_wait_open")
    return False


def _face_right(session: ControllerSession) -> None:
    if int(session.state.facing) != 8:
        hold(session, 1, "RIGHT", reason="xf_face")


def attempt_xfactor(
    session: ControllerSession,
    *,
    charge_target: int = CHARGE_COMBO,
    hold_max: int = 160,
    watch: int = WATCH_FRAMES,
) -> XFactorWindowEvidence:
    """Select PB, hold X to charge ≥60 (aim 120), release toward Phantoon."""
    start = session.frame
    beams = int(session.state.equipped_beams)
    if is_morph(int(session.state.pose)):
        try:
            unmorph(session)
        except Exception:
            pass
    try:
        ensure_weapon(session, WEAPON_POWER_BOMBS)
    except RuntimeError:
        pass
    _face_right(session)
    hp0, pb0 = int(session.state.enemy0_hp), int(session.state.power_bombs)
    opened = eye_open(session.state, _env_of(session))
    charge_peak = beam_charge(_env_of(session))
    types: list[int] = []
    proj_log: list[dict[str, object]] = []
    min_pb, ready = pb0, xfactor_ready(session.state)
    charge_at_release = charge_peak
    if ready:
        for _ in range(hold_max):
            if _dead(session) or int(session.state.enemy0_hp) == 0:
                break
            names = ["X"] if int(session.state.facing) == 8 else ["RIGHT", "X"]
            hold(session, 1, *names, reason="xf_charge")
            ch = beam_charge(_env_of(session))
            charge_peak = max(charge_peak, ch)
            min_pb = min(min_pb, int(session.state.power_bombs))
            _note_projs(session, types, proj_log, ch)
            if int(session.state.power_bombs) < pb0 and ch < charge_peak:
                charge_at_release = ch
                break
            if ch >= charge_target:
                charge_at_release = ch
                break
        charge_at_release = max(charge_at_release, beam_charge(_env_of(session)))
        _face_right(session)
        hold(session, 2, reason="xf_release")
    else:
        hold(session, 1, reason="xf_not_ready")
    for _ in range(watch):
        if _dead(session):
            break
        hold(session, 1, reason="xf_watch")
        min_pb = min(min_pb, int(session.state.power_bombs))
        _note_projs(session, types, proj_log, beam_charge(_env_of(session)))
        if int(session.state.enemy0_hp) < hp0:
            break
    hp1, pb1 = int(session.state.enemy0_hp), int(session.state.power_bombs)
    hp_drop = max(0, hp0 - hp1)
    chips = hp_drop > 0
    combo_class = classify_combo(types)
    if not ready:
        outcome, notes = "not_ready", "PB not selected or 0 ammo / morph — did not fire."
    elif not chips:
        outcome, notes = "ice_on_xfactor_miss", _MISS_NOTE
    else:
        outcome = "ice_on_xfactor_chip"
        notes = f"Ice-on combo class={combo_class} chipped {hp_drop}. Not true Wave Shield."
    return XFactorWindowEvidence(
        opened=opened,
        hp_before=hp0,
        hp_after=hp1,
        hp_drop=hp_drop,
        pb_before=pb0,
        pb_after=pb1,
        pb_spent=max(0, pb0 - min_pb),
        charge_peak=charge_peak,
        charge_at_release=charge_at_release,
        combo_charge=charge_peak >= CHARGE_COMBO,
        ice_equipped=ice_equipped(beams),
        true_wave_shield=true_wave_shield(beams),
        equipped_beams=beams,
        projectile_types=tuple(types),
        combo_class=combo_class,
        chips=chips,
        outcome=outcome,
        frames=session.frame - start,
        notes=notes,
        projectiles=tuple(proj_log),
    )

