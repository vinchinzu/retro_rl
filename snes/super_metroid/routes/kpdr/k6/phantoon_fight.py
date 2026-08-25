"""Phantoon room fight wrapper for the continuous spine (K6).

Thin hop: natural ``0xCD13`` entry → wiki missile doppler
(``play_phantoon_doppler_fight``). Charge-only, charge+missiles, and
Ice-on X-Factor stay in ``combat/phantoon*.py`` as research. Combat is
imported inside the callables so spine/graph load stays cycle-free.
Not continuous evidence until composed on ``--to phantoon``.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState, read_bank7e_wram
from super_metroid.routes.controller_common import require_room
from super_metroid.routes.kpdr.room_ids import ROOM_PHANTOON
from super_metroid.routes.runtime import ControllerSession, RouteSession, Split

# Wrecked Ship boss bits ($7E:D82B); bit 0 = Phantoon. Duplicated from
# combat.phantoon so this module can load during spine construction.
ADDR_WS_BOSS_BITS = 0xD82B
PHANTOON_BOSS_BIT = 0x01

__all__ = [
    "ADDR_WS_BOSS_BITS",
    "PHANTOON_BOSS_BIT",
    "phantoon_boss_bit_set",
    "play_phantoon_room_fight",
    "require_phantoon_defeated",
]


def phantoon_boss_bit_set(session: ControllerSession) -> bool:
    """True when Wrecked Ship ``$7E:D82B`` bit 0 is set.

    Low ``env.get_ram()`` never contains that byte — peek bank 7E live,
    fall back to parsed ``boss_bits`` for unit doubles.
    """
    env = getattr(session, "env", None)
    if env is not None:
        try:
            ram = read_bank7e_wram(env)
            return bool(int(ram[ADDR_WS_BOSS_BITS]) & PHANTOON_BOSS_BIT)
        except Exception:
            pass
    from super_metroid.combat.features import boss_defeated_in_state, phantoon_catalog

    return boss_defeated_in_state(session.state, phantoon_catalog())


def require_phantoon_defeated(
    session: RouteSession, splits: list[Split], result: object = None
) -> None:
    """SpineHop ``after``: boss bit must be set. Never trust low WRAM alone."""
    del splits, result
    if phantoon_boss_bit_set(session):
        return
    raise RuntimeError(
        "phantoon_fight: Wrecked Ship $D82B bit 0 not set: "
        f"{session.state}"
    )


def play_phantoon_room_fight(session: ControllerSession) -> SuperMetroidState:
    """From Phantoon-room doorway entry: wiki 2-2-N doppler until boss bit.

    Public KPDR/PRKD doppler: 10f missile spacing, Super only if HP ≤ 600.
    Assist energy+ammo stay on (primary track). Charge-only 20537f is
    research; this body is the pin-faster 12118f dual (rr-7lc5).
    """
    from super_metroid.combat.phantoon_doppler import play_phantoon_doppler_fight

    require_room(session, ROOM_PHANTOON, "phantoon_room_fight")
    if phantoon_boss_bit_set(session) and int(session.state.enemy0_hp) == 0:
        return session.state
    evidence = play_phantoon_doppler_fight(session)
    if evidence.outcome != "phantoon_defeated":
        raise RuntimeError(
            f"phantoon_room_fight: fight failed ({evidence.outcome}): "
            f"{session.state}"
        )
    return session.state
