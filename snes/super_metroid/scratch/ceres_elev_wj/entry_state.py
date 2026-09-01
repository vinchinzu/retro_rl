"""Scratch: cache the elevator fast-entry snapshot so sweeps skip the boot.

The product reverse to ROOM_CERES_ELEVATOR + gs 8 is ~4,500 frames. Play it
once, write ``env.em.get_state()`` to a gitignored ``.state`` next to this
file, and let every sweep reload it in a second. Scratch only — nothing in
the package may reference this file.
"""

from __future__ import annotations

from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import parse_state
from super_metroid.routes.kpdr.ceres.geometry import CERES_DATA_DIR
from super_metroid.routes.kpdr.ceres.magnet import (
    CeresFallingEscapeTrack,
    ceres_falling_escape_action,
    play_ceres_magnet_to_falling,
)
from super_metroid.routes.kpdr.ceres.outbound import (
    play_ceres_flat_to_scientist,
    play_ceres_outbound_to_ridley,
)
from super_metroid.routes.kpdr.ceres.scientist import play_ceres_scientist_to_magnet
from super_metroid.routes.kpdr.ceres.spine import _boot_pin
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_ELEVATOR
from super_metroid.routes.runtime import ActionSpan

HERE = Path(__file__).resolve().parent
ENTRY_STATE = HERE / "elev_entry.state"


def snap(state, **extra) -> dict:
    row = {
        "gs": int(state.game_state),
        "x": int(state.samus_x),
        "y": int(state.samus_y),
        "pose": int(state.pose),
        "mx": int(state.momentum_x),
        "vy": int(state.velocity_y),
        "vd": int(state.vertical_direction),
        "mt": int(state.movement_type),
        "inv": int(state.invincibility_timer),
    }
    row.update(extra)
    return row


def play_to_entry(session) -> None:
    play_ceres_outbound_to_ridley(session)
    session.span(ActionSpan(("LEFT", "A"), 24, "ceres_ridley_exit"))
    play_ceres_flat_to_scientist(session)
    play_ceres_scientist_to_magnet(session)
    play_ceres_magnet_to_falling(session)
    track = CeresFallingEscapeTrack()
    for _ in range(500):
        st = session.state
        if int(st.room_id) == ROOM_CERES_ELEVATOR and int(st.game_state) == 8:
            return
        names, track = ceres_falling_escape_action(st, track)
        session.step(buttons(*names) if names else idle_action(), "ceres_falling")
    raise SystemExit(f"never reached elev gs8: {snap(session.state)}")


def open_entry() -> tuple[object, object]:
    """Env + session parked on the elevator fast entry (cached after run 1)."""
    env, session = _boot_pin(CERES_DATA_DIR / "ceres_first_control.state")
    if ENTRY_STATE.exists():
        env.em.set_state(ENTRY_STATE.read_bytes())
        session.state = parse_state(env.get_ram(), frame=session.frame)
        return env, session
    play_to_entry(session)
    ENTRY_STATE.write_bytes(env.em.get_state())
    return env, session


if __name__ == "__main__":
    env, session = open_entry()
    try:
        print("entry", snap(session.state))
        print("state", ENTRY_STATE)
    finally:
        env.close()
