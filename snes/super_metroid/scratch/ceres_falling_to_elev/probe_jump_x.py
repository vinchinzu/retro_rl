"""Scratch: tune _CERES_FALLING_DOOR_JUMP_X against the real door policy.

Boot first-control, play the product reverse until the exit phase starts,
snapshot, then replay the product action loop per candidate takeoff x.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import parse_state
from super_metroid.routes.kpdr.ceres import magnet
from super_metroid.routes.kpdr.ceres.geometry import (
    CERES_DATA_DIR,
    _CERES_FALLING_DOOR_LEDGE_Y,
)
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
from super_metroid.takeoff import PlatformHop, TakeoffWindow

OUT = Path(__file__).resolve().parent / "jump_x_sweep.json"
BASE = magnet.CERES_FALLING_DOOR_HOP


def snap(state, **extra) -> dict:
    row = {
        "gs": int(state.game_state),
        "x": int(state.samus_x),
        "y": int(state.samus_y),
        "pose": int(state.pose),
        "mx": int(state.momentum_x),
        "inv": int(state.invincibility_timer),
        "vy": int(state.velocity_y),
        "vd": int(state.vertical_direction),
    }
    row.update(extra)
    return row


def run(session, track: CeresFallingEscapeTrack) -> dict:
    frames: list[dict] = []
    for _ in range(200):
        st = session.state
        if int(st.room_id) == ROOM_CERES_ELEVATOR and int(st.game_state) == 8:
            frames.append(snap(st, event="dest_gs8"))
            return {"dest": snap(st), "frames": frames}
        names, track = ceres_falling_escape_action(st, track)
        if int(st.samus_x) <= 100:
            frames.append(snap(st, act=list(names)))
        session.step(buttons(*names) if names else idle_action(), "probe")
    return {"dest": snap(session.state, event="timeout"), "frames": frames}


def main() -> None:
    env, session = _boot_pin(CERES_DATA_DIR / "ceres_first_control.state")
    try:
        play_ceres_outbound_to_ridley(session)
        session.span(ActionSpan(("LEFT", "A"), 24, "ceres_ridley_exit"))
        play_ceres_flat_to_scientist(session)
        play_ceres_scientist_to_magnet(session)
        play_ceres_magnet_to_falling(session)

        track = CeresFallingEscapeTrack()
        for _ in range(400):
            if track.phase == "exit":
                break
            names, track = ceres_falling_escape_action(session.state, track)
            session.step(
                buttons(*names) if names else idle_action(), "ceres_falling"
            )
        else:
            raise SystemExit(f"never reached exit phase: {snap(session.state)}")
        seat = snap(session.state)
        print("exit-phase seat", seat, "track", track, flush=True)

        blob = env.em.get_state()
        runs = []
        for jump_x in (28, 30, 33, 36, 39, 42):
            magnet.CERES_FALLING_DOOR_HOP = PlatformHop(
                _CERES_FALLING_DOOR_LEDGE_Y,
                16,
                70,
                TakeoffWindow((16, jump_x), "LEFT", min_momentum=1),
            )
            env.em.set_state(blob)
            session.state = parse_state(env.get_ram(), frame=session.frame)
            res = run(session, replace(track))
            res["jump_x"] = jump_x
            runs.append(res)
            d = res["dest"]
            band = 624 <= d["y"] <= 641 and d["vd"] == 1 and d["vy"] > 0
            print(
                f"jump_x={jump_x:3d} ({d['x']},{d['y']}) p{d['pose']} mx{d['mx']} "
                f"vy{d['vy']} vd{d['vd']} inv{d['inv']} band={int(band)}",
                flush=True,
            )
        magnet.CERES_FALLING_DOOR_HOP = BASE
    finally:
        env.close()
    OUT.write_text(json.dumps({"seat": seat, "runs": runs}, indent=1) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
