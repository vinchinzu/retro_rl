"""Scratch: sweep the Falling west-door leave from first-control.

One boot to the door ledge, snapshot, then replay branch variants off that
in-memory snapshot. Dest WRAM is the gs=9 leave frame frozen, so the sweep
scores the leave frame directly. Not a product CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import FACING_LEFT, parse_state
from super_metroid.routes.kpdr.ceres.geometry import (
    CERES_DATA_DIR,
    _CERES_FALLING_DOOR_LEDGE_Y,
)
from super_metroid.routes.kpdr.ceres import magnet
from super_metroid.routes.kpdr.ceres.magnet import (
    CeresFallingEscapeTrack,
    _ceres_fast_entry_window,
    _ceres_grounded,
    _tas_l_pump,
    ceres_falling_escape_action,
    play_ceres_magnet_to_falling,
)
from super_metroid.routes.kpdr.ceres.outbound import (
    play_ceres_flat_to_scientist,
    play_ceres_outbound_to_ridley,
)
from super_metroid.routes.kpdr.ceres.scientist import play_ceres_scientist_to_magnet
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_ELEVATOR, ROOM_CERES_FALLING
from super_metroid.routes.runtime import ActionSpan

OUT = Path(__file__).resolve().parent / "door_leave_sweep.json"

_strict_window = magnet._ceres_fast_entry_window


def _relaxed_window(state) -> bool:
    """Strict window with momentum_x >= 1 — momentum_x >= 2 is unreachable."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == 8
        and 210 <= int(state.samus_x) <= 220
        and 624 <= int(state.samus_y) <= 641
        and int(state.pose) in (25, 26, 27, 28)
        and int(state.vertical_direction) == 1
        and int(state.velocity_y) > 0
        and int(state.momentum_x) >= 1
        and int(state.invincibility_timer) > 0
    )


def snap(state, **extra) -> dict:
    row = {
        "gs": int(state.game_state),
        "x": int(state.samus_x),
        "y": int(state.samus_y),
        "pose": int(state.pose),
        "mx": int(state.momentum_x),
        "mxs": int(state.momentum_x_sub),
        "sf": int(state.speed_flag),
        "inv": int(state.invincibility_timer),
        "vx": int(state.velocity_x),
        "vy": int(state.velocity_y),
        "vd": int(state.vertical_direction),
        "mt": int(state.movement_type),
        "kbt": int(state.knockback_timer),
        "face": int(state.facing),
    }
    row.update(extra)
    return row


@dataclass(frozen=True)
class Variant:
    name: str
    pre: tuple[tuple[str, ...], ...] = ()
    jump_x_hi: int = 60
    min_momentum: int = 2
    release_y: int = 0  # 0 = hold A the whole ascent
    turn_x: int = 0  # 0 = never air-turn RIGHT
    spin: bool = True
    air_hold: tuple[str, ...] | None = None  # override the air button set


def door_leave(session, var: Variant, *, max_frames: int = 320) -> dict:
    frames: list[dict] = []
    pump_i = 0
    jumped = False
    hazard = False
    takeoff = None
    for i in range(max_frames):
        st = session.state
        room = int(st.room_id)
        gs = int(st.game_state)
        x = int(st.samus_x)
        y = int(st.samus_y)
        pose = int(st.pose)
        hazard = hazard or int(st.movement_type) == 21

        if room == ROOM_CERES_ELEVATOR and gs == 8:
            win = bool(_ceres_fast_entry_window(st))
            frames.append(snap(st, event="dest_gs8", window=win))
            return {
                "variant": var.name,
                "window": win,
                "hazard": hazard,
                "takeoff": takeoff,
                "dest": snap(st),
                "frames": frames,
            }

        if i < len(var.pre):
            names = var.pre[i]
        elif room == ROOM_CERES_ELEVATOR or gs in (9, 11):
            names = ("A",)
        elif _ceres_grounded(st) and y <= _CERES_FALLING_DOOR_LEDGE_Y + 8:
            if (
                not jumped
                and x <= var.jump_x_hi
                and int(st.momentum_x) >= var.min_momentum
                and int(st.facing) == FACING_LEFT
                and int(st.invincibility_timer) > 0
                and pose not in (137, 138)
            ):
                names = ("LEFT", "B", "A")
                jumped = True
                takeoff = (x, y, int(st.momentum_x), int(st.invincibility_timer))
            else:
                names = _tas_l_pump("LEFT", pump_i, st)
                pump_i += 1
        elif var.air_hold is not None:
            names = var.air_hold
        else:
            names = ("RIGHT",) if 0 < var.turn_x and x <= var.turn_x else ("LEFT",)
            if var.spin:
                names = names + ("B",)
            if y > var.release_y:
                names = names + ("A",)

        if x <= 100 or room == ROOM_CERES_ELEVATOR:
            frames.append(snap(st, act=list(names)))
        session.step(buttons(*names) if names else idle_action(), "probe_door")
    return {
        "variant": var.name,
        "window": False,
        "hazard": hazard,
        "takeoff": takeoff,
        "dest": snap(session.state, event="timeout"),
        "frames": frames,
    }


def build_variants() -> list[Variant]:
    """Crouch out the Ceres door ($E23F), then jump early enough to leave rising."""
    out: list[Variant] = []
    for k in (8, 10, 12):
        for hi in (44, 40, 38, 36, 34, 30):
            out.append(
                Variant(
                    f"c{k}_x{hi}",
                    pre=(("DOWN",),) * k,
                    jump_x_hi=hi,
                    min_momentum=1,
                )
            )
    return out


def main() -> None:
    pin = CERES_DATA_DIR / "ceres_first_control.state"
    from super_metroid.routes.kpdr.ceres.spine import _boot_pin

    env, session = _boot_pin(pin)
    try:
        play_ceres_outbound_to_ridley(session)
        session.span(ActionSpan(("LEFT", "A"), 24, "ceres_ridley_exit"))
        play_ceres_flat_to_scientist(session)
        play_ceres_scientist_to_magnet(session)
        play_ceres_magnet_to_falling(session)

        track = CeresFallingEscapeTrack()
        seat = None
        for _ in range(400):
            st = session.state
            if (
                track.phase == "exit"
                and int(st.room_id) == ROOM_CERES_FALLING
                and int(st.game_state) == 8
                and _ceres_grounded(st)
                and int(st.samus_y) <= _CERES_FALLING_DOOR_LEDGE_Y + 8
            ):
                seat = snap(st)
                break
            names, track = ceres_falling_escape_action(st, track)
            session.step(
                buttons(*names) if names else idle_action(),
                f"ceres_falling_{track.phase}",
            )
        if seat is None:
            raise SystemExit(f"never reached door seat: {snap(session.state)}")
        print("seat", seat, flush=True)

        blob = env.em.get_state()
        runs = []
        for var in build_variants():
            env.em.set_state(blob)
            session.state = parse_state(env.get_ram(), frame=session.frame)
            res = door_leave(session, var)
            d = res["dest"]
            in_band = (
                624 <= d["y"] <= 641
                and d["pose"] in (25, 26, 27, 28)
                and d["vd"] == 1
                and d["vy"] > 0
                and d["inv"] > 0
            )
            res["in_band"] = in_band
            if in_band:
                magnet._ceres_fast_entry_window = _relaxed_window
                try:
                    res["climb_475"] = bool(magnet._ceres_entry_to_475(session))
                finally:
                    magnet._ceres_fast_entry_window = _strict_window
                res["after_climb"] = snap(session.state)
            runs.append(res)
            d = res["dest"]
            print(
                f"{var.name:10s} win={str(res['window']):5s} hz={int(res['hazard'])} "
                f"({d['x']},{d['y']}) p{d['pose']} mx{d['mx']} vy{d['vy']} "
                f"vd{d['vd']} inv{d['inv']}",
                flush=True,
            )
        report = {"seat": seat, "runs": runs}
    finally:
        env.close()
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
