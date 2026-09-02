"""Scratch: retune the elev entry->475 wall jump off the new Falling door.

Boots first-control once, plays the product reverse to the elevator fast
entry (ROOM_CERES_ELEVATOR + gs 8), snapshots there, then replays candidate
input tapes off that one snapshot. The TAS tape itself is a candidate: the
lsnes slice `sniq_100_ceres_open` covers TAS frames 8639.. so elev_wj
fast_entry 13073 is slice index 4434.
"""

from __future__ import annotations

import json
import sys
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
OUT = HERE / "entry_475_sweep.json"
GAME_DIR = Path(__file__).resolve().parents[2]
SLICE = GAME_DIR / "tas" / "slices" / "sniq_100_ceres_open.json"
FIRST_PAD_FRAME = 8639


def tas_frames() -> list[tuple[str, ...]]:
    doc = json.loads(SLICE.read_text())
    out: list[tuple[str, ...]] = []
    for seg in doc["segments"]:
        for _ in range(int(seg["n"])):
            out.append(tuple(seg["b"]))
    return out


def tas_tape(start_frame: int, count: int) -> list[tuple[str, ...]]:
    frames = tas_frames()
    i = start_frame - FIRST_PAD_FRAME
    return frames[i : i + count]


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


def play_tape(session, tape) -> list[dict]:
    rows = []
    for i, names in enumerate(tape):
        session.step(buttons(*names) if names else idle_action(), "probe_wj")
        rows.append(snap(session.state, f=i + 1, act=" ".join(names) or "-"))
    return rows


def reach_entry(session) -> None:
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


def summarize(rows: list[dict]) -> dict:
    best = min(rows, key=lambda r: r["y"])
    latch = next((r for r in rows if r["pose"] in (131, 132)), None)
    return {
        "min_y": best["y"],
        "min_y_at": best["f"],
        "min_y_x": best["x"],
        "latch": {k: latch[k] for k in ("f", "x", "y", "pose")} if latch else None,
        "end": {k: rows[-1][k] for k in ("f", "x", "y", "pose", "vy", "mt")},
    }


def main() -> None:
    hold_list = [int(v) for v in sys.argv[1:]] or [23]
    env, session = _boot_pin(CERES_DATA_DIR / "ceres_first_control.state")
    runs = []
    try:
        reach_entry(session)
        entry = snap(session.state)
        print("entry", entry, flush=True)
        blob = env.em.get_state()

        head = tas_tape(13074, 7)        # LEFT A / A / LEFT A / A A A / LEFT
        tail = tas_tape(13104, 60)       # RIGHT A ... through plant_475 and past
        for hold in hold_list:
            tape = list(head) + [("LEFT", "A")] * hold + list(tail)
            env.em.set_state(blob)
            session.state = parse_state(env.get_ram(), frame=session.frame)
            rows = play_tape(session, tape)
            summ = summarize(rows)
            summ["hold"] = hold
            runs.append({"hold": hold, "summary": summ, "rows": rows})
            print(
                f"hold={hold:3d} min_y={summ['min_y']:4d}@f{summ['min_y_at']} "
                f"x{summ['min_y_x']} latch={summ['latch']} end={summ['end']}",
                flush=True,
            )
    finally:
        env.close()
    OUT.write_text(json.dumps({"entry": entry, "runs": runs}, indent=1) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
