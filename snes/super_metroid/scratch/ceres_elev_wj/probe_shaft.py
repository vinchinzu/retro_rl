"""Scratch: map the elevator shaft off the Falling-door entry snapshot.

Boots once, snapshots at ROOM_CERES_ELEVATOR + gs 8, then replays named
input tapes to find the right/left wall x, the reachable apex, and whether
a right-wall wall jump (pose 132) is available at all from this entry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from super_metroid.combat.enemies.scan import list_enemies

HERE = Path(__file__).resolve().parent
OUT = HERE / "shaft_map.json"


@dataclass(frozen=True)
class Variant:
    name: str
    tape: tuple[tuple[str, ...], ...]


def spans(*pairs) -> tuple[tuple[str, ...], ...]:
    out = []
    for names, n in pairs:
        out.extend([tuple(names)] * n)
    return tuple(out)


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


VARIANTS = [
    Variant("hold_A", spans((("A",), 60))),
    Variant("right_A", spans((("RIGHT", "A"), 60))),
    Variant("left_A", spans((("LEFT", "A"), 60))),
    Variant("right_then_left", spans((("RIGHT", "A"), 8), (("LEFT",), 1), (("LEFT", "A"), 40))),
    Variant("right12_left", spans((("RIGHT", "A"), 12), (("LEFT",), 1), (("LEFT", "A"), 40))),
    Variant("right20_left", spans((("RIGHT", "A"), 20), (("LEFT",), 1), (("LEFT", "A"), 40))),
    Variant("right_hold_long", spans((("RIGHT", "A"), 100))),
]


def main() -> None:
    env, session = _boot_pin(CERES_DATA_DIR / "ceres_first_control.state")
    runs = []
    try:
        reach_entry(session)
        entry = snap(session.state)
        enemies = [
            {
                "id": hex(int(e.enemy_id)),
                "x": int(e.x),
                "y": int(e.y),
                "xr": int(e.x_radius),
                "yr": int(e.y_radius),
            }
            for e in list_enemies(session)
        ]
        print("entry", entry, flush=True)
        print("enemies", enemies, flush=True)
        blob = env.em.get_state()
        for var in VARIANTS:
            env.em.set_state(blob)
            session.state = parse_state(env.get_ram(), frame=session.frame)
            rows = []
            for i, names in enumerate(var.tape):
                session.step(buttons(*names) if names else idle_action(), "probe")
                rows.append(snap(session.state, f=i + 1, act=" ".join(names) or "-"))
            top = min(rows, key=lambda r: r["y"])
            xmax = max(r["x"] for r in rows)
            xmin = min(r["x"] for r in rows)
            latch = next((r for r in rows if r["pose"] in (131, 132)), None)
            print(
                f"{var.name:18s} apex y{top['y']}@f{top['f']} x{top['x']} "
                f"xrange[{xmin},{xmax}] latch={latch} end=({rows[-1]['x']},"
                f"{rows[-1]['y']}) p{rows[-1]['pose']}",
                flush=True,
            )
            runs.append({"name": var.name, "rows": rows})
    finally:
        env.close()
    OUT.write_text(
        json.dumps({"entry": entry, "enemies": enemies, "runs": runs}, indent=1) + "\n"
    )
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
