"""Scratch: run the product elevator climb off the cached entry snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

from super_metroid.ram import parse_state
from retro_harness.actions import idle_action
from super_metroid.routes.kpdr.ceres.magnet import _ceres_reactive_elev_climb
from super_metroid.routes.kpdr.room_ids import ROOM_LANDING_SITE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry, snap  # noqa: E402


def main() -> None:
    env, session = open_entry()
    try:
        env.em.set_state(ENTRY_STATE.read_bytes())
        session.state = parse_state(env.get_ram(), frame=session.frame)
        start = int(session.frame)
        print("entry", snap(session.state), flush=True)
        _ceres_reactive_elev_climb(session)
        climbed = int(session.frame)
        print("gs32 ", snap(session.state), "climb", climbed - start)
        for reason, n in session.action_reasons.items():
            if reason.startswith("ceres_elev"):
                print(f"  {reason:32s} {n}")
        last = climbed
        seen = None
        for _ in range(4000):
            st = session.state
            if seen != (int(st.room_id), int(st.game_state)):
                seen = (int(st.room_id), int(st.game_state))
                print(f"  +{int(session.frame) - start:5d} room={hex(seen[0])} gs={seen[1]}")
                last = int(session.frame)
            if int(st.room_id) == ROOM_LANDING_SITE and int(st.game_state) == 8:
                break
            session.step(idle_action(), "probe_ride")
        print("to landing gs8", int(session.frame) - start)
        stable = 0
        for _ in range(1200):
            if int(session.state.samus_y) == 1088:
                stable += 1
                if stable >= 30:
                    break
            else:
                stable = 0
            session.step(idle_action(), "probe_settle")
        print("to settle", int(session.frame) - start, "unused", last)
    finally:
        env.close()


if __name__ == "__main__":
    main()
