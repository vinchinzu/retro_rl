"""Full Ceres station from first-control pin. Scratch only."""

from __future__ import annotations

import json

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.paths import GAME_DIR
from super_metroid.ram import GS_CERES_LEAVE, parse_env_state, probe_pin
from super_metroid.room_timer import RoomTimer, format_segment_time
from super_metroid.routes.kpdr.ceres.outbound import (
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
)
from super_metroid.routes.kpdr.ceres.spine import ceres_hops_vs_tas
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_ELEVATOR, ROOM_LANDING_SITE
from super_metroid.routes.runtime import ActionSpan

PIN = GAME_DIR / "scratch" / "ceres_first_control.state"
OUT = GAME_DIR / "scratch" / "ceres_station_now.json"


def _escape_clock(state) -> dict[str, int | str]:
    """Raw BCD countdown plus the on-screen ``MM'SS\"FF`` rendering."""
    minutes = int(state.escape_timer_minutes)
    seconds = int(state.escape_timer_seconds)
    frames = int(state.escape_timer_frames)
    return {
        "minutes_bcd": minutes,
        "seconds_bcd": seconds,
        "frames_bcd": frames,
        "display": f"{minutes:02X}'{seconds:02X}\"{frames:02X}",
    }


class _Session:
    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.timer = RoomTimer()
        self.ceres_leave = None
        self.countdown_start = None
        self.elevator_platform = None
        self._last_timer_type = int(self.state.timer_type)
        self.ceres_shaft_trace = []

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        self.timer.observe(self.state)
        timer_type = int(self.state.timer_type)
        if (
            self.countdown_start is None
            and timer_type == 6
            and self._last_timer_type != 6
        ):
            self.countdown_start = {
                "frame": self.frame,
                "clock": _escape_clock(self.state),
            }
        self._last_timer_type = timer_type
        if (
            self.elevator_platform is None
            and int(self.state.room_id) == ROOM_CERES_ELEVATOR
            and int(self.state.game_state) == 32
        ):
            self.elevator_platform = {
                "frame": self.frame,
                "clock": _escape_clock(self.state),
                "state": probe_pin(self.state),
            }
        if self.ceres_leave is None and int(self.state.game_state) in GS_CERES_LEAVE:
            self.ceres_leave = {
                "frame": self.frame,
                "clock": _escape_clock(self.state),
                "state": probe_pin(self.state),
            }
        return self.state

    def span(self, span: ActionSpan) -> None:
        action = buttons(*span.names) if span.names else idle_action()
        for _ in range(span.frames):
            self.step(action, span.reason)

    def wait_until(self, predicate, *, timeout: int, reason: str) -> int:
        for waited in range(timeout + 1):
            if predicate(self.state):
                return waited
            self.step(idle_action(), reason)
        raise TimeoutError(f"{reason} timed out at frame {self.frame}: {self.state}")


def _run() -> dict:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        session = _Session(env, assist)
        play_ceres_outbound_to_ridley(session)
        ridley = session.frame
        play_ceres_escape_to_landing(session)
        hops = ceres_hops_vs_tas(
            [visit.to_dict() for visit in session.timer.visits],
            landing_frame=session.frame,
            first_control_frame=0,
        )
        return {
            "success": int(session.state.room_id) == ROOM_LANDING_SITE
            and int(session.state.game_state) == 8,
            "ridley": ridley,
            "landing": session.frame,
            "leave": session.ceres_leave,
            "countdown_start": session.countdown_start,
            "elevator_platform": session.elevator_platform,
            "timing": format_segment_time(session.frame),
            "hops": hops["hops"],
            "trace": session.ceres_shaft_trace,
            "end": probe_pin(session.state),
        }
    finally:
        env.close()


def main() -> None:
    runs = [_run(), _run()]
    report = {
        "success": all(run["success"] for run in runs),
        "dual_exact": runs[0] == runs[1],
        "runs": runs,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    first = runs[0]
    print(
        f"success={report['success']} dual_exact={report['dual_exact']} "
        f"ridley={first['ridley']} landing={first['landing']} "
        f"leave={None if first['leave'] is None else first['leave']['frame']} "
        f"platform_clock={None if first['elevator_platform'] is None else first['elevator_platform']['clock']['display']}"
    )
    for hop in first["hops"]:
        delta = hop.get("delta_frames")
        print(
            f"{str(hop.get('name')):22s} our={hop['our_frames']:4d} "
            f"tas={hop.get('tas_frames')} d={delta:+d}"
            if delta is not None
            else f"{hop.get('name')} our={hop['our_frames']}"
        )
    print("trace", [t.get("side") for t in first["trace"]])


if __name__ == "__main__":
    main()
