"""Bench the sole Ceres outbound policy vs TAS. Scratch only."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.paths import GAME_DIR
from super_metroid.ram import parse_env_state, probe_pin
from super_metroid.room_timer import RoomTimer, format_segment_time
from super_metroid.routes.kpdr.ceres.outbound import play_ceres_to_ridley_door
from super_metroid.routes.kpdr.ceres.spine import ceres_hops_vs_tas
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_RIDLEY
from super_metroid.routes.runtime import ActionSpan

PIN = GAME_DIR / "scratch" / "ceres_first_control.state"
OUT = GAME_DIR / "scratch" / "ceres_outbound_bench.json"


class _Session:
    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.timer = RoomTimer()

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        self.timer.observe(self.state)
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


def main() -> None:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        session = _Session(env, assist)
        play_ceres_to_ridley_door(session)
        comparison = ceres_hops_vs_tas(
            [visit.to_dict() for visit in session.timer.visits],
            first_control_frame=0,
        )
        report = {
            "success": int(session.state.room_id) == ROOM_CERES_RIDLEY
            and int(session.state.game_state) == 8,
            "frames": session.frame,
            "timing": format_segment_time(session.frame),
            "end": probe_pin(session.state),
            "hops": comparison["hops"],
        }
    finally:
        env.close()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"success={report['success']} frames={report['frames']}")
    for hop in report["hops"]:
        tas = hop.get("tas_frames")
        delta = hop.get("delta_frames")
        print(
            f"{hop.get('name') or hop['from']+'>'+hop['to']:22s} "
            f"our={hop['our_frames']:4d} tas={tas} d={delta:+d}"
            if delta is not None
            else f"{hop.get('name')} our={hop['our_frames']}"
        )


if __name__ == "__main__":
    main()
