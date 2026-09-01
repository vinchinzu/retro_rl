#!/usr/bin/env python3
"""Benchmark the main Ceres movement policy against the native lsnes TAS."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.paths import GAME_DIR
from super_metroid.ram import GS_CERES_LEAVE, parse_env_state, probe_pin
from super_metroid.room_timer import RoomTimer, format_segment_time
from super_metroid.routes.kpdr.ceres.outbound import (
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
    play_ceres_to_ridley_door,
)
from super_metroid.routes.kpdr.ceres.spine import ceres_hops_vs_tas
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_RIDLEY, ROOM_LANDING_SITE
from super_metroid.routes.runtime import ActionSpan

PIN = GAME_DIR / "scratch" / "ceres_first_control.state"
OUT = GAME_DIR / "recordings" / "room_timings" / "ceres_vs_tas.json"


class _Session:
    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.timer = RoomTimer()
        self.ceres_leave: dict | None = None

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        self.timer.observe(self.state)
        if self.ceres_leave is None and int(self.state.game_state) in GS_CERES_LEAVE:
            self.ceres_leave = {
                "frame": self.frame,
                "state": probe_pin(self.state),
                "countdown_display": _countdown_display(self.state),
            }
        return self.state

    def span(self, span: ActionSpan) -> None:
        action = buttons(*span.names) if span.names else idle_action()
        for _ in range(span.frames):
            self.step(action, span.reason)

    def spans(self, spans: list[ActionSpan]) -> None:
        for span in spans:
            self.span(span)

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
        start = probe_pin(session.state)
        play_ceres_to_ridley_door(session)
        end = probe_pin(session.state)
        comparison = ceres_hops_vs_tas(
            [visit.to_dict() for visit in session.timer.visits],
            first_control_frame=0,
        )
        hops = comparison["hops"]
        product = sum(int(hop["our_frames"]) for hop in hops)
        tas = sum(int(hop["tas_frames"]) for hop in hops)
        return {
            "success": int(session.state.room_id) == ROOM_CERES_RIDLEY
            and int(session.state.game_state) == 8,
            "frames": session.frame,
            "timing": format_segment_time(session.frame),
            "start": start,
            "end": end,
            "product_hop_frames": product,
            "tas_hop_frames": tas,
            "delta_frames": product - tas,
            "beats_tas": product < tas,
            "hops": hops,
        }
    finally:
        env.close()


def _bcd(value: int) -> int:
    return ((int(value) >> 4) * 10) + (int(value) & 0x0F)


def _countdown_display(state) -> str:
    return (
        f"{_bcd(state.escape_timer_minutes):02d}:"
        f"{_bcd(state.escape_timer_seconds):02d}."
        f"{_bcd(state.escape_timer_frames):02d}"
    )


def _run_full_station() -> dict:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        session = _Session(env, assist)
        play_ceres_outbound_to_ridley(session)
        ridley_countdown_frame = session.frame
        play_ceres_escape_to_landing(session)
        if session.ceres_leave is None:
            raise RuntimeError("Ceres reached Landing without elevator leave")
        return {
            "success": int(session.state.room_id) == ROOM_LANDING_SITE
            and int(session.state.game_state) == 8,
            "ridley_countdown_frame": ridley_countdown_frame,
            "ceres_elevator_exit": session.ceres_leave,
            "landing_settled_frame": session.frame,
            "to_elevator_exit": format_segment_time(session.ceres_leave["frame"]),
            "to_landing_settled": format_segment_time(session.frame),
        }
    finally:
        env.close()


def main() -> None:
    runs = [_run(), _run()]
    exact = runs[0] == runs[1]
    full_runs = [_run_full_station(), _run_full_station()]
    full_exact = full_runs[0] == full_runs[1]
    report = {
        "kind": "super_metroid_ceres_vs_tas",
        "policy": "CERES_SPINE main policy (no legacy fallback)",
        "source": str(PIN),
        "success": all(run["success"] for run in runs),
        "dual_exact": exact,
        "full_station_dual_exact": full_exact,
        "beats_tas": all(run["beats_tas"] for run in runs),
        "runs": runs,
        "full_station_runs": full_runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    first = runs[0]
    flag = "GREEN" if report["success"] and exact else "RED"
    print(
        f"{flag} dual_exact={exact} product={first['product_hop_frames']}f "
        f"tas={first['tas_hop_frames']}f delta={first['delta_frames']:+d}f"
    )
    for hop in first["hops"]:
        print(
            f"{hop['name']:22s} product={hop['our_frames']:4d}f "
            f"tas={hop['tas_frames']:4d}f delta={hop['delta_frames']:+4d}f"
        )
    full = full_runs[0]
    leave = full["ceres_elevator_exit"]
    print(
        f"full_station dual_exact={full_exact} elevator_exit={leave['frame']}f "
        f"clock={full['to_elevator_exit']['clock']} "
        f"countdown={leave['countdown_display']} "
        f"landing={full['landing_settled_frame']}f"
    )
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
