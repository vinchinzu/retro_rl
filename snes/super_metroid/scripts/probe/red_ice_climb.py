#!/usr/bin/env python3
"""Verify Red Tower Ice checkpoint edges from the natural Bat handoff.

The default command runs the first edge twice plus an enemy-phase sweep.  It
does not claim the full Red Tower -> Hellway room clear.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.env import write_state_bytes  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.paths import INTEGRATION_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.room_timer import format_segment_time  # noqa: E402
from super_metroid.routes.kpdr.red_tower.red_ice_climb import (  # noqa: E402
    LOWER_RIPPER_1,
    LOWER_RIPPER_2,
    LOWER_RIPPER_3,
    LOWER_RIPPER_4,
    LOW_RIPPER_3_Y,
    LOW_RIPPER_4_Y,
    MID_FLOOR,
    POLICY_ID,
    TUNNEL_FLOOR,
    THIN_SEAT,
    UPPER_RIPPER_1,
    UPPER_RIPPER_1_Y,
    UPPER_RIPPER_2,
    UPPER_RIPPER_2_Y,
    UPPER_RIPPER_3,
    UPPER_RIPPER_3_Y,
    UPPER_RIPPER_4,
    UPPER_RIPPER_4_Y,
    HELLWAY_SILL,
    checkpoint_supported,
    play_bottom_to_ripper1,
    ripper_at_height,
)
from super_metroid.routes.kpdr.red_tower.red_ice_r1_to_r2 import (  # noqa: E402
    POLICY_ID as R12_POLICY,
    play_ripper1_to_ripper2,
)
from super_metroid.routes.kpdr.red_tower.red_ice_r2_to_r3 import (  # noqa: E402
    POLICY_ID as R23_POLICY,
    play_ripper2_to_ripper3,
)
from super_metroid.routes.kpdr.red_tower.red_ice_r3_to_r4 import (  # noqa: E402
    POLICY_ID as R34_POLICY,
    play_ripper3_to_ripper4,
)
from super_metroid.routes.kpdr.red_tower.red_ice_r4_to_tunnel import (  # noqa: E402
    POLICY_ID as R4TUN_POLICY,
    play_ripper4_to_tunnel,
)
from super_metroid.routes.kpdr.red_tower.red_ice_tunnel_to_mid import (  # noqa: E402
    POLICY_ID as TUNMID_POLICY,
    play_tunnel_to_mid_floor,
)
from super_metroid.routes.kpdr.red_tower.red_ice_mid_to_thin import (  # noqa: E402
    POLICY_ID as MIDTHIN_POLICY,
    play_mid_floor_to_thin_seat,
)
from super_metroid.routes.kpdr.red_tower.red_ice_thin_to_ur1 import (  # noqa: E402
    POLICY_ID as THINUR1_POLICY,
    play_thin_seat_to_upper_ripper1,
)
from super_metroid.routes.kpdr.red_tower.red_ice_upper_hops import (  # noqa: E402
    POLICY_ID_UR12 as UR12_POLICY,
    POLICY_ID_UR23 as UR23_POLICY,
    POLICY_ID_UR34 as UR34_POLICY,
    play_upper_ripper1_to_2,
    play_upper_ripper2_to_3,
    play_upper_ripper3_to_4,
)
from super_metroid.routes.kpdr.red_tower.red_ice_ur3_to_hellway import (  # noqa: E402
    POLICY_ID as UR3HW_POLICY,
    play_upper_ripper3_to_hellway,
)
from super_metroid.routes.kpdr.red_tower.red_ice_to_hellway import (  # noqa: E402
    POLICY_ID as ICEHW_POLICY,
    play_ice_climb_to_hellway,
)

DEFAULT_SOURCE = INTEGRATION_DIR / "scratch" / "post_ice_bat_to_red_pure.state"
DEFAULT_OUTPUT = INTEGRATION_DIR / "scratch" / "red_ice_lower_ripper1_pure.state"


class ProbeSession:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.assist = UnlimitedResourcesAssist()
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")

    def step(self, action, reason: str):
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        return self.state


def run_once(source: Path, phase_offset: int, *, save: Path | None = None) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        for _ in range(max(0, int(phase_offset))):
            session.step(idle_action(), "red_ice_phase_perturb")
        start = session.frame
        started_at = time.perf_counter()
        error = None
        try:
            play_bottom_to_ripper1(session)
        except Exception as exc:  # report the full sweep instead of stopping early
            error = str(exc)
        enemy = ripper_at_height(env, 2376)
        policy_frames = int(session.frame - start)
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        green = error is None and checkpoint_supported(env, session.state, LOWER_RIPPER_1)
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "phaseOffset": int(phase_offset),
            "policyFrames": policy_frames,
            "fps": round(policy_frames / elapsed, 1),
            "totalFrames": int(session.frame),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "freezeTimer": int(enemy.freeze_timer) if enemy is not None else 0,
            "enemyX": int(enemy.x) if enemy is not None else None,
            "error": error,
        }
    finally:
        env.close()


def _offsets(value: str) -> list[int]:
    if value == "full":
        return list(range(0, 241, 8))
    return [int(part.strip(), 0) for part in value.split(",") if part.strip()]


def run_r12(source: Path, *, save: Path | None = None) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        start = session.frame
        started_at = time.perf_counter()
        error = None
        try:
            play_ripper1_to_ripper2(session)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        enemy = ripper_at_height(env, 2280)
        policy_frames = int(session.frame - start)
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        green = error is None and checkpoint_supported(env, session.state, LOWER_RIPPER_2)
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "policy": R12_POLICY,
            "policyFrames": policy_frames,
            "fps": round(policy_frames / elapsed, 1),
            "totalFrames": int(session.frame),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "freezeTimer": int(enemy.freeze_timer) if enemy is not None else 0,
            "enemyX": int(enemy.x) if enemy is not None else None,
            "error": error,
        }
    finally:
        env.close()


def run_r23(source: Path, *, save: Path | None = None) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        start = session.frame
        started_at = time.perf_counter()
        error = None
        try:
            play_ripper2_to_ripper3(session)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        enemy = ripper_at_height(env, LOW_RIPPER_3_Y)
        policy_frames = int(session.frame - start)
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        green = error is None and checkpoint_supported(env, session.state, LOWER_RIPPER_3)
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "policy": R23_POLICY,
            "policyFrames": policy_frames,
            "fps": round(policy_frames / elapsed, 1),
            "totalFrames": int(session.frame),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "freezeTimer": int(enemy.freeze_timer) if enemy is not None else 0,
            "enemyX": int(enemy.x) if enemy is not None else None,
            "error": error,
        }
    finally:
        env.close()


def run_r34(source: Path, *, save: Path | None = None) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        start = session.frame
        started_at = time.perf_counter()
        error = None
        try:
            play_ripper3_to_ripper4(session)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        enemy = ripper_at_height(env, LOW_RIPPER_4_Y)
        policy_frames = int(session.frame - start)
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        green = error is None and checkpoint_supported(env, session.state, LOWER_RIPPER_4)
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "policy": R34_POLICY,
            "policyFrames": policy_frames,
            "fps": round(policy_frames / elapsed, 1),
            "totalFrames": int(session.frame),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "freezeTimer": int(enemy.freeze_timer) if enemy is not None else 0,
            "enemyX": int(enemy.x) if enemy is not None else None,
            "error": error,
        }
    finally:
        env.close()


def run_r4tun(source: Path, *, save: Path | None = None) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        start = session.frame
        started_at = time.perf_counter()
        error = None
        try:
            play_ripper4_to_tunnel(session)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        policy_frames = int(session.frame - start)
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        timed = format_segment_time(policy_frames)
        green = error is None and TUNNEL_FLOOR.matches(session.state)
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "policy": R4TUN_POLICY,
            "policyFrames": policy_frames,
            "time": timed,
            "fps": round(policy_frames / elapsed, 1),
            "totalFrames": int(session.frame),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "error": error,
        }
    finally:
        env.close()


def run_tunmid(source: Path, *, save: Path | None = None) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        start = session.frame
        error = None
        try:
            play_tunnel_to_mid_floor(session)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        policy_frames = int(session.frame - start)
        green = error is None and MID_FLOOR.matches(session.state)
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "policy": TUNMID_POLICY,
            "policyFrames": policy_frames,
            "time": format_segment_time(policy_frames),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "error": error,
        }
    finally:
        env.close()


def run_supported(
    source: Path,
    play_fn,
    checkpoint,
    policy: str,
    *,
    save: Path | None = None,
    enemy_y: int | None = None,
    accept_rooms: tuple[int, ...] = (),
) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        start = session.frame
        error = None
        try:
            play_fn(session)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        enemy = ripper_at_height(env, enemy_y) if enemy_y is not None else None
        policy_frames = int(session.frame - start)
        green = error is None and (
            checkpoint_supported(env, session.state, checkpoint)
            or int(session.state.room_id) in accept_rooms
        )
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "policy": policy,
            "policyFrames": policy_frames,
            "time": format_segment_time(policy_frames),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "freezeTimer": int(enemy.freeze_timer) if enemy is not None else 0,
            "enemyX": int(enemy.x) if enemy is not None else None,
            "error": error,
        }
    finally:
        env.close()


def _print_dual(runs: list[dict[str, Any]], policy: str) -> int:
    green = all(row["green"] for row in runs)
    dual_exact = (
        runs[0]["policyFrames"] == runs[1]["policyFrames"]
        and runs[0]["xy"] == runs[1]["xy"]
        and runs[0]["pose"] == runs[1]["pose"]
    )
    timed = runs[0]["time"]
    print(
        f"{'GREEN' if green else 'RED'} {policy} dual={dual_exact} "
        f"frames={timed['frames']} seconds={timed['seconds']} "
        f"clock={timed['clock']} xy={runs[0]['xy']} p={runs[0]['pose']} "
        f"err={runs[0]['error']}"
    )
    return 0 if green else 1


def run_thinur1(source: Path, *, save: Path | None = None) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        start = session.frame
        error = None
        try:
            play_thin_seat_to_upper_ripper1(session)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        enemy = ripper_at_height(env, UPPER_RIPPER_1_Y)
        policy_frames = int(session.frame - start)
        green = error is None and checkpoint_supported(env, session.state, UPPER_RIPPER_1)
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "policy": THINUR1_POLICY,
            "policyFrames": policy_frames,
            "time": format_segment_time(policy_frames),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "freezeTimer": int(enemy.freeze_timer) if enemy is not None else 0,
            "enemyX": int(enemy.x) if enemy is not None else None,
            "error": error,
        }
    finally:
        env.close()


def run_midthin(source: Path, *, save: Path | None = None) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        start = session.frame
        error = None
        try:
            play_mid_floor_to_thin_seat(session)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        policy_frames = int(session.frame - start)
        green = error is None and THIN_SEAT.matches(session.state)
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "policy": MIDTHIN_POLICY,
            "policyFrames": policy_frames,
            "time": format_segment_time(policy_frames),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "error": error,
        }
    finally:
        env.close()


def run_chain(source: Path, *, save: Path | None = None) -> dict[str, Any]:
    env = make_dev_env()
    try:
        boot_from_state(env, source, settle_frames=0)
        session = ProbeSession(env)
        start = session.frame
        started_at = time.perf_counter()
        error = None
        try:
            play_ice_climb_to_hellway(session)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        policy_frames = int(session.frame - start)
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        timed = format_segment_time(policy_frames)
        green = error is None and HELLWAY_SILL.matches(session.state)
        if green and save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(save, env.em.get_state())
        return {
            "green": green,
            "policy": ICEHW_POLICY,
            "policyFrames": policy_frames,
            "time": timed,
            "fps": round(policy_frames / elapsed, 1),
            "totalFrames": int(session.frame),
            "room": f"0x{int(session.state.room_id):04X}",
            "xy": [int(session.state.samus_x), int(session.state.samus_y)],
            "pose": int(session.state.pose),
            "error": error,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--edge",
        choices=("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "chain"),
        default="1",
        help="1=bottom→r1 … 11=ur3→ur4, 12=ur3→hellway, chain=bottom→hellway_sill",
    )
    parser.add_argument(
        "--phase-offsets",
        default="full",
        help="comma-separated idle offsets, or 'full' for 0..240 step 8",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=DEFAULT_OUTPUT,
        type=Path,
        help="save the offset-0 checkpoint state (default scratch path)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.edge == "2":
        runs = [run_r12(args.source, save=args.save), run_r12(args.source)]
        green = all(row["green"] for row in runs)
        dual_exact = runs[0]["policyFrames"] == runs[1]["policyFrames"] and runs[0]["xy"] == runs[1]["xy"]
        report = {
            "policy": R12_POLICY,
            "scope": "lower_ripper_1->lower_ripper_2",
            "green": green,
            "dualExact": dual_exact,
            "runs": runs,
            "nonClaim": "Hellway exit is not implemented by this checkpoint edge",
        }
        mark = "GREEN" if green else "RED"
        print(
            f"{mark} {R12_POLICY} dual={dual_exact} "
            f"frames={runs[0]['policyFrames']} xy={runs[0]['xy']} p={runs[0]['pose']}"
        )
        if args.json:
            print(json.dumps(report, indent=2))
        return 0 if green else 1
    if args.edge == "3":
        runs = [run_r23(args.source, save=args.save), run_r23(args.source)]
        green = all(row["green"] for row in runs)
        dual_exact = runs[0]["policyFrames"] == runs[1]["policyFrames"] and runs[0]["xy"] == runs[1]["xy"]
        report = {
            "policy": R23_POLICY,
            "scope": "lower_ripper_2->lower_ripper_3",
            "green": green,
            "dualExact": dual_exact,
            "runs": runs,
            "nonClaim": "Hellway exit is not implemented by this checkpoint edge",
        }
        mark = "GREEN" if green else "RED"
        print(
            f"{mark} {R23_POLICY} dual={dual_exact} "
            f"frames={runs[0]['policyFrames']} xy={runs[0]['xy']} p={runs[0]['pose']}"
            f" err={runs[0]['error']}"
        )
        if args.json:
            print(json.dumps(report, indent=2))
        return 0 if green else 1
    if args.edge == "4":
        runs = [run_r34(args.source, save=args.save), run_r34(args.source)]
        green = all(row["green"] for row in runs)
        dual_exact = runs[0]["policyFrames"] == runs[1]["policyFrames"] and runs[0]["xy"] == runs[1]["xy"]
        report = {
            "policy": R34_POLICY,
            "scope": "lower_ripper_3->lower_ripper_4",
            "green": green,
            "dualExact": dual_exact,
            "runs": runs,
            "nonClaim": "Hellway exit is not implemented by this checkpoint edge",
        }
        mark = "GREEN" if green else "RED"
        print(
            f"{mark} {R34_POLICY} dual={dual_exact} "
            f"frames={runs[0]['policyFrames']} xy={runs[0]['xy']} p={runs[0]['pose']}"
            f" err={runs[0]['error']}"
        )
        if args.json:
            print(json.dumps(report, indent=2))
        return 0 if green else 1
    if args.edge == "5":
        runs = [run_r4tun(args.source, save=args.save), run_r4tun(args.source)]
        green = all(row["green"] for row in runs)
        dual_exact = (
            runs[0]["policyFrames"] == runs[1]["policyFrames"]
            and runs[0]["xy"] == runs[1]["xy"]
        )
        report = {
            "policy": R4TUN_POLICY,
            "scope": "lower_ripper_4->tunnel_floor",
            "green": green,
            "dualExact": dual_exact,
            "runs": runs,
            "nonClaim": "Hellway exit is not implemented by this checkpoint edge",
        }
        mark = "GREEN" if green else "RED"
        timed = runs[0].get("time") or format_segment_time(runs[0]["policyFrames"])
        print(
            f"{mark} {R4TUN_POLICY} dual={dual_exact} "
            f"frames={timed['frames']} seconds={timed['seconds']} "
            f"clock={timed['clock']} xy={runs[0]['xy']} p={runs[0]['pose']}"
            f" err={runs[0]['error']}"
        )
        if args.json:
            print(json.dumps(report, indent=2))
        return 0 if green else 1
    if args.edge == "6":
        runs = [run_tunmid(args.source, save=args.save), run_tunmid(args.source)]
        green = all(row["green"] for row in runs)
        dual_exact = (
            runs[0]["policyFrames"] == runs[1]["policyFrames"]
            and runs[0]["xy"] == runs[1]["xy"]
            and runs[0]["pose"] == runs[1]["pose"]
        )
        timed = runs[0]["time"]
        print(
            f"{'GREEN' if green else 'RED'} {TUNMID_POLICY} dual={dual_exact} "
            f"frames={timed['frames']} seconds={timed['seconds']} "
            f"clock={timed['clock']} xy={runs[0]['xy']} p={runs[0]['pose']} "
            f"err={runs[0]['error']}"
        )
        if args.json:
            print(json.dumps({"green": green, "dualExact": dual_exact, "runs": runs}, indent=2))
        return 0 if green else 1
    if args.edge == "7":
        runs = [run_midthin(args.source, save=args.save), run_midthin(args.source)]
        green = all(row["green"] for row in runs)
        dual_exact = (
            runs[0]["policyFrames"] == runs[1]["policyFrames"]
            and runs[0]["xy"] == runs[1]["xy"]
            and runs[0]["pose"] == runs[1]["pose"]
        )
        timed = runs[0]["time"]
        print(
            f"{'GREEN' if green else 'RED'} {MIDTHIN_POLICY} dual={dual_exact} "
            f"frames={timed['frames']} seconds={timed['seconds']} "
            f"clock={timed['clock']} xy={runs[0]['xy']} p={runs[0]['pose']} "
            f"err={runs[0]['error']}"
        )
        if args.json:
            print(json.dumps({"green": green, "dualExact": dual_exact, "runs": runs}, indent=2))
        return 0 if green else 1
    if args.edge == "8":
        runs = [run_thinur1(args.source, save=args.save), run_thinur1(args.source)]
        code = _print_dual(runs, THINUR1_POLICY)
        if args.json:
            print(json.dumps({"green": code == 0, "runs": runs}, indent=2))
        return code
    if args.edge == "9":
        runs = [
            run_supported(
                args.source,
                play_upper_ripper1_to_2,
                UPPER_RIPPER_2,
                UR12_POLICY,
                save=args.save,
                enemy_y=UPPER_RIPPER_2_Y,
            ),
            run_supported(
                args.source,
                play_upper_ripper1_to_2,
                UPPER_RIPPER_2,
                UR12_POLICY,
                enemy_y=UPPER_RIPPER_2_Y,
            ),
        ]
        code = _print_dual(runs, UR12_POLICY)
        if args.json:
            print(json.dumps({"green": code == 0, "runs": runs}, indent=2))
        return code
    if args.edge == "10":
        runs = [
            run_supported(
                args.source,
                play_upper_ripper2_to_3,
                UPPER_RIPPER_3,
                UR23_POLICY,
                save=args.save,
                enemy_y=UPPER_RIPPER_3_Y,
            ),
            run_supported(
                args.source,
                play_upper_ripper2_to_3,
                UPPER_RIPPER_3,
                UR23_POLICY,
                enemy_y=UPPER_RIPPER_3_Y,
            ),
        ]
        code = _print_dual(runs, UR23_POLICY)
        if args.json:
            print(json.dumps({"green": code == 0, "runs": runs}, indent=2))
        return code
    if args.edge == "11":
        runs = [
            run_supported(
                args.source,
                play_upper_ripper3_to_4,
                UPPER_RIPPER_4,
                UR34_POLICY,
                save=args.save,
                enemy_y=UPPER_RIPPER_4_Y,
            ),
            run_supported(
                args.source,
                play_upper_ripper3_to_4,
                UPPER_RIPPER_4,
                UR34_POLICY,
                enemy_y=UPPER_RIPPER_4_Y,
            ),
        ]
        code = _print_dual(runs, UR34_POLICY)
        if args.json:
            print(json.dumps({"green": code == 0, "runs": runs}, indent=2))
        return code
    if args.edge == "12":
        runs = [
            run_supported(
                args.source,
                play_upper_ripper3_to_hellway,
                HELLWAY_SILL,
                UR3HW_POLICY,
                save=args.save,
            ),
            run_supported(
                args.source,
                play_upper_ripper3_to_hellway,
                HELLWAY_SILL,
                UR3HW_POLICY,
            ),
        ]
        code = _print_dual(runs, UR3HW_POLICY)
        if args.json:
            print(json.dumps({"green": code == 0, "runs": runs}, indent=2))
        return code
    if args.edge == "chain":
        row = run_chain(args.source, save=args.save)
        mark = "GREEN" if row["green"] else "RED"
        timed = row.get("time") or format_segment_time(row["policyFrames"])
        print(
            f"{mark} chain frames={timed['frames']} seconds={timed['seconds']} "
            f"clock={timed['clock']} xy={row['xy']} "
            f"p={row['pose']} err={row['error']}"
        )
        if args.json:
            print(json.dumps(row, indent=2))
        print("  chain through ordinary Hellway left-door; play_red_to_hellway uses this climb")
        return 0 if row["green"] else 1

    offsets = _offsets(args.phase_offsets)
    # Two independent offset-zero boots are the exact dual check.  The sweep
    # follows, excluding duplicate zero in the compact report.
    runs = [run_once(args.source, 0, save=args.save), run_once(args.source, 0)]
    runs.extend(run_once(args.source, value) for value in offsets if value != 0)
    green = all(row["green"] for row in runs)
    exact_fields = ("green", "policyFrames", "room", "xy", "pose", "freezeTimer", "enemyX", "error")
    dual_exact = all(runs[0][field] == runs[1][field] for field in exact_fields)
    report = {
        "policy": POLICY_ID,
        "scope": "bottom_floor->lower_ripper_1",
        "green": green,
        "dualExact": dual_exact,
        "runs": runs,
        "policyFrames": {
            "min": min(row["policyFrames"] for row in runs),
            "max": max(row["policyFrames"] for row in runs),
        },
        "fps": {
            "min": min(row["fps"] for row in runs),
            "max": max(row["fps"] for row in runs),
        },
        "saved": str(args.save) if args.save is not None and runs[0]["green"] else None,
        "nonClaim": "Hellway exit is not implemented by this checkpoint edge",
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        mark = "GREEN" if green else "RED"
        print(
            f"{mark} {POLICY_ID} dual={report['dualExact']} "
            f"runs={len(runs)} policy_frames="
            f"{report['policyFrames']['min']}..{report['policyFrames']['max']} "
            f"fps={report['fps']['min']:.0f}..{report['fps']['max']:.0f}"
        )
        for row in runs:
            if not row["green"]:
                print(f"  RED offset={row['phaseOffset']}: {row['error']} {row['xy']}")
        if report["saved"]:
            print(f"  checkpoint -> {report['saved']}")
        print("  partial only: lower_ripper_1; Hellway remains RED")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
