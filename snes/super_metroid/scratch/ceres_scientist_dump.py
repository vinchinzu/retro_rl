"""Dump / dual Ceres 4 (Scientist → Flat) from the magnet-scientist leave pin.

Scratch only. TAS never presses A. Fade already matches (161 vs 161).
"""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from retro_harness.env import write_state_bytes
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.hop_glance import final_from_state, grade_final
from super_metroid.leave_specs import LeaveSpec
from super_metroid.ram import parse_env_state, probe_pin
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.ceres.arm_pump import (
    _ceres_arm_pump_until,
    _ceres_wait_ordinary,
)
from super_metroid.routes.kpdr.ceres.scientist import play_ceres_scientist_to_flat
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_FLAT, ROOM_CERES_RIDLEY
from super_metroid.routes.runtime import ActionSpan

GAME_DIR = Path(__file__).resolve().parents[1]
PIN = GAME_DIR / "scratch" / "post_ceres_magnet_scientist.state"
LEAVE = GAME_DIR / "scratch" / "post_ceres_scientist_flat.state"
OUT = GAME_DIR / "scratch" / "ceres_scientist_dump.json"
DUAL = GAME_DIR / "scratch" / "ceres_scientist_to_flat_dual.json"
LEAVE_SPEC = LeaveSpec(
    hop="ceres_scientist_to_flat",
    room=ROOM_CERES_FLAT,
    x=(20, 60),
    y=(120, 160),
    pose_class="any",
)


class _Sess:
    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.reasons: dict[str, int] = {}

    def step(self, action, reason: str = "") -> None:
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        if reason:
            self.reasons[reason] = self.reasons.get(reason, 0) + 1

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


def _done(st) -> bool:
    return int(st.room_id) == ROOM_CERES_FLAT and int(st.game_state) == 8


def _run_once(*, write_leave: bool) -> dict:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        sess = _Sess(env, assist)
        start = probe_pin(sess.state)
        play_ceres_scientist_to_flat(sess)
        ok = _done(sess.state)
        leave_f = sess.frame
        leave = probe_pin(sess.state)
        misses = grade_final(final_from_state(sess.state), LEAVE_SPEC)
        if ok and write_leave:
            write_state_bytes(LEAVE, env.em.get_state())
        join = None
        if ok:
            try:
                if sess.state.room_id != ROOM_CERES_RIDLEY:
                    _ceres_arm_pump_until(
                        sess,
                        "RIGHT",
                        reason="ceres_out_flat_band",
                        max_frames=900,
                        done=lambda s: s.room_id == ROOM_CERES_RIDLEY,
                        stuck_jump_after=40,
                    )
                _ceres_wait_ordinary(
                    sess, ROOM_CERES_RIDLEY, reason="ceres_ridley_door", timeout=200
                )
                join = {
                    "success": int(sess.state.room_id) == ROOM_CERES_RIDLEY
                    and int(sess.state.game_state) == 8,
                    "frames": sess.frame - leave_f,
                    "end": probe_pin(sess.state),
                }
            except Exception as exc:  # scratch join probe
                join = {
                    "success": False,
                    "frames": sess.frame - leave_f,
                    "error": str(exc),
                    "end": probe_pin(sess.state),
                }
        timed = format_segment_time(leave_f)
        return {
            "success": ok,
            "frames": leave_f,
            "seconds": timed["seconds"],
            "clock": timed["clock"],
            "start": start,
            "end": leave,
            "misses": misses,
            "reasons": sess.reasons,
            "flat_join": join,
            "leave_state": str(LEAVE) if ok and write_leave else None,
        }
    finally:
        env.close()


def main() -> None:
    runs = [_run_once(write_leave=True), _run_once(write_leave=False)]
    frames = [int(r["frames"]) for r in runs]
    exact = bool(runs[0]["success"] and runs[1]["success"] and frames[0] == frames[1])
    timed = format_segment_time(frames[0])
    dual = {
        "success": all(r["success"] for r in runs),
        "dual_exact": exact,
        "hop": "ceres_scientist_to_flat",
        "pin": str(PIN),
        "leave": str(LEAVE),
        "frames": frames,
        "timing": timed,
        "hop_glance_misses": runs[0]["misses"],
        "runs": [
            {
                "frames": r["frames"],
                "final": {
                    "room": r["end"]["roomId"],
                    "x": r["end"]["x"],
                    "y": r["end"]["y"],
                    "pose": r["end"]["pose"],
                    "gs": 8 if r["success"] else r["end"].get("game_state"),
                    "dt": r["end"].get("door_transition", 0),
                    "health": 99,
                },
                "misses": r["misses"],
                "reasons": r["reasons"],
                "flat_join": r["flat_join"],
            }
            for r in runs
        ],
    }
    DUAL.write_text(json.dumps(dual, indent=2) + "\n")
    OUT.write_text(
        json.dumps(
            {
                "state": str(PIN),
                "before_frames": 301,
                "tas_gs8": 264,
                "tas_dwell": 103,
                "dual": dual,
            },
            indent=2,
        )
        + "\n"
    )
    end = runs[0]["end"]
    join = runs[0]["flat_join"] or {}
    print(
        f"dual_exact={exact} {frames[0]}f/{frames[1]}f {timed['clock']} "
        f"leave 0x{int(end['roomId']):04X} ({end['x']}, {end['y']}) p{end['pose']} "
        f"misses={runs[0]['misses']} "
        f"flat ok={join.get('success')} {join.get('frames')}f "
        f"({(join.get('end') or {}).get('x')}, {(join.get('end') or {}).get('y')}) "
        f"p{(join.get('end') or {}).get('pose')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
