"""Dump / dual reverse Ceres 5 (Flat → Scientist). Scratch only.

Skip Ridley fight tuning. Pin is the product Ridley-leave → Flat gs=8.
Sniq 100% lsnes (gs=8 f11939→door f12036) never presses A: LEFT+B+L/R
across y=139 from (472,139) p18.
"""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from retro_harness.env import write_state_bytes
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.ceres_ridley import (
    CeresRidleyStrategy,
    play_ceres_ridley_fight,
    require_ceres_ridley_countdown,
)
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.hop_glance import final_from_state, grade_final
from super_metroid.leave_specs import LeaveSpec
from super_metroid.ram import parse_env_state, probe_pin
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.ceres.arm_pump import (
    _ceres_arm_pump_until,
    _ceres_clear_knockback,
)
from super_metroid.routes.kpdr.ceres.outbound import play_ceres_flat_to_scientist
from super_metroid.routes.kpdr.ceres.scientist import play_ceres_scientist_to_magnet
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_SCIENTIST,
)
from super_metroid.routes.runtime import ActionSpan
from super_metroid.routes.skills.knockback import is_knockback

GAME_DIR = Path(__file__).resolve().parents[1]
RIDLEY_ENTER = (
    GAME_DIR
    / "custom_integrations"
    / "SuperMetroid-Snes"
    / "scratch"
    / "ceres_ridley_enter.state"
)
PIN = GAME_DIR / "scratch" / "post_ceres_ridley_flat.state"
LEAVE = GAME_DIR / "scratch" / "post_ceres_flat_scientist.state"
SCI_LEAVE = GAME_DIR / "scratch" / "post_ceres_scientist_magnet.state"
MAGNET_LEAVE = GAME_DIR / "scratch" / "post_ceres_magnet_falling.state"
OUT = GAME_DIR / "scratch" / "ceres_escape_dump.json"
DUAL = GAME_DIR / "scratch" / "ceres_flat_to_scientist_dual.json"
SCI_DUAL = GAME_DIR / "scratch" / "ceres_scientist_to_magnet_dual.json"
MAGNET_DUAL = GAME_DIR / "scratch" / "ceres_magnet_to_falling_dual.json"
LEAVE_SPEC = LeaveSpec(
    hop="ceres_flat_to_scientist",
    room=ROOM_CERES_SCIENTIST,
    x=(450, 490),
    y=(120, 160),
    pose_class="any",
)
SCI_LEAVE_SPEC = LeaveSpec(
    hop="ceres_scientist_to_magnet",
    room=ROOM_CERES_MAGNET,
    x=(180, 250),
    y=(360, 420),
    pose_class="any",
)
TAS_GS8 = 259
TAS_DWELL = 97
TAS_SCI_GS8 = 263
TAS_SCI_DWELL = 101


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
    return int(st.room_id) == ROOM_CERES_SCIENTIST and int(st.game_state) == 8


def _product_flat(sess: _Sess) -> None:
    """Pre-this-sitting Flat reverse: LEFT arm-pump, stuck-jump on."""
    if _done(sess.state):
        return
    _ceres_arm_pump_until(
        sess,
        "LEFT",
        reason="ceres_reverse_arm_pump",
        max_frames=700,
        done=lambda s: int(s.room_id) == ROOM_CERES_SCIENTIST
        and int(s.game_state) == 8,
    )


def _sci_done(st) -> bool:
    return int(st.room_id) == ROOM_CERES_MAGNET and int(st.game_state) == 8


def _product_scientist(sess: _Sess) -> None:
    """Pre-this-sitting Scientist reverse: LEFT arm-pump, stuck-jump on."""
    if _sci_done(sess.state):
        return
    _ceres_arm_pump_until(
        sess,
        "LEFT",
        reason="ceres_reverse_arm_pump",
        max_frames=700,
        done=lambda s: int(s.room_id) == ROOM_CERES_MAGNET and int(s.game_state) == 8,
    )


def capture_flat_pin() -> dict:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, RIDLEY_ENTER, settle_frames=0)
        sess = _Sess(env, assist)
        start = probe_pin(sess.state)
        evidence = play_ceres_ridley_fight(sess, strategy=CeresRidleyStrategy())
        require_ceres_ridley_countdown(evidence)
        for _ in range(40):
            if not is_knockback(sess.state):
                break
            sess.step(idle_action(), "ceres_ridley_settle")
        # Product Ridley exit — not tuned this sitting.
        sess.span(ActionSpan(("LEFT", "A"), 24, "ceres_ridley_exit"))
        for _ in range(220):
            st = sess.state
            if int(st.room_id) == ROOM_CERES_FLAT and int(st.game_state) == 8:
                break
            if is_knockback(st):
                _ceres_clear_knockback(sess, "LEFT", reason="ceres_ridley_exit")
                continue
            sess.step(buttons("LEFT", "B"), "ceres_ridley_to_flat")
        else:
            raise TimeoutError(f"Flat ordinary missed after Ridley: {sess.state}")
        write_state_bytes(PIN, env.em.get_state())
        timed = format_segment_time(sess.frame)
        return {
            "success": True,
            "fight_frames": evidence.end_frame - evidence.start_frame,
            "capture_frames": sess.frame,
            "seconds": timed["seconds"],
            "clock": timed["clock"],
            "start": start,
            "end": probe_pin(sess.state),
            "pin": str(PIN),
        }
    finally:
        env.close()


def _run_once(*, write_leave: bool, policy: str) -> dict:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        sess = _Sess(env, assist)
        start = probe_pin(sess.state)
        if policy == "product":
            _product_flat(sess)
        else:
            play_ceres_flat_to_scientist(sess)
        ok = _done(sess.state)
        leave_f = sess.frame
        leave = probe_pin(sess.state)
        misses = grade_final(final_from_state(sess.state), LEAVE_SPEC)
        if ok and write_leave:
            write_state_bytes(LEAVE, env.em.get_state())
        join = None
        if ok:
            try:
                play_ceres_scientist_to_magnet(sess)
                join = {
                    "success": _sci_done(sess.state),
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
            "policy": policy,
            "frames": leave_f,
            "seconds": timed["seconds"],
            "clock": timed["clock"],
            "start": start,
            "end": leave,
            "misses": misses,
            "reasons": sess.reasons,
            "magnet_join": join,
            "leave_state": str(LEAVE) if ok and write_leave else None,
        }
    finally:
        env.close()


def _run_scientist(*, write_leave: bool, policy: str) -> dict:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, LEAVE, settle_frames=0)
        sess = _Sess(env, assist)
        start = probe_pin(sess.state)
        if policy == "product":
            _product_scientist(sess)
        else:
            play_ceres_scientist_to_magnet(sess)
        ok = _sci_done(sess.state)
        leave_f = sess.frame
        leave = probe_pin(sess.state)
        misses = grade_final(final_from_state(sess.state), SCI_LEAVE_SPEC)
        if ok and write_leave:
            write_state_bytes(SCI_LEAVE, env.em.get_state())
        join = None
        if ok:
            try:
                from super_metroid.routes.kpdr.ceres.magnet import (
                    _ceres_reactive_magnet_escape,
                )

                _ceres_reactive_magnet_escape(sess)
                join = {
                    "success": int(sess.state.room_id) == ROOM_CERES_FALLING
                    and int(sess.state.game_state) == 8,
                    "frames": sess.frame - leave_f,
                    "end": probe_pin(sess.state),
                }
                if join["success"] and write_leave:
                    write_state_bytes(MAGNET_LEAVE, env.em.get_state())
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
            "policy": policy,
            "frames": leave_f,
            "seconds": timed["seconds"],
            "clock": timed["clock"],
            "start": start,
            "end": leave,
            "misses": misses,
            "reasons": sess.reasons,
            "falling_join": join,
            "leave_state": str(SCI_LEAVE) if ok and write_leave else None,
        }
    finally:
        env.close()


def main() -> None:
    capture = None
    if not PIN.exists():
        print("capturing Flat pin from Ridley enter (fight not tuned)", flush=True)
        capture = capture_flat_pin()
        print(
            f"pin {PIN.name} {capture['end']} fight={capture['fight_frames']}f "
            f"total={capture['capture_frames']}f",
            flush=True,
        )
    before = _run_once(write_leave=False, policy="product")
    runs = [
        _run_once(write_leave=True, policy="tas"),
        _run_once(write_leave=False, policy="tas"),
    ]
    frames = [int(r["frames"]) for r in runs]
    exact = bool(runs[0]["success"] and runs[1]["success"] and frames[0] == frames[1])
    timed = format_segment_time(frames[0])
    dual = {
        "success": all(r["success"] for r in runs),
        "dual_exact": exact,
        "hop": "ceres_flat_to_scientist",
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
                    "pose": r["end"].get("pose"),
                    "gs": 8 if r["success"] else r["end"].get("game_state"),
                    "dt": r["end"].get("door_transition", 0),
                    "health": r["end"].get("health"),
                },
                "misses": r["misses"],
                "reasons": r["reasons"],
                "magnet_join": r["magnet_join"],
            }
            for r in runs
        ],
    }
    DUAL.write_text(json.dumps(dual, indent=2) + "\n")
    OUT.write_text(
        json.dumps(
            {
                "state": str(PIN),
                "capture": capture,
                "before": {
                    "success": before["success"],
                    "frames": before["frames"],
                    "seconds": before["seconds"],
                    "clock": before["clock"],
                    "end": before["end"],
                    "reasons": before["reasons"],
                    "magnet_join": before["magnet_join"],
                },
                "tas_gs8": TAS_GS8,
                "tas_dwell": TAS_DWELL,
                "dual": dual,
            },
            indent=2,
        )
        + "\n"
    )
    end = runs[0]["end"]
    join = runs[0]["magnet_join"] or {}
    print(
        f"flat before={before['frames']}f {before['clock']} "
        f"after={frames[0]}f/{frames[1]}f dual_exact={exact} {timed['clock']} "
        f"tas={TAS_GS8}f leave 0x{int(end['roomId']):04X} "
        f"({end['x']}, {end['y']}) p{end.get('pose')} misses={runs[0]['misses']} "
        f"sci join ok={join.get('success')} {join.get('frames')}f",
        flush=True,
    )
    sci_before = _run_scientist(write_leave=False, policy="product")
    sci_runs = [
        _run_scientist(write_leave=True, policy="tas"),
        _run_scientist(write_leave=False, policy="tas"),
    ]
    sci_frames = [int(r["frames"]) for r in sci_runs]
    sci_exact = bool(
        sci_runs[0]["success"]
        and sci_runs[1]["success"]
        and sci_frames[0] == sci_frames[1]
    )
    sci_timed = format_segment_time(sci_frames[0])
    sci_dual = {
        "success": all(r["success"] for r in sci_runs),
        "dual_exact": sci_exact,
        "hop": "ceres_scientist_to_magnet",
        "pin": str(LEAVE),
        "leave": str(SCI_LEAVE),
        "frames": sci_frames,
        "timing": sci_timed,
        "hop_glance_misses": sci_runs[0]["misses"],
        "runs": [
            {
                "frames": r["frames"],
                "final": {
                    "room": r["end"]["roomId"],
                    "x": r["end"]["x"],
                    "y": r["end"]["y"],
                    "pose": r["end"].get("pose"),
                    "gs": 8 if r["success"] else r["end"].get("game_state"),
                    "dt": r["end"].get("door_transition", 0),
                    "health": r["end"].get("health"),
                },
                "misses": r["misses"],
                "reasons": r["reasons"],
                "falling_join": r["falling_join"],
            }
            for r in sci_runs
        ],
    }
    SCI_DUAL.write_text(json.dumps(sci_dual, indent=2) + "\n")
    magnet_runs = [r["falling_join"] for r in sci_runs]
    magnet_frames = [int(r["frames"]) for r in magnet_runs]
    magnet_exact = bool(
        all(r["success"] for r in magnet_runs)
        and magnet_frames[0] == magnet_frames[1]
        and magnet_runs[0]["end"] == magnet_runs[1]["end"]
    )
    magnet_dual = {
        "success": all(r["success"] for r in magnet_runs),
        "dual_exact": magnet_exact,
        "hop": "ceres_magnet_to_falling",
        "pin": str(SCI_LEAVE),
        "leave": str(MAGNET_LEAVE),
        "frames": magnet_frames,
        "timing": format_segment_time(magnet_frames[0]),
        "tas_gs8": 331,
        "runs": magnet_runs,
    }
    MAGNET_DUAL.write_text(json.dumps(magnet_dual, indent=2) + "\n")
    payload = json.loads(OUT.read_text())
    payload["scientist"] = {
        "state": str(LEAVE),
        "before": {
            "success": sci_before["success"],
            "frames": sci_before["frames"],
            "seconds": sci_before["seconds"],
            "clock": sci_before["clock"],
            "end": sci_before["end"],
            "reasons": sci_before["reasons"],
            "falling_join": sci_before["falling_join"],
        },
        "tas_gs8": TAS_SCI_GS8,
        "tas_dwell": TAS_SCI_DWELL,
        "dual": sci_dual,
    }
    payload["magnet_escape"] = magnet_dual
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    sci_end = sci_runs[0]["end"]
    sci_join = sci_runs[0]["falling_join"] or {}
    print(
        f"sci before={sci_before['frames']}f {sci_before['clock']} "
        f"after={sci_frames[0]}f/{sci_frames[1]}f dual_exact={sci_exact} "
        f"{sci_timed['clock']} tas={TAS_SCI_GS8}f leave 0x{int(sci_end['roomId']):04X} "
        f"({sci_end['x']}, {sci_end['y']}) p{sci_end.get('pose')} "
        f"misses={sci_runs[0]['misses']} "
        f"falling ok={sci_join.get('success')} {sci_join.get('frames')}f",
        flush=True,
    )


if __name__ == "__main__":
    main()
