"""Re-phase Ceres elev WJ after Ridley fresh_fifth_jump. Scratch only."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from retro_harness.env import write_state_bytes
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.enemies.scan import enemies_from_ram
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.ram import parse_env_state, probe_pin
from super_metroid.routes.controller_common import is_wall_latch
from super_metroid.routes.kpdr.ceres.elev_escape import (
    _CERES_475_TO_363_WALLJUMP,
    _CERES_FAST_ENTRY_WALLJUMP,
    _ceres_475_to_363,
    _ceres_any_wall_latch,
    _ceres_at_checkpoint,
    _ceres_checkpoint_hop,
    _ceres_fast_entry_window,
    _ceres_planted_at,
)
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_ELEVATOR
from super_metroid.routes.runtime import ActionSpan, hold
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.routes.skills.walljump import (
    PreciseWallJumpTiming,
    precise_walljump_once,
)

GAME_DIR = Path(__file__).resolve().parents[1]
PIN = GAME_DIR / "scratch" / "ceres_elev_fast_window.state"
PIN_475 = GAME_DIR / "scratch" / "ceres_elev_475_scan.state"
OUT = GAME_DIR / "scratch" / "ceres_elev_rephase.json"

CHECKPOINTS = (571, 475, 363, 267, 171)


class _Sess:
    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
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


def _snap(st, ram=None) -> dict:
    enemies = []
    if ram is not None:
        enemies = [
            {"slot": e.slot, "id": e.enemy_id, "x": e.x, "y": e.y, "hp": e.hp}
            for e in enemies_from_ram(ram)
            if abs(e.x - int(st.samus_x)) < 80 and abs(e.y - int(st.samus_y)) < 120
        ]
    return {
        "pin": probe_pin(st),
        "x": int(st.samus_x),
        "y": int(st.samus_y),
        "pose": int(st.pose),
        "vy": int(st.velocity_y),
        "mom": int(st.momentum_x),
        "gs": int(st.game_state),
        "room": f"0x{int(st.room_id):04X}",
        "kb": int(is_knockback(st)),
        "health": int(st.health),
        "fast": bool(_ceres_fast_entry_window(st)),
        "near_enemies": enemies,
    }


def _boot(path: Path) -> tuple[object, UnlimitedResourcesAssist, _Sess]:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    assist.attach_env(env)
    boot_from_state(env, path, settle_frames=0)
    return env, assist, _Sess(env, assist)


def _reload(sess: _Sess, path: Path) -> None:
    boot_from_state(sess.env, path, settle_frames=0)
    sess.frame = 0
    sess.state = parse_env_state(sess.env, mode="nav")


def _close(env) -> None:
    env.close()


def inspect_pin(path: Path) -> dict:
    env, _assist, sess = _boot(path)
    try:
        return _snap(sess.state, sess.env.get_ram())
    finally:
        _close(env)


def _seat(st) -> int | None:
    for y in CHECKPOINTS:
        if _ceres_at_checkpoint(st, y):
            return y
    return None


def _run_wj(
    sess: _Sess,
    path: Path,
    timing: PreciseWallJumpTiming,
    *,
    delay: int = 0,
    hold_a: bool = True,
    success_knockback: bool = False,
) -> dict:
    _reload(sess, path)
    start = _snap(sess.state, sess.env.get_ram())
    for _ in range(delay):
        names = ("A",) if hold_a else ()
        sess.step(buttons(*names) if names else idle_action(), "delay")
    delayed = _snap(sess.state, sess.env.get_ram())
    miss = None

    def _success(st) -> bool:
        if _ceres_planted_at(st, 475):
            return True
        return bool(success_knockback and _ceres_at_checkpoint(st, 475))

    try:
        precise_walljump_once(
            sess,
            timing,
            start_when=None,
            contact_when=is_wall_latch,
            success_when=_success,
            landing_timeout=48,
            reason="rephase_wj",
        )
    except RuntimeError as exc:
        miss = str(exc)
    end = sess.state
    return {
        "delay": delay,
        "hold_a": hold_a,
        "coast": timing.coast_frames,
        "into": timing.into_frames,
        "start": start,
        "after_delay": delayed,
        "end": _snap(end, sess.env.get_ram()),
        "planted_475": bool(_ceres_planted_at(end, 475)),
        "checkpoint": _seat(end),
        "miss": miss,
    }


def scan_delay_wj(sess: _Sess, path: Path, delays: range) -> list[dict]:
    hits = []
    for delay in delays:
        row = _run_wj(sess, path, _CERES_FAST_ENTRY_WALLJUMP, delay=delay)
        compact = {
            "delay": delay,
            "after_y": row["after_delay"]["y"],
            "after_pose": row["after_delay"]["pose"],
            "end_x": row["end"]["x"],
            "end_y": row["end"]["y"],
            "end_pose": row["end"]["pose"],
            "kb": row["end"]["kb"],
            "planted_475": row["planted_475"],
            "checkpoint": row["checkpoint"],
            "miss": row["miss"],
        }
        hits.append(compact)
        if row["planted_475"] or row["checkpoint"] in (475, 363, 267, 171):
            break
    return hits


def scan_short_wj(sess: _Sess, path: Path) -> list[dict]:
    """Immediate / short-coast WJ — y=637 has less air than y=628."""
    rows = []
    for coast in (0, 2, 4, 6, 8, 10):
        for into in (4, 8, 12, 16):
            timing = PreciseWallJumpTiming(
                into="RIGHT",
                away="LEFT",
                coast_frames=coast,
                into_frames=into,
                release_frames=2,
                jump_frames=36,
            )
            row = _run_wj(sess, path, timing)
            compact = {
                "coast": coast,
                "into": into,
                "end_x": row["end"]["x"],
                "end_y": row["end"]["y"],
                "end_pose": row["end"]["pose"],
                "kb": row["end"]["kb"],
                "planted_475": row["planted_475"],
                "checkpoint": row["checkpoint"],
            }
            rows.append(compact)
            if row["planted_475"] or row["checkpoint"] in (475, 363, 267, 171):
                return rows
    return rows


def walk_then_475_wj(
    sess: _Sess, path: Path, *, target_x: int, idle: int
) -> dict:
    _reload(sess, path)
    start = _snap(sess.state, sess.env.get_ram())
    for _ in range(idle):
        sess.step(idle_action(), "idle")
    for _ in range(40):
        x = int(sess.state.samus_x)
        if abs(x - target_x) <= 2:
            break
        name = "RIGHT" if x < target_x else "LEFT"
        sess.step(buttons(name), "walk")
    walked = _snap(sess.state, sess.env.get_ram())
    planted = bool(_ceres_planted_at(sess.state, 475))
    reached_363 = False
    miss = None
    if planted:
        try:
            reached_363 = _ceres_475_to_363(sess)
        except Exception as exc:  # noqa: BLE001
            miss = str(exc)
            reached_363 = _ceres_planted_at(sess.state, 363)
    end = sess.state
    return {
        "target_x": target_x,
        "idle": idle,
        "start": start,
        "walked": walked,
        "end": _snap(end, sess.env.get_ram()),
        "planted_start": planted,
        "reached_363": reached_363,
        "checkpoint": _seat(end),
        "miss": miss,
    }


def scan_475_left_wj(sess: _Sess, path: Path) -> list[dict]:
    rows = []
    for idle in (0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64):
        for target_x in (123, 140, 155, 163, 175):
            row = walk_then_475_wj(sess, path, target_x=target_x, idle=idle)
            compact = {
                "idle": idle,
                "target_x": target_x,
                "walk_x": row["walked"]["x"],
                "walk_y": row["walked"]["y"],
                "walk_pose": row["walked"]["pose"],
                "end_x": row["end"]["x"],
                "end_y": row["end"]["y"],
                "end_pose": row["end"]["pose"],
                "kb": row["end"]["kb"],
                "reached_363": row["reached_363"],
                "checkpoint": row["checkpoint"],
            }
            rows.append(compact)
            if row["reached_363"]:
                return rows
    return rows


def scan_apex_wj(sess: _Sess, path: Path) -> list[dict]:
    """Hold A to the y≈620 apex, then a short right-wall jump over steam."""
    rows = []
    for delay in (3, 4, 5, 6, 7, 8):
        for coast in (0, 1, 2, 4):
            for into in (2, 4, 6, 8, 12):
                timing = PreciseWallJumpTiming(
                    into="RIGHT",
                    away="LEFT",
                    coast_frames=coast,
                    into_frames=into,
                    release_frames=2,
                    jump_frames=40,
                )
                row = _run_wj(
                    sess,
                    path,
                    timing,
                    delay=delay,
                    success_knockback=True,
                )
                compact = {
                    "delay": delay,
                    "coast": coast,
                    "into": into,
                    "after_y": row["after_delay"]["y"],
                    "end_x": row["end"]["x"],
                    "end_y": row["end"]["y"],
                    "end_pose": row["end"]["pose"],
                    "kb": row["end"]["kb"],
                    "min_hint": row["end"]["y"],
                    "planted_475": row["planted_475"],
                    "checkpoint": row["checkpoint"],
                    "miss": None if row["planted_475"] or row["checkpoint"] else (
                        "contact" if row["miss"] and "contact" in row["miss"] else "outcome"
                    ),
                }
                rows.append(compact)
                if row["planted_475"] or row["checkpoint"] in (475, 363, 267, 171):
                    return rows
    return rows


def trace_wj_frames(path: Path, *, delay: int, timing: PreciseWallJumpTiming) -> list[dict]:
    env, _assist, sess = _boot(path)
    try:
        for _ in range(delay):
            sess.step(buttons("A"), "delay")
        frames = []
        min_y = int(sess.state.samus_y)
        try:
            def _contact(st):
                return is_wall_latch(st) or _ceres_any_wall_latch(st)

            def _success(st):
                return _ceres_planted_at(st, 475) or _ceres_at_checkpoint(st, 475)

            # Manual phases so we can sample every frame.
            phases = (
                (timing.coast_frames, timing.coast_buttons, "coast"),
                (timing.into_frames, (timing.into, *timing.approach_buttons), "into"),
                (timing.release_frames, (timing.away,), "rel"),
                (timing.jump_frames, (timing.away, *timing.jump_buttons), "jump"),
                (48, (), "land"),
            )
            done = False
            for n, names, label in phases:
                for _ in range(n):
                    sess.step(buttons(*names) if names else idle_action(), label)
                    st = sess.state
                    y = int(st.samus_y)
                    min_y = min(min_y, y)
                    row = {
                        "f": sess.frame,
                        "ph": label,
                        "x": int(st.samus_x),
                        "y": y,
                        "pose": int(st.pose),
                        "vy": int(st.velocity_y),
                        "vd": int(st.vertical_direction),
                        "kb": int(is_knockback(st)),
                        "latch": int(_ceres_any_wall_latch(st)),
                    }
                    if sess.frame % 2 == 0 or row["latch"] or row["kb"] or y <= 480:
                        frames.append(row)
                    if _success(st):
                        done = True
                        break
                if done:
                    break
        finally:
            frames.append({"min_y": min_y, "end": _snap(sess.state)})
        return frames
    finally:
        _close(env)


def scan_475_right_hop(sess: _Sess, path: Path) -> list[dict]:
    rows = []
    for idle in range(0, 81, 4):
        _reload(sess, path)
        for _ in range(idle):
            sess.step(idle_action(), "idle")
        ok = _ceres_checkpoint_hop(
            sess, side="RIGHT", runup=4, target_y=363
        )
        end = sess.state
        compact = {
            "idle": idle,
            "ok": bool(ok),
            "end_x": int(end.samus_x),
            "end_y": int(end.samus_y),
            "end_pose": int(end.pose),
            "kb": int(is_knockback(end)),
            "checkpoint": _seat(end),
            "planted_363": bool(_ceres_planted_at(end, 363)),
        }
        rows.append(compact)
        if compact["planted_363"] or compact["checkpoint"] in (363, 267, 171):
            return rows
    return rows


def scan_475_idle_window(sess: _Sess, path: Path) -> list[dict]:
    rows = []
    for idle in range(28, 49):
        _reload(sess, path)
        for _ in range(idle):
            sess.step(idle_action(), "idle")
        ok = _ceres_checkpoint_hop(sess, side="RIGHT", runup=4, target_y=363)
        end = sess.state
        rows.append(
            {
                "idle": idle,
                "ok": bool(ok),
                "end_x": int(end.samus_x),
                "end_y": int(end.samus_y),
                "end_pose": int(end.pose),
                "kb": int(is_knockback(end)),
                "planted_363": bool(_ceres_planted_at(end, 363)),
            }
        )
    return rows


def climb_from_475(path: Path, *, idle: int) -> dict:
    from super_metroid.routes.kpdr.ceres.elev_escape import (
        _ceres_checkpoint_shaft,
        _ceres_elev_leaving,
        _ceres_elev_top_to_ship,
    )

    env, _assist, sess = _boot(path)
    try:
        for _ in range(idle):
            sess.step(idle_action(), "idle")
        hopped = _ceres_checkpoint_hop(sess, side="RIGHT", runup=4, target_y=363)
        after_hop = _snap(sess.state)
        shaft_ok = False
        ship_ok = False
        err = None
        if hopped:
            try:
                shaft_ok = _ceres_checkpoint_shaft(sess)
                _ceres_elev_top_to_ship(sess)
                ship_ok = bool(_ceres_elev_leaving(sess.state))
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
        return {
            "idle": idle,
            "hopped": bool(hopped),
            "after_hop": after_hop,
            "shaft_ok": shaft_ok,
            "ship_ok": ship_ok,
            "end": _snap(sess.state),
            "frames": sess.frame,
            "err": err,
        }
    finally:
        _close(env)


def climb_from_fast_window(path: Path) -> dict:
    from super_metroid.routes.kpdr.ceres.elev_escape import (
        _ceres_elev_leaving,
        _ceres_reactive_elev_climb,
    )

    env, _assist, sess = _boot(path)
    try:
        sess.ceres_shaft_trace = []
        err = None
        try:
            _ceres_reactive_elev_climb(sess)
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        return {
            "leave": bool(_ceres_elev_leaving(sess.state)),
            "end": _snap(sess.state),
            "frames": sess.frame,
            "err": err,
            "trace": sess.ceres_shaft_trace[-8:],
        }
    finally:
        _close(env)


PIN_363 = GAME_DIR / "scratch" / "ceres_elev_363_scan.state"


def plant_363_pin(src: Path, dest: Path, *, idle: int = 32) -> dict:
    env, _assist, sess = _boot(src)
    try:
        for _ in range(idle):
            sess.step(idle_action(), "idle")
        ok = _ceres_checkpoint_hop(sess, side="RIGHT", runup=4, target_y=363)
        write_state_bytes(dest, sess.env.em.get_state())
        return {"ok": bool(ok), "end": _snap(sess.state, sess.env.get_ram())}
    finally:
        _close(env)


def scan_363_left_hop(sess: _Sess, path: Path) -> list[dict]:
    rows = []
    for idle in range(0, 49, 2):
        for runup in (0, 2, 4, 6):
            _reload(sess, path)
            for _ in range(idle):
                sess.step(idle_action(), "idle")
            ok = _ceres_checkpoint_hop(
                sess, side="LEFT", runup=runup, target_y=267
            )
            end = sess.state
            compact = {
                "idle": idle,
                "runup": runup,
                "ok": bool(ok),
                "end_x": int(end.samus_x),
                "end_y": int(end.samus_y),
                "end_pose": int(end.pose),
                "kb": int(is_knockback(end)),
                "checkpoint": _seat(end),
                "planted_267": bool(_ceres_planted_at(end, 267)),
            }
            rows.append(compact)
            if compact["planted_267"] or compact["checkpoint"] in (267, 171):
                return rows
    return rows


def scan_363_right_hop(sess: _Sess, path: Path) -> list[dict]:
    rows = []
    for idle in range(0, 49, 2):
        for runup in (0, 2, 4):
            _reload(sess, path)
            for _ in range(idle):
                sess.step(idle_action(), "idle")
            ok = _ceres_checkpoint_hop(
                sess, side="RIGHT", runup=runup, target_y=267
            )
            end = sess.state
            compact = {
                "idle": idle,
                "runup": runup,
                "ok": bool(ok),
                "end_x": int(end.samus_x),
                "end_y": int(end.samus_y),
                "end_pose": int(end.pose),
                "kb": int(is_knockback(end)),
                "checkpoint": _seat(end),
                "planted_267": bool(_ceres_planted_at(end, 267)),
            }
            rows.append(compact)
            if compact["planted_267"] or compact["checkpoint"] in (267, 171):
                return rows
    return rows


PIN_267 = GAME_DIR / "scratch" / "ceres_elev_267_scan.state"


def scan_363_idle_window(sess: _Sess, path: Path) -> list[dict]:
    rows = []
    for idle in range(30, 49):
        _reload(sess, path)
        for _ in range(idle):
            sess.step(idle_action(), "idle")
        ok = _ceres_checkpoint_hop(sess, side="LEFT", runup=0, target_y=267)
        end = sess.state
        rows.append(
            {
                "idle": idle,
                "ok": bool(ok),
                "end_x": int(end.samus_x),
                "end_y": int(end.samus_y),
                "end_pose": int(end.pose),
                "planted_267": bool(_ceres_planted_at(end, 267)),
            }
        )
    return rows


def climb_from_363(path: Path, *, idle: int) -> dict:
    from super_metroid.routes.kpdr.ceres.elev_escape import (
        _ceres_elev_leaving,
        _ceres_elev_top_to_ship,
    )

    env, _assist, sess = _boot(path)
    try:
        for _ in range(idle):
            sess.step(idle_action(), "idle")
        to_267 = _ceres_checkpoint_hop(sess, side="LEFT", runup=0, target_y=267)
        after_267 = _snap(sess.state)
        write_state_bytes(PIN_267, sess.env.em.get_state())
        to_171 = False
        ship_ok = False
        err = None
        if to_267:
            to_171 = _ceres_checkpoint_hop(
                sess, side="RIGHT", runup=0, target_y=171, start_x=131
            )
            try:
                _ceres_elev_top_to_ship(sess)
                ship_ok = bool(_ceres_elev_leaving(sess.state))
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
        return {
            "idle": idle,
            "to_267": bool(to_267),
            "after_267": after_267,
            "to_171": bool(to_171),
            "ship_ok": ship_ok,
            "end": _snap(sess.state),
            "frames": sess.frame,
            "err": err,
        }
    finally:
        _close(env)


def climb_from_475_shaft(path: Path) -> dict:
    from super_metroid.routes.kpdr.ceres.elev_escape import (
        _ceres_checkpoint_shaft,
        _ceres_elev_leaving,
        _ceres_elev_top_to_ship,
    )

    env, _assist, sess = _boot(path)
    try:
        err = None
        shaft_ok = False
        ship_ok = False
        try:
            shaft_ok = _ceres_checkpoint_shaft(sess)
            _ceres_elev_top_to_ship(sess)
            ship_ok = bool(_ceres_elev_leaving(sess.state))
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        return {
            "shaft_ok": shaft_ok,
            "ship_ok": ship_ok,
            "end": _snap(sess.state),
            "frames": sess.frame,
            "err": err,
        }
    finally:
        _close(env)


def main() -> None:
    report = {
        "climb_475": climb_from_475_shaft(PIN_475),
        "climb_fast": climb_from_fast_window(PIN),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "climb_475_ship": report["climb_475"]["ship_ok"],
        "climb_475_frames": report["climb_475"]["frames"],
        "climb_475_err": report["climb_475"]["err"],
        "climb_475_end": report["climb_475"]["end"],
        "climb_fast_leave": report["climb_fast"]["leave"],
        "climb_fast_frames": report["climb_fast"]["frames"],
        "climb_fast_err": report["climb_fast"]["err"],
        "climb_fast_end": report["climb_fast"]["end"],
        "climb_fast_trace": report["climb_fast"]["trace"],
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
