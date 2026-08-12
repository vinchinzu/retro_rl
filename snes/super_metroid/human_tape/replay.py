"""Hop-level open-loop replay from live gzip anchors (anti-desync)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from super_metroid.human_tape.anchors import as_xy, parse_room_id
from super_metroid.human_tape.hops import load_task_json, resolve_hop_slice


def resolve_assist(assist: bool | Any = True) -> Any | None:
    """Normalize assist flag / instance for hop step loops.

    ``True`` → fresh ``UnlimitedResourcesAssist`` (record-path parity).
    ``False`` / ``None`` → no assist.
    Other objects are returned as-is (reuse telemetry).
    """
    if assist is True:
        from super_metroid.assist import UnlimitedResourcesAssist

        return UnlimitedResourcesAssist()
    if assist is False or assist is None:
        return None
    return assist


def frame_action(frame: Sequence[int] | np.ndarray) -> np.ndarray:
    """SNES-12 int list → int8 action array for ``env.step``."""
    return np.array(frame, dtype=np.int8)


def iter_hop_steps(
    env: Any,
    frames: Sequence[Sequence[int]],
    start_i: int,
    end_i: int,
    *,
    assist: bool | Any = True,
) -> Iterator[tuple[int, Any]]:
    """Yield ``(index, state)`` after each applied frame + optional assist.

    Does **not** load state — caller boots the env. Inclusive
    ``[start_i, end_i]``. Shared by ``replay_hop`` and ``lockstep_scan``.
    """
    from super_metroid.ram import parse_env_state

    n = len(frames)
    if n == 0:
        raise ValueError("empty frames")
    s = max(0, int(start_i))
    e = min(int(end_i), n - 1)
    if e < s:
        raise ValueError(f"empty slice start_i={start_i} end_i={end_i} n={n}")

    assist_obj = resolve_assist(assist)
    for i in range(s, e + 1):
        env.step(frame_action(frames[i]))
        if assist_obj is not None:
            st_now = parse_env_state(env, mode="nav")
            assist_obj.apply(env.data, st_now)
        st = parse_env_state(env, mode="nav")
        yield i, st


def replay_hop(
    env: Any,
    frames: Sequence[Sequence[int]],
    start_i: int,
    end_i: int,
    *,
    settle_frames: int = 0,
    assist: bool | Any = True,
) -> dict[str, Any]:
    """Open-loop step ``frames[start_i:end_i]`` inclusive; return final pin.

    Does **not** load state — caller boots the env (e.g. ``boot_from_state``).
    Optional ``settle_frames`` idle after the slice (door settle / observe).

    ``assist`` (default True): apply contract ``UnlimitedResourcesAssist`` after
    each step — same pattern as ``guided_human`` record. Required for combat
    hops (Mother Brain rainbow drain, ammo spray). Pass False for clean-track
    experiments, or an existing assist instance to reuse telemetry.
    """
    from retro_harness.actions import idle_action
    from super_metroid.ram import parse_env_state

    assist_obj = resolve_assist(assist)
    st = parse_env_state(env, mode="nav")
    start_pin = {
        "room_id": int(st.room_id),
        "xy": [int(st.samus_x), int(st.samus_y)],
        "pose": int(st.pose),
        "game_state": int(st.game_state),
        "phase": str(getattr(st.phase, "name", st.phase)),
    }

    stepped = 0
    for _i, st in iter_hop_steps(env, frames, start_i, end_i, assist=assist_obj):
        stepped += 1

    for _ in range(max(0, int(settle_frames))):
        env.step(idle_action())
        stepped += 1
        if assist_obj is not None:
            st_now = parse_env_state(env, mode="nav")
            assist_obj.apply(env.data, st_now)

    st = parse_env_state(env, mode="nav")
    phase = getattr(st.phase, "name", None) or getattr(st.phase, "value", st.phase)
    s = max(0, int(start_i))
    e = min(int(end_i), len(frames) - 1)
    out: dict[str, Any] = {
        "ok_steps": stepped,
        "start_i": s,
        "end_i": e,
        "start": start_pin,
        "room_id": int(st.room_id),
        "room": f"0x{int(st.room_id):04X}",
        "xy": [int(st.samus_x), int(st.samus_y)],
        "pose": int(st.pose),
        "phase": str(phase),
        "game_state": int(st.game_state),
        "door_transition": int(getattr(st, "door_transition", 0) or 0),
        "health": int(st.health),
        "energy": int(st.health),
        "assist": bool(assist_obj is not None),
    }
    if assist_obj is not None and hasattr(assist_obj, "report"):
        out["assist_report"] = assist_obj.report()
    return out


def check_hop_green(
    result: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    expected_leave_room: int | str | None,
    expected_end_xy: Sequence[int] | None = None,
    *,
    xy_tol: int = 24,
    dual: bool = False,
    start_room: int | str | None = None,
) -> dict[str, Any]:
    """Green if leave-room matches (or left start_room) and optional xy band.

    When ``dual=True``, ``result`` may be a list of two run dicts — both must
    be individually green.
    """
    if dual:
        runs = list(result) if isinstance(result, Sequence) and not isinstance(result, Mapping) else [result]
        if len(runs) < 2:
            return {
                "ok": False,
                "dual": True,
                "reason": f"dual requires 2 results, got {len(runs)}",
                "runs": [],
            }
        checks = [
            check_hop_green(
                r,
                expected_leave_room,
                expected_end_xy,
                xy_tol=xy_tol,
                dual=False,
                start_room=start_room,
            )
            for r in runs[:2]
        ]
        ok = all(c.get("ok") for c in checks)
        return {
            "ok": ok,
            "dual": True,
            "runs": checks,
            "reason": None if ok else "dual_mismatch",
        }

    res = dict(result)  # type: ignore[arg-type]
    got_room = int(res.get("room_id") or 0)
    got_xy = as_xy(res.get("xy")) or [0, 0]
    exp_leave = parse_room_id(expected_leave_room) if expected_leave_room is not None else None
    exp_start = parse_room_id(start_room) if start_room is not None else None

    room_ok = True
    left_start = True
    if exp_leave is not None:
        room_ok = got_room == exp_leave
    elif exp_start is not None:
        left_start = True
        room_ok = True
    if exp_start is not None and exp_leave is not None:
        left_start = got_room != exp_start

    xy_ok = True
    xy_delta = None
    if expected_end_xy is not None:
        ex, ey = int(expected_end_xy[0]), int(expected_end_xy[1])
        dx = abs(got_xy[0] - ex)
        dy = abs(got_xy[1] - ey)
        xy_delta = [dx, dy]
        xy_ok = dx <= int(xy_tol) and dy <= int(xy_tol)

    ok = bool(room_ok and xy_ok)
    reason = None
    if not room_ok:
        reason = (
            f"room 0x{got_room:04X} != leave "
            f"0x{exp_leave:04X}" if exp_leave is not None else "room_mismatch"
        )
    elif not xy_ok:
        reason = f"xy {got_xy} outside tol={xy_tol} of {list(expected_end_xy or [])}"

    return {
        "ok": ok,
        "dual": False,
        "room_ok": room_ok,
        "xy_ok": xy_ok,
        "left_start": left_start,
        "got_room": got_room,
        "got_room_hex": f"0x{got_room:04X}",
        "got_xy": got_xy,
        "expected_leave_room": exp_leave,
        "expected_end_xy": list(expected_end_xy) if expected_end_xy is not None else None,
        "xy_tol": int(xy_tol),
        "xy_delta": xy_delta,
        "reason": reason,
        "result": res,
    }


def run_hop_replay(
    task_path: Path | str,
    *,
    hop_index: int | None = None,
    from_frame: int | None = None,
    to_frame: int | None = None,
    frames_count: int | None = None,
    room: int | str | None = None,
    to_room: int | str | None = None,
    anchor_path: Path | str | None = None,
    dual: bool = False,
    xy_tol: int = 24,
    settle_frames: int = 0,
    boot_settle: int = 0,
    leave_extra: int = 1,
    assist: bool | Any = True,
    env: Any | None = None,
    allow_anchor_mismatch: bool = False,
) -> dict[str, Any]:
    """End-to-end: resolve slice, boot anchor, replay, check green (optional dual).

    Default ``assist=True`` matches guided_human record (unlimited energy+ammo
    under ASSIST_CONTRACT). Combat hops (MB rainbow) RED without it.

    Fails loud when the matched anchor room does not match hop start room
    (unless ``allow_anchor_mismatch`` or caller forces ``anchor_path``).
    """
    from super_metroid.dev.common import boot_from_state, make_dev_env

    path = Path(task_path)
    data = load_task_json(path)
    frames = data.get("frames") or []
    slice_info = resolve_hop_slice(
        path,
        hop_index=hop_index,
        from_frame=from_frame,
        to_frame=to_frame,
        frames_count=frames_count,
        room=room,
        to_room=to_room,
        leave_extra=leave_extra,
        task_data=data,
    )

    if (
        slice_info.get("anchor_room_mismatch")
        and not allow_anchor_mismatch
        and anchor_path is None
    ):
        return {
            "ok": False,
            "green": False,
            "reason": slice_info.get("anchor_warning")
            or "anchor room does not match hop start room",
            "slice": slice_info,
            "runs": [],
        }

    ap = Path(anchor_path) if anchor_path else None
    if ap is None and slice_info.get("anchor_path"):
        ap = Path(str(slice_info["anchor_path"]))
    if ap is None or not ap.is_file():
        return {
            "ok": False,
            "green": False,
            "reason": f"no anchor state (path={ap})",
            "slice": slice_info,
            "runs": [],
        }

    # If caller forced --from-frame / --frames, honor replay_start from those
    replay_start = int(slice_info["replay_start"])
    end_index = int(slice_info["end_index"])
    if from_frame is not None:
        replay_start = int(from_frame)
    if to_frame is not None:
        end_index = int(to_frame)
    elif frames_count is not None:
        end_index = replay_start + int(frames_count) - 1
        end_index = min(end_index, len(frames) - 1)

    owns_env = env is None
    if env is None:
        env = make_dev_env()

    runs: list[dict[str, Any]] = []
    n_runs = 2 if dual else 1
    try:
        for _ in range(n_runs):
            boot_from_state(env, ap, settle_frames=boot_settle)
            result = replay_hop(
                env,
                frames,
                replay_start,
                end_index,
                settle_frames=settle_frames,
                assist=assist,
            )
            result["anchor_path"] = str(ap)
            result["replay_start"] = replay_start
            result["replay_end"] = end_index
            runs.append(result)
    finally:
        if owns_env:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    expected_leave = slice_info.get("leave_room")
    if to_room is not None:
        expected_leave = parse_room_id(to_room)
    expected_xy = slice_info.get("end_xy")
    check = check_hop_green(
        runs if dual else runs[0],
        expected_leave,
        expected_xy,
        xy_tol=xy_tol,
        dual=dual,
        start_room=slice_info.get("start_room"),
    )
    return {
        "ok": bool(check.get("ok")),
        "green": bool(check.get("ok")),
        "check": check,
        "slice": slice_info,
        "anchor_path": str(ap),
        "replay_start": replay_start,
        "replay_end": end_index,
        "runs": runs,
        "dual": dual,
        "assist": bool(assist) if not isinstance(assist, bool) else assist,
    }
