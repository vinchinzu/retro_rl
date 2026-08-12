"""Offline midpoints + single-pass lockstep materialize."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.anchors import (
    fingerprint,
    load_anchors_index,
    write_gzip_state,
)
from super_metroid.human_tape.hops import load_task_json, resolve_hop_slice
from super_metroid.human_tape.replay import (
    check_hop_green,
    iter_hop_steps,
    replay_hop,
    resolve_assist,
)

# Metroid latch / knockback / freeze-class poses often mark combat phase cuts.
_COMBAT_PIN_POSES = frozenset({84, 138, 232, 233, 235})


def propose_trace_midpoints(
    trace: Sequence[Mapping[str, Any]],
    start_index: int,
    end_index: int,
    *,
    end_xy: Sequence[int] | None = None,
    big_move_dx: int = 100,
    big_move_dy: int = 64,
    floor_y_delta: int = 80,
    energy_drop: int = 40,
    min_gap: int = 90,
) -> list[dict[str, Any]]:
    """Offline midpoint *candidates* from a hop's trace (no emulator).

    These are edit / re-record hints and lockstep dump targets — **not**
    live gzip anchors until materialize_lockstep_mid writes a state.

    ``min_gap`` is always enforced between successive accepted candidates
    (by insertion order; final list is sorted by index).
    """
    if not trace:
        return []
    lo = max(0, int(start_index))
    hi = min(int(end_index), len(trace) - 1)
    if hi < lo:
        return []

    candidates: list[dict[str, Any]] = []
    seen_i: set[int] = set()

    def _add(i: int, kind: str, note: str = "") -> None:
        """Accept index if new and at least min_gap after last candidate."""
        if i < lo or i > hi or i in seen_i:
            return
        if candidates and i - int(candidates[-1]["index"]) < int(min_gap):
            return
        row = trace[i]
        room = int(row.get("room") or 0)
        entry = {
            "index": i,
            "frame": int(row.get("frame", i)),
            "kind": kind,
            "room": f"0x{room:04X}",
            "room_id": room,
            "xy": [int(row.get("x", 0)), int(row.get("y", 0))],
            "pose": int(row.get("pose", 0)),
            "energy": row.get("energy"),
            "note": note,
        }
        candidates.append(entry)
        seen_i.add(i)

    start_y = int(trace[lo].get("y", 0))
    max_y = start_y
    floor_landed = False
    prev = trace[lo]
    for i in range(lo + 1, hi + 1):
        row = trace[i]
        x, y = int(row.get("x", 0)), int(row.get("y", 0))
        px, py = int(prev.get("x", 0)), int(prev.get("y", 0))
        pose = int(row.get("pose", 0))
        prev_pose = int(prev.get("pose", 0))
        e = int(row.get("energy") or 0)
        pe = int(prev.get("energy") or 0)

        if y > max_y:
            max_y = y
        # First deep floor land relative to hop start
        if (
            not floor_landed
            and y >= start_y + int(floor_y_delta)
            and abs(int(row.get("vy") or 0)) <= 1
        ):
            _add(i, "floor_land", f"y={y} from start_y={start_y}")
            floor_landed = True

        if abs(x - px) >= int(big_move_dx) or abs(y - py) >= int(big_move_dy):
            _add(i, "big_move", f"dx={x - px} dy={y - py}")

        if pose != prev_pose and pose in _COMBAT_PIN_POSES:
            _add(i, "combat_pose", f"pose {prev_pose}→{pose}")

        if e - pe <= -int(energy_drop):
            _add(i, "energy_drop", f"energy {pe}→{e}")

        prev = row

    # Pre-leave: last ordinary frame still in start room near end_xy band
    start_room = int(trace[lo].get("room") or 0)
    if end_xy is not None and len(end_xy) >= 2:
        ex, ey = int(end_xy[0]), int(end_xy[1])
        best_i = None
        best_d = 10**9
        for i in range(hi, lo, -1):
            row = trace[i]
            if int(row.get("room") or 0) != start_room:
                continue
            dx = abs(int(row.get("x", 0)) - ex)
            dy = abs(int(row.get("y", 0)) - ey)
            d = dx + dy
            if d < best_d:
                best_d = d
                best_i = i
            if d <= 32:
                break
        if best_i is not None:
            _add(best_i, "pre_leave", f"dist≈{best_d} to end_xy")

    # Mid dwell quarter markers when hop is long and sparse
    dwell = hi - lo + 1
    if dwell >= 800 and len(candidates) < 3:
        for frac, label in ((0.25, "q1"), (0.5, "q2"), (0.75, "q3")):
            _add(lo + int(dwell * frac), "quarter", label)

    candidates.sort(key=lambda c: int(c["index"]))
    return candidates


def _trace_match(
    st: Any,
    row: Mapping[str, Any],
    *,
    xy_tol: int,
    pose_tol: bool,
) -> tuple[bool, dict[str, Any]]:
    got_xy = [int(st.samus_x), int(st.samus_y)]
    want_xy = [int(row.get("x", 0)), int(row.get("y", 0))]
    got_room = int(st.room_id)
    want_room = int(row.get("room") or 0)
    got_pose = int(st.pose)
    want_pose = int(row.get("pose") or 0)
    room_ok = got_room == want_room
    xy_ok = (
        abs(got_xy[0] - want_xy[0]) <= int(xy_tol)
        and abs(got_xy[1] - want_xy[1]) <= int(xy_tol)
    )
    pose_ok = True if not pose_tol else got_pose == want_pose
    ok = bool(room_ok and xy_ok and pose_ok)
    return ok, {
        "room_ok": room_ok,
        "xy_ok": xy_ok,
        "pose_ok": pose_ok,
        "got_room": got_room,
        "got_xy": got_xy,
        "got_pose": got_pose,
        "want_room": want_room,
        "want_xy": want_xy,
        "want_pose": want_pose,
    }


def lockstep_scan(
    env: Any,
    frames: Sequence[Sequence[int]],
    trace: Sequence[Mapping[str, Any]],
    start_i: int,
    end_i: int,
    *,
    xy_tol: int = 12,
    pose_strict: bool = False,
    sample_every: int = 1,
    assist: bool | Any = True,
) -> dict[str, Any]:
    """Step hop frames comparing emulator state to the recorded ``trace``.

    Contiguous-only: stops on the first mismatch. Keeps ``last_ok_i`` and
    ``last_ok_blob`` (emulator state bytes at last match) for single-pass
    materialize.
    """
    assist_obj = resolve_assist(assist)
    s = max(0, int(start_i))
    e = min(int(end_i), len(frames) - 1)
    every = max(1, int(sample_every))
    last_ok_i: int | None = None
    last_ok_blob: bytes | None = None
    last_ok_st: Any | None = None
    first_mismatch: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = []

    # After boot the anchor was dumped at frame F; next input is F+1
    # (=start_i). Trace row i is state *after* applying frames[i] on record.
    for i, st in iter_hop_steps(env, frames, s, e, assist=assist_obj):
        if i >= len(trace):
            break
        row = trace[i]
        ok, detail = _trace_match(st, row, xy_tol=xy_tol, pose_tol=pose_strict)
        if ok:
            last_ok_i = i
            last_ok_blob = env.em.get_state()
            last_ok_st = st
        else:
            first_mismatch = {"index": i, "frame": int(row.get("frame", i)), **detail}
            samples.append({"index": i, "ok": False, **detail})
            break  # contiguous-only: stop at first desync
        if (i - s) % every == 0:
            samples.append({"index": i, "ok": True, **detail})

    return {
        "start_i": s,
        "end_i": e,
        "last_match": last_ok_i,
        # Contiguous-only product rule: last_match is always contiguous.
        "contiguous_last_match": last_ok_i,
        "last_ok_i": last_ok_i,
        "last_ok_blob": last_ok_blob,
        "last_ok_st": last_ok_st,
        "first_mismatch": first_mismatch,
        "samples": samples[-40:],  # cap
        "sample_count": len(samples),
        "assist": assist_obj is not None,
        "xy_tol": int(xy_tol),
        "pose_strict": bool(pose_strict),
    }


def materialize_lockstep_mid(
    task_path: Path | str,
    *,
    hop_index: int | None = None,
    from_frame: int | None = None,
    to_frame: int | None = None,
    target_index: int | None = None,
    anchor_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    update_index: bool = True,
    xy_tol: int = 12,
    pose_strict: bool = False,
    boot_settle: int = 0,
    leave_extra: int = 0,
    assist: bool | Any = True,
    dual_verify: bool = True,
    label: str = "mid",
    env: Any | None = None,
) -> dict[str, Any]:
    """Recover a **live** mid gzip pin by lockstep from a hop enter anchor.

    Single contiguous lockstep pass from the enter anchor: keeps last OK
    emulator blob; stops on first mismatch (no post-recover matching). Dumps
    ``last_ok_blob`` (or target when still within the contiguous match run).
    Refuses if ``target_index`` is past last OK.

    Writes under ``tasks/<tape>_anchors/`` and appends to ``*_anchors.json``
    with kind ``mid_lockstep``.
    """
    from super_metroid.dev.common import boot_from_state, make_dev_env
    from super_metroid.ram import parse_env_state

    path = Path(task_path)
    data = load_task_json(path)
    frames = data.get("frames") or []
    trace = list(data.get("trace") or [])
    slice_info = resolve_hop_slice(
        path,
        hop_index=hop_index,
        from_frame=from_frame,
        to_frame=to_frame,
        leave_extra=leave_extra,
        task_data=data,
    )
    ap = Path(anchor_path) if anchor_path else None
    if ap is None and slice_info.get("anchor_path"):
        ap = Path(str(slice_info["anchor_path"]))
    if ap is None or not ap.is_file():
        return {
            "ok": False,
            "reason": f"no enter anchor (path={ap})",
            "slice": slice_info,
        }

    replay_start = int(slice_info["replay_start"])
    end_index = int(slice_info["end_index"])
    if from_frame is not None:
        replay_start = int(from_frame)
    if to_frame is not None:
        end_index = int(to_frame)

    # If target is set, only need to scan through target (still contiguous).
    scan_end = end_index
    if target_index is not None:
        scan_end = min(end_index, int(target_index))

    owns_env = env is None
    if env is None:
        env = make_dev_env()

    try:
        boot_from_state(env, ap, settle_frames=boot_settle)
        scan = lockstep_scan(
            env,
            frames,
            trace,
            replay_start,
            scan_end,
            xy_tol=xy_tol,
            pose_strict=pose_strict,
            sample_every=max(1, (scan_end - replay_start) // 40 or 1),
            assist=assist,
        )
        last_ok = scan.get("last_ok_i")
        if last_ok is None:
            last_ok = scan.get("last_match")
        blob = scan.get("last_ok_blob")
        last_ok_st = scan.get("last_ok_st")

        if target_index is not None:
            dump_i = int(target_index)
            if last_ok is None or dump_i > int(last_ok):
                return {
                    "ok": False,
                    "reason": (
                        f"target_index {dump_i} is past last_ok "
                        f"{last_ok} (first_mismatch={scan.get('first_mismatch')})"
                    ),
                    "slice": slice_info,
                    "scan": {
                        k: v
                        for k, v in scan.items()
                        if k not in ("last_ok_blob", "last_ok_st")
                    },
                    "last_match": last_ok,
                }
            if dump_i != int(last_ok):
                # Contiguous run reached past target only if scan_end > target;
                # with scan_end == target, last_ok should equal target on success.
                return {
                    "ok": False,
                    "reason": (
                        f"target_index {dump_i} did not lockstep-match "
                        f"(last_ok={last_ok}, first_mismatch={scan.get('first_mismatch')})"
                    ),
                    "slice": slice_info,
                    "scan": {
                        k: v
                        for k, v in scan.items()
                        if k not in ("last_ok_blob", "last_ok_st")
                    },
                    "last_match": last_ok,
                }
        else:
            dump_i = last_ok
            if dump_i is None:
                return {
                    "ok": False,
                    "reason": "no lockstep match frames",
                    "slice": slice_info,
                    "scan": {
                        k: v
                        for k, v in scan.items()
                        if k not in ("last_ok_blob", "last_ok_st")
                    },
                }
            dump_i = int(dump_i)

        if not blob:
            return {
                "ok": False,
                "reason": "no lockstep state blob",
                "slice": slice_info,
                "last_match": last_ok,
            }

        # Prefer recorded last_ok_st; fall back to env (at last_ok when scan
        # ended on match, or one step past when mismatch stopped the scan —
        # in mismatch case last_ok_st is still the prior match).
        if last_ok_st is not None:
            st = last_ok_st
        else:
            st = parse_env_state(env, mode="nav")
        room = int(st.room_id)
        frame_label = (
            int(trace[dump_i].get("frame", dump_i)) if dump_i < len(trace) else dump_i
        )

        anchors_index = load_anchors_index(path) or {
            "task": path.stem,
            "anchors_dir": str(path.with_name(path.stem + "_anchors")),
            "count": 0,
            "anchors": [],
        }
        adir = Path(
            str(
                out_dir
                or anchors_index.get("anchors_dir")
                or path.with_name(path.stem + "_anchors")
            )
        )
        adir.mkdir(parents=True, exist_ok=True)
        fname = f"f{frame_label:06d}_{label}_0x{room:04X}.state"
        state_path = adir / fname
        write_gzip_state(state_path, blob)

        fp = fingerprint(
            frame=frame_label,
            room_id=room,
            x=int(st.samus_x),
            y=int(st.samus_y),
            pose=int(st.pose),
            items=int(getattr(st, "collected_items", 0) or 0) or None,
            beams=int(getattr(st, "collected_beams", 0) or 0) or None,
            energy=int(st.health),
            kind="mid_lockstep",
            path=str(state_path),
            extra={
                "source": "lockstep_materialize",
                "trace_index": dump_i,
                "hop_index": slice_info.get("hop_index"),
                "from_anchor": str(ap),
                "xy_tol": int(xy_tol),
                "last_match": last_ok,
                "first_mismatch": scan.get("first_mismatch"),
            },
        )

        if update_index:
            rows = list(anchors_index.get("anchors") or [])
            # Replace same path/frame if re-run (mid_lockstep kind)
            rows = [
                r
                for r in rows
                if not (
                    int(r.get("frame") or -1) == frame_label
                    and str(r.get("kind")) == "mid_lockstep"
                    and Path(str(r.get("path") or "")).name == fname
                )
            ]
            rows.append(fp)
            rows.sort(key=lambda r: int(r.get("frame") or 0))
            anchors_index["anchors"] = rows
            anchors_index["count"] = len(rows)
            anchors_index["anchors_dir"] = str(adir)
            idx_path = path.with_name(path.stem + "_anchors.json")
            idx_path.write_text(
                json.dumps(anchors_index, indent=2) + "\n", encoding="utf-8"
            )
            fp["index_path"] = str(idx_path)

        dual_ok = None
        dual_detail = None
        if dual_verify:
            # Reboot enter → replay to dump_i; check xy band vs fingerprint
            runs = []
            for _ in range(2):
                boot_from_state(env, ap, settle_frames=boot_settle)
                r = replay_hop(
                    env,
                    frames,
                    replay_start,
                    dump_i,
                    settle_frames=0,
                    assist=assist,
                )
                runs.append(r)
            dual_detail = check_hop_green(
                runs,
                room,  # still in room
                fp["xy"],
                xy_tol=max(int(xy_tol), 8),
                dual=True,
                start_room=slice_info.get("start_room"),
            )
            dual_ok = bool(dual_detail.get("ok"))

        return {
            "ok": True if dual_ok is not False else False,
            "mid": fp,
            "state_path": str(state_path),
            "dump_index": dump_i,
            "last_match": last_ok,
            "first_mismatch": scan.get("first_mismatch"),
            "slice": slice_info,
            "dual_verify": dual_ok,
            "dual_detail": dual_detail,
            "scan_summary": {
                "last_match": last_ok,
                "first_mismatch": scan.get("first_mismatch"),
                "xy_tol": xy_tol,
            },
        }
    finally:
        if owns_env:
            close = getattr(env, "close", None)
            if callable(close):
                close()
