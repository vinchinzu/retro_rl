"""HappyLee & Mars608 warpless TAS (#3728M) — 32-exit / no-warp movie.

Parallel to the any% warps track. Play exported slices from **this** movie
only. HappyLee #1715M warp 1-1 / W4 1-2 and the hand-built ``smb_1_2_flag``
body are a different phase and desync this chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from smb.paths import GAME_DIR, MODELS_DIR, RECORDINGS_DIR

WARPLESS_FM2 = GAME_DIR / "tas" / "ref" / "happylee_mars608_warpless_3728M.fm2"
WARPLESS_BK2 = GAME_DIR / "tas" / "ref" / "happylee_mars608_warpless_3728M.fm2.bk2"
WARPLESS_PUBLICATION = "https://tasvideos.org/3728M"
WARPLESS_SUBMISSION = "https://tasvideos.org/5975S"
WARPLESS_AUTHOR = "HappyLee & Mars608"
WARPLESS_TIME = "18:36.78"
# Published FCEUX 2.2.1 movie length (input frames).
WARPLESS_FRAMES = 67_117
WARPLESS_START_FRAME = 41  # first non-idle (Start); same frame as warps #1715
WARPLESS_FIRST_LR = 196  # first L+R accel; prefix matches warps through f218

WARPLESS_REPORT_DIR = RECORDINGS_DIR / "tas_import" / "warpless_3728M"
WARPLESS_EXITS_REPORT = WARPLESS_REPORT_DIR / "exits.json"

DEFAULT_ALIGN_SKIP = 0

# Isolated Level1_1 (settle=2). Same FM2 start as warps 1-1; body is longer
# because warpless 1-1 diverges at movie frame 219 and ends at 1-2 flag-route.
WL_1_1_SETTLE = 2
WL_1_1_FM2_START = 190
WL_1_1_LEAVE_FRAMES = 1754
WL_1_1_SEED = "smb_1_1_warpless_slice.json"

# 1-2 flag pipe (not W4). Same FM2 start as warps 1-2; body goes to 1-3.
WL_1_2_FM2_START = 2109
WL_1_2_LEAVE_FRAMES = 2544
WL_1_2_CTRL_WAIT = 165
WL_1_2_SEED = "smb_1_2_warpless_flag_slice.json"

# 1-3 athletic (mushroom route). After 1-2 flag leave, wait=0; FM2 @4653.
WL_1_3_FM2_HINT = WL_1_2_FM2_START + WL_1_2_LEAVE_FRAMES  # 4653
WL_1_3_FM2_START = 4653
WL_1_3_LEAVE_FRAMES = 1740
WL_1_3_CTRL_WAIT = 0
WL_1_3_SEED = "smb_1_3_warpless_slice.json"
# 1-4 castle. Movie index after the 1-3 body; search after idle to 1-4 control.
WL_1_4_FM2_HINT = WL_1_3_FM2_START + WL_1_3_LEAVE_FRAMES  # 6393

WARPLESS_PUBLICATION_ID = "3728M"
WARP_PUBLICATION_ID = "1715M"
CHAIN_TARGETS = ("1-1", "1-2", "1-3")
WARPLESS_SEEDS: dict[str, str] = {
    "1-1": WL_1_1_SEED,
    "1-2": WL_1_2_SEED,
    "1-3": WL_1_3_SEED,
}

OnStep = Callable[[Any, Any, str, int], None]


def warpless_present() -> bool:
    """True when the vendored FM2 is on disk (gitignored)."""
    try:
        return WARPLESS_FM2.exists() and WARPLESS_FM2.stat().st_size > 1000
    except OSError:
        return False


def bk2_present() -> bool:
    try:
        return WARPLESS_BK2.exists() and WARPLESS_BK2.stat().st_size > 1000
    except OSError:
        return False


def summary_dict() -> dict[str, object]:
    """Offline provenance for reports (does not parse the movie)."""
    return {
        "publication": WARPLESS_PUBLICATION,
        "submission": WARPLESS_SUBMISSION,
        "author": WARPLESS_AUTHOR,
        "time": WARPLESS_TIME,
        "num_frames": WARPLESS_FRAMES,
        "fm2": str(WARPLESS_FM2),
        "bk2": str(WARPLESS_BK2),
        "fm2_present": warpless_present(),
        "bk2_present": bk2_present(),
        "first_start_frame": WARPLESS_START_FRAME,
        "first_lr_frame": WARPLESS_FIRST_LR,
        "route_id": "smb_all_exits",
        "note": (
            "Warpless / 32-exit TAS. 1-2 is the flag pipe, not W4. "
            "1-3 athletic 1740f @4653 → 1-4. Do not mix with happylee_warps_1715M slices."
        ),
    }


def slice_path(stage_id: str) -> Path:
    key = _stage_key(stage_id)
    if key not in WARPLESS_SEEDS:
        raise KeyError(f"unknown warpless stage {stage_id!r}; known: {list(WARPLESS_SEEDS)}")
    return MODELS_DIR / WARPLESS_SEEDS[key]


def slices_present(to: str = "1-3") -> bool:
    """True when every exported #3728M seed up to *to* is on disk."""
    try:
        return all(slice_path(sid).is_file() for sid in _legs_to(to))
    except KeyError:
        return False


def _stage_key(stage_id: str) -> str:
    return stage_id.strip().lower().replace("_", "-")


def _legs_to(target: str) -> tuple[str, ...]:
    key = _stage_key(target)
    if key not in CHAIN_TARGETS:
        raise KeyError(f"target must be one of {CHAIN_TARGETS}, got {target!r}")
    return CHAIN_TARGETS[: CHAIN_TARGETS.index(key) + 1]


def require_warpless_slice(data: dict[str, Any], *, stage_id: str) -> dict[str, Any]:
    """Reject any% warp / hand-built flag cuts. 32-exit needs #3728M only."""
    key = _stage_key(stage_id)
    source = str(data.get("source", ""))
    note = str(data.get("note", ""))
    blob = f"{source} {note} {data.get('fm2', '')} {data.get('level_id', '')}"
    if WARP_PUBLICATION_ID in blob and WARPLESS_PUBLICATION_ID not in blob:
        raise ValueError(
            f"{key} seed is a warp #{WARP_PUBLICATION_ID} cut; "
            f"32-exit chain needs warpless #{WARPLESS_PUBLICATION_ID}"
        )
    if WARPLESS_PUBLICATION_ID not in blob and "warpless" not in blob.lower():
        raise ValueError(
            f"{key} seed is not a warpless #{WARPLESS_PUBLICATION_ID} cut: {source!r}"
        )
    route = str(data.get("route_id", ""))
    if route != "smb_all_exits":
        raise ValueError(f"{key} route_id={route!r}, expected smb_all_exits")
    got_stage = str(data.get("stage_id", "") or "")
    if got_stage and got_stage != key:
        raise ValueError(f"{key} seed stage_id={got_stage!r}")
    return data


def load_warpless_slice(stage_id: str) -> tuple[dict[str, Any], list[list[int]]]:
    """Load one exported #3728M body. Raises if missing or the wrong movie."""
    from smb.policy import expand_nes9_rle, load_nes9_rle_seed

    key = _stage_key(stage_id)
    path = slice_path(key)
    if not path.is_file():
        export_flag = {
            "1-1": "--export-1-1",
            "1-2": "--export-1-2-flag",
            "1-3": "--export-1-3",
        }[key]
        raise FileNotFoundError(
            f"missing {path}; export with: uv run python -m smb.scripts.annotate_fm2 {export_flag}"
        )
    data = load_nes9_rle_seed(path)
    require_warpless_slice(data, stage_id=key)
    frames = expand_nes9_rle(data)
    return data, frames


def _snap_brief(snap: Any) -> dict[str, int]:
    return {
        "world": int(snap.world),
        "level": int(snap.level),
        "dash_level": int(getattr(snap, "dash_level", getattr(snap, "level_number", -1)) or -1),
        "oper_mode": int(snap.oper_mode),
        "player_state": int(snap.player_state),
        "player_x": int(snap.player_x),
        "player_y": int(snap.player_y),
        "timer": int(getattr(snap, "timer", 0) or 0),
        "lives": int(getattr(snap, "lives", 0) or 0),
    }


def _emit_step(
    env: Any,
    action: Any,
    *,
    label: str,
    frame_i: int,
    on_step: OnStep | None,
) -> Any:
    from smb.tas.replay import to_action9

    act = to_action9(action)
    obs, *_ = env.step(act)
    if on_step is not None:
        on_step(obs, act, label, frame_i)
    return obs


def _idle_until(
    env: Any,
    pred: Callable[[Any], bool],
    *,
    label: str,
    start_frame: int,
    max_wait: int,
    on_step: OnStep | None,
) -> tuple[int, Any]:
    from smb.ram import read_snapshot
    from smb.tas.replay import IDLE

    wait = 0
    snap = read_snapshot(env.get_ram(), frame=start_frame)
    while wait < max_wait:
        snap = read_snapshot(env.get_ram(), frame=start_frame + wait)
        if pred(snap):
            return wait, snap
        _emit_step(
            env, IDLE, label=label, frame_i=start_frame + wait + 1, on_step=on_step
        )
        wait += 1
    return wait, snap


def _play_body(
    env: Any,
    frames: list[list[int]],
    *,
    label: str,
    start_frame: int,
    start_lives: int,
    stop: Callable[[Any], bool] | None,
    on_step: OnStep | None,
) -> dict[str, Any]:
    from smb.ram import PLAYER_STATE_DYING, read_snapshot

    death: int | None = None
    stop_at: int | None = None
    max_x = 0
    last = read_snapshot(env.get_ram(), frame=start_frame)
    for i, fr in enumerate(frames):
        fnum = start_frame + i + 1
        _emit_step(env, fr, label=label, frame_i=fnum, on_step=on_step)
        last = read_snapshot(env.get_ram(), frame=fnum)
        px = int(last.player_x)
        if 0 < px < 20_000:
            max_x = max(max_x, px)
        if int(last.lives) < start_lives or int(last.player_state) == PLAYER_STATE_DYING:
            death = fnum
            break
        if stop is not None and stop(last):
            stop_at = fnum
            break
    played = (stop_at or death or (start_frame + len(frames))) - start_frame
    return {
        "label": label,
        "played": played,
        "body_len": len(frames),
        "death": death,
        "stop_at": stop_at,
        "max_x": max_x,
        "end_frame": start_frame + played,
        "end_snap": _snap_brief(last),
    }


def play_warpless_to(
    env: Any,
    *,
    to: str = "1-3",
    settle: int = WL_1_1_SETTLE,
    on_step: OnStep | None = None,
    max_wait: int = 600,
) -> dict[str, Any]:
    """Level1_1 → #3728M 1-1 → 1-2 flag → 1-3. Never loads #1715M warp seeds.

    *env* must already be reset into ``Level1_1``. Caller closes it.
    """
    from smb.ram import read_snapshot
    from smb.reactive_12 import is_surface_control
    from smb.tas.replay import IDLE
    from smb.tas.stages import is_1_3_control, is_1_4_control

    target = _stage_key(to)
    legs = _legs_to(target)
    bodies: dict[str, list[list[int]]] = {}
    seeds: dict[str, str] = {}
    for sid in legs:
        meta, frames = load_warpless_slice(sid)
        bodies[sid] = frames
        seeds[sid] = str(slice_path(sid))
        if int(meta.get("num_frames", -1)) != len(frames):
            raise ValueError(f"{sid} num_frames mismatch")

    stages: dict[str, Any] = {"seeds": seeds}
    frame = 0
    start_lives: int | None = None

    for _ in range(settle):
        frame += 1
        _emit_step(env, IDLE, label="settle", frame_i=frame, on_step=on_step)
        if start_lives is None:
            snap = read_snapshot(env.get_ram(), frame=frame)
            if 0 <= int(snap.lives) <= 8:
                start_lives = int(snap.lives)
    stages["settle"] = {"frames": settle}
    if start_lives is None:
        start_lives = int(read_snapshot(env.get_ram(), 0).lives)

    st11 = _play_body(
        env,
        bodies["1-1"],
        label="wl_1_1",
        start_frame=frame,
        start_lives=start_lives,
        stop=None,
        on_step=on_step,
    )
    frame = int(st11["end_frame"])
    stages["1_1"] = st11
    if st11["death"] is not None:
        return _chain_result(False, "death_1_1", target, settle, frame, stages)

    if target == "1-1":
        ok = st11["death"] is None and int(st11["max_x"]) >= 2500
        return _chain_result(ok, "clear_1_1" if ok else "1_1_incomplete", target, settle, frame, stages)

    wait12, ctrl12 = _idle_until(
        env,
        is_surface_control,
        label="wait_1_2",
        start_frame=frame,
        max_wait=max_wait,
        on_step=on_step,
    )
    frame += wait12
    stages["ctrl_wait_1_2"] = wait12
    stages["ctrl_1_2"] = _snap_brief(ctrl12)
    if not is_surface_control(ctrl12):
        return _chain_result(False, "surface_control_timeout", target, settle, frame, stages)

    st12 = _play_body(
        env,
        bodies["1-2"],
        label="wl_1_2_flag",
        start_frame=frame,
        start_lives=int(ctrl12.lives),
        stop=is_1_3_control,
        on_step=on_step,
    )
    frame = int(st12["end_frame"])
    stages["1_2"] = st12
    if st12["death"] is not None:
        return _chain_result(False, "death_1_2", target, settle, frame, stages)

    wait13, ctrl13 = _idle_until(
        env,
        is_1_3_control,
        label="wait_1_3",
        start_frame=frame,
        max_wait=max_wait,
        on_step=on_step,
    )
    frame += wait13
    stages["ctrl_wait_1_3"] = wait13
    stages["ctrl_1_3"] = _snap_brief(ctrl13)
    if not is_1_3_control(ctrl13):
        return _chain_result(False, "1_3_control_timeout", target, settle, frame, stages)

    if target == "1-2":
        return _chain_result(True, "1_3_control", target, settle, frame, stages)

    st13 = _play_body(
        env,
        bodies["1-3"],
        label="wl_1_3",
        start_frame=frame,
        start_lives=int(ctrl13.lives),
        stop=is_1_4_control,
        on_step=on_step,
    )
    frame = int(st13["end_frame"])
    stages["1_3"] = st13
    if st13["death"] is not None:
        return _chain_result(False, "death_1_3", target, settle, frame, stages)

    end = read_snapshot(env.get_ram(), frame=frame)
    ok = is_1_4_control(end)
    stages["ctrl_1_4"] = _snap_brief(end)
    return _chain_result(ok, "1_4_control" if ok else "missed_1_4", target, settle, frame, stages)


def _chain_result(
    ok: bool,
    outcome: str,
    target: str,
    settle: int,
    frame: int,
    stages: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": ok,
        "success": ok,
        "outcome": outcome,
        "target": target,
        "start_state": "Level1_1",
        "settle": settle,
        "source": "HappyLee & Mars608 warpless #3728M",
        "frame": frame,
        "stages": stages,
        "end_snap": (stages.get("ctrl_1_4") or stages.get("ctrl_1_3")
                     or stages.get("ctrl_1_2") or (stages.get("1_1") or {}).get("end_snap")),
    }
