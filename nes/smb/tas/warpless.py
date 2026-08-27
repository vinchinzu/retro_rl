"""HappyLee & Mars608 warpless TAS (#3728M) — 32-exit / no-warp movie.

Parallel to the any% warps track. Play exported slices from **this** movie
only. HappyLee #1715M warp 1-1 / W4 1-2 and the hand-built ``smb_1_2_flag``
body are a different phase and desync this chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from smb.paths import GAME_DIR, MODELS_DIR, RECORDINGS_DIR
from smb.tas.stages import is_dash_control, is_ending_axe

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
WL_1_4_SEED = "smb_1_4_warpless_slice.json"
WL_1_4_FM2_START = 6393
WL_1_4_LEAVE_FRAMES = 1702
WL_1_4_CTRL_WAIT = 0
WL_2_1_FM2_HINT = WL_1_4_FM2_START + WL_1_4_LEAVE_FRAMES  # 8095
WL_2_1_SEED = "smb_2_1_warpless_slice.json"
WL_2_1_FM2_START = 8095
WL_2_1_LEAVE_FRAMES = 2356
WL_2_1_CTRL_WAIT = 0
WL_2_2_FM2_HINT = WL_2_1_FM2_START + WL_2_1_LEAVE_FRAMES  # 10451

WARPLESS_PUBLICATION_ID = "3728M"
WARP_PUBLICATION_ID = "1715M"

OnStep = Callable[[Any, Any, str, int], None]


@dataclass(frozen=True)
class WarplessLeg:
    """One 32-exit body: control gate, leave gate, verified FM2 window."""

    id: str
    world: int
    dash: int
    leave_world: int | None
    leave_dash: int | None
    fm2_start: int
    body_frames: int
    seed_name: str
    ctrl_wait: int | None = None
    leave_ending: bool = False
    max_play: int = 3200
    search_window: int = 80
    stop_on_leave: bool = True
    control_fn: Callable[[Any], bool] | None = None

    def control(self, snap: Any) -> bool:
        if self.control_fn is not None:
            return self.control_fn(snap)
        if not is_dash_control(snap, self.world, self.dash):
            return False
        # Flag→overworld drop-in is y=0 then land y=176. 1-2 uses
        # is_surface_control (y>=160); other overworld legs match that.
        # Castles (dash 3) stand at y=80.
        if self.dash != 3 and int(getattr(snap, "player_y", 0) or 0) < 160:
            return False
        return True

    def leave(self, snap: Any) -> bool:
        if self.leave_ending:
            return is_ending_axe(snap)
        if self.leave_world is None or self.leave_dash is None:
            return False
        return is_dash_control(snap, self.leave_world, self.leave_dash)

    @property
    def extracted(self) -> bool:
        return self.body_frames > 0 and self.fm2_start > 0

    @property
    def leave_id(self) -> str:
        if self.leave_ending:
            return "ending"
        if self.leave_world is None or self.leave_dash is None:
            return ""
        return f"{self.leave_world + 1}-{self.leave_dash + 1}"

    @property
    def leave_outcome(self) -> str:
        if self.leave_ending:
            return "ending_axe"
        lid = self.leave_id
        return f"{lid.replace('-', '_')}_control" if lid else "leave"

    @property
    def target_name(self) -> str:
        return self.leave_outcome

    @property
    def seed_path(self) -> Path:
        return MODELS_DIR / self.seed_name

    @property
    def note(self) -> str:
        return _stage_note(self.world + 1, self.dash + 1)

    @property
    def start_state(self) -> str:
        if self.id == "1-1":
            return "Level1_1"
        pred = predecessor_leg(self.id)
        pred_tag = "1_2_flag" if pred is not None and pred.id == "1-2" else (
            pred.id.replace("-", "_") if pred is not None else "start"
        )
        return f"{self.id}_control_after_warpless_{pred_tag}"

    @property
    def level_id(self) -> str:
        if self.id == "1-2":
            return "smb_1_2_warpless_flag"
        return f"smb_{self.id.replace('-', '_')}_warpless"


def _surface_1_2(snap: Any) -> bool:
    from smb.reactive_12 import is_surface_control

    return is_surface_control(snap)


def _stage_note(world_1: int, level_1: int) -> str:
    sid = f"{world_1}-{level_1}"
    if sid == "8-4":
        return (
            "32-exit 8-4 maze + Bowser + axe (oper_mode=2). "
            "Do not fold into happylee warps slices."
        )
    if level_1 == 2:
        return (
            f"32-exit {sid} flag pipe. Not a warp room. "
            "Do not fold into happylee warps slices."
        )
    if level_1 == 3:
        return f"32-exit {sid} athletic. Do not fold into happylee warps slices."
    if level_1 == 4:
        return f"32-exit {sid} castle. Do not fold into happylee warps slices."
    return f"32-exit {sid} overworld. Do not fold into happylee warps slices."


def _max_play_for(world_1: int, level_1: int) -> int:
    if world_1 == 8 and level_1 == 4:
        return 6000
    if level_1 == 4:
        return 2500
    if level_1 == 2:
        return 3500
    return 3200


def _verified_cuts() -> dict[str, tuple[int, int, int | None, str]]:
    """id → (fm2_start, body_frames, ctrl_wait, seed_name)."""
    cuts: dict[str, tuple[int, int, int | None, str]] = {
        "1-1": (WL_1_1_FM2_START, WL_1_1_LEAVE_FRAMES, None, WL_1_1_SEED),
        "1-2": (WL_1_2_FM2_START, WL_1_2_LEAVE_FRAMES, WL_1_2_CTRL_WAIT, WL_1_2_SEED),
        "1-3": (WL_1_3_FM2_START, WL_1_3_LEAVE_FRAMES, WL_1_3_CTRL_WAIT, WL_1_3_SEED),
        "1-4": (WL_1_4_FM2_START, WL_1_4_LEAVE_FRAMES, WL_1_4_CTRL_WAIT, WL_1_4_SEED),
        "2-1": (WL_2_1_FM2_START, WL_2_1_LEAVE_FRAMES, WL_2_1_CTRL_WAIT, WL_2_1_SEED),
    }
    return cuts


def _build_legs() -> tuple[WarplessLeg, ...]:
    verified = _verified_cuts()
    legs: list[WarplessLeg] = []
    for world in range(8):
        for dash in range(4):
            sid = f"{world + 1}-{dash + 1}"
            if dash < 3:
                leave_world, leave_dash, ending = world, dash + 1, False
            elif world < 7:
                leave_world, leave_dash, ending = world + 1, 0, False
            else:
                leave_world, leave_dash, ending = None, None, True
            start, body, wait, seed = verified.get(
                sid,
                (0, 0, None, f"smb_{world + 1}_{dash + 1}_warpless_slice.json"),
            )
            legs.append(
                WarplessLeg(
                    id=sid,
                    world=world,
                    dash=dash,
                    leave_world=leave_world,
                    leave_dash=leave_dash,
                    leave_ending=ending,
                    fm2_start=start,
                    body_frames=body,
                    seed_name=seed,
                    ctrl_wait=wait,
                    max_play=_max_play_for(world + 1, dash + 1),
                    stop_on_leave=sid != "1-1",
                    control_fn=_surface_1_2 if sid == "1-2" else None,
                )
            )
    return tuple(legs)


WARPLESS_LEGS: tuple[WarplessLeg, ...] = _build_legs()
WARPLESS_BY_ID: dict[str, WarplessLeg] = {leg.id: leg for leg in WARPLESS_LEGS}
CHAIN_TARGETS: tuple[str, ...] = tuple(leg.id for leg in WARPLESS_LEGS if leg.extracted)
WARPLESS_SEEDS: dict[str, str] = {leg.id: leg.seed_name for leg in WARPLESS_LEGS}

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
            "1-3 athletic 1740f @4653 → 1-4. 1-4 castle 1702f @6393 → 2-1. "
            "Do not mix with happylee_warps_1715M slices."
        ),
    }


def get_leg(stage_id: str) -> WarplessLeg:
    """Lookup by ``W-L`` (``1-4``, ``1_4``). Raises ``KeyError`` if unknown."""
    key = _stage_key(stage_id)
    if key not in WARPLESS_BY_ID:
        raise KeyError(f"unknown warpless stage {stage_id!r}; known: 1-1…8-4")
    return WARPLESS_BY_ID[key]


def predecessor_leg(stage_id: str) -> WarplessLeg | None:
    """Previous 32-exit leg, or ``None`` for 1-1."""
    key = _stage_key(stage_id)
    ids = [leg.id for leg in WARPLESS_LEGS]
    if key not in ids:
        raise KeyError(f"unknown warpless stage {stage_id!r}")
    idx = ids.index(key)
    return None if idx == 0 else WARPLESS_LEGS[idx - 1]


def fm2_hint(stage_id: str) -> int:
    """Movie index after predecessor body (add ctrl-wait at search time)."""
    key = _stage_key(stage_id)
    pred = predecessor_leg(key)
    if pred is None:
        return WL_1_1_FM2_START
    if not pred.extracted:
        raise ValueError(f"predecessor {pred.id} is not extracted")
    return pred.fm2_start + pred.body_frames


def slice_path(stage_id: str) -> Path:
    return get_leg(stage_id).seed_path


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
    ids = tuple(leg.id for leg in WARPLESS_LEGS)
    if key not in ids:
        raise KeyError(f"target must be one of 1-1…8-4, got {target!r}")
    return ids[: ids.index(key) + 1]


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
        }.get(key, f"--search {key} --export")
        raise FileNotFoundError(
            f"missing {path}; export with: uv run python -m smb.scripts.annotate_fm2 {export_flag}"
        )
    data = load_nes9_rle_seed(path)
    require_warpless_slice(data, stage_id=key)
    frames = expand_nes9_rle(data)
    return data, frames


def _dash_field(snap: Any) -> int:
    raw = getattr(snap, "dash_level", None)
    if raw is None:
        raw = getattr(snap, "level_number", None)
    if raw is None:
        return -1
    return int(raw)


def _snap_brief(snap: Any) -> dict[str, int]:
    return {
        "world": int(snap.world),
        "level": int(snap.level),
        "dash_level": _dash_field(snap),
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


def _sid_key(stage_id: str) -> str:
    return stage_id.replace("-", "_")


def _body_label(stage_id: str) -> str:
    if stage_id == "1-2":
        return "wl_1_2_flag"
    return f"wl_{_sid_key(stage_id)}"


def play_warpless_to(
    env: Any,
    *,
    to: str = "1-3",
    settle: int = WL_1_1_SETTLE,
    on_step: OnStep | None = None,
    max_wait: int = 600,
) -> dict[str, Any]:
    """Level1_1 → exported #3728M bodies through *to*. Never loads #1715M.

    *env* must already be reset into ``Level1_1``. Caller closes it.
    Driven by :data:`WARPLESS_LEGS` (control + leave predicates), not
    per-target branches.
    """
    from smb.ram import read_snapshot
    from smb.tas.replay import IDLE

    target = _stage_key(to)
    ids = _legs_to(target)
    bodies: dict[str, list[list[int]]] = {}
    seeds: dict[str, str] = {}
    for sid in ids:
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

    for i, sid in enumerate(ids):
        leg = get_leg(sid)
        sk = _sid_key(sid)
        if i > 0:
            wait, ctrl = _idle_until(
                env,
                leg.control,
                label=f"wait_{sk}",
                start_frame=frame,
                max_wait=max_wait,
                on_step=on_step,
            )
            frame += wait
            stages[f"ctrl_wait_{sk}"] = wait
            stages[f"ctrl_{sk}"] = _snap_brief(ctrl)
            if not leg.control(ctrl):
                timeout = (
                    "surface_control_timeout" if sid == "1-2" else f"{sk}_control_timeout"
                )
                return _chain_result(False, timeout, target, settle, frame, stages)
            start_lives = int(ctrl.lives)

        stop = leg.leave if leg.stop_on_leave else None
        st = _play_body(
            env,
            bodies[sid],
            label=_body_label(sid),
            start_frame=frame,
            start_lives=start_lives,
            stop=stop,
            on_step=on_step,
        )
        frame = int(st["end_frame"])
        stages[sk] = st
        if st["death"] is not None:
            return _chain_result(False, f"death_{sk}", target, settle, frame, stages)

        if sid == "1-1" and target == "1-1":
            ok = st["death"] is None and int(st["max_x"]) >= 2500
            return _chain_result(
                ok, "clear_1_1" if ok else "1_1_incomplete", target, settle, frame, stages
            )

        if sid != target:
            continue

        end = read_snapshot(env.get_ram(), frame=frame)
        if stop is not None and not leg.leave(end):
            wait, end = _idle_until(
                env,
                leg.leave,
                label=f"wait_{_sid_key(leg.leave_id) or 'leave'}",
                start_frame=frame,
                max_wait=max_wait,
                on_step=on_step,
            )
            frame += wait
        ok = True if stop is None else leg.leave(end)
        outcome = leg.leave_outcome
        stages[f"ctrl_{_sid_key(leg.leave_id) or 'leave'}"] = _snap_brief(end)
        return _chain_result(
            ok, outcome if ok else f"missed_{outcome}", target, settle, frame, stages
        )

    return _chain_result(False, "empty_chain", target, settle, frame, stages)


def _chain_result(
    ok: bool,
    outcome: str,
    target: str,
    settle: int,
    frame: int,
    stages: dict[str, Any],
) -> dict[str, Any]:
    ctrls = [
        v
        for k, v in stages.items()
        if k.startswith("ctrl_") and not k.startswith("ctrl_wait_") and isinstance(v, dict)
    ]
    last_body = None
    for sid in reversed(tuple(leg.id for leg in WARPLESS_LEGS)):
        row = stages.get(_sid_key(sid))
        if isinstance(row, dict) and "end_snap" in row:
            last_body = row["end_snap"]
            break
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
        "end_snap": (ctrls[-1] if ctrls else last_body),
    }
