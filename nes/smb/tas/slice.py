"""Control-relative FM2 level slices for fceumm (HappyLee warps path).

Power-on full movies desync (longer blackout than FCEUX). Working path:

1. Clear prior stage with a verified body (e.g. HappyLee 1-1 slice).
2. Idle to a named control gate (``is_surface_control``, etc.).
3. Search even/odd FM2 indices near the expected movie offset.
4. Export the body as ``nes9_rle`` — **no L+R sanitize**.

Frame parity matters: after an odd control-wait, **odd** FM2 start indices
often clear while even die (hitbox / enemy phase).

Stage metadata and control predicates live in :mod:`smb.tas.stages`.
Replay primitives live in :mod:`smb.tas.replay`. Chain builders (reach_* /
verify_*) live in :mod:`smb.tas.chain`. Named probe/export/search wrappers
below keep existing call sites stable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from smb.paths import MODELS_DIR
from smb.policy import expand_nes9_rle, load_nes9_rle_seed
from smb.ram import read_snapshot, reached_ending
from smb.reactive_12 import is_surface_control
from smb.tas.fm2 import frames_to_nes9_rle_payload, parse_fm2
from smb.tas.replay import (
    get_state,
    idle_until,
    make_level1_env,
    play_body,
    set_state,
    to_action9,
)
from smb.tas.stages import (
    DEFAULT_FM2,
    DEFAULT_HL_1_1,
    DEFAULT_HL_1_2,
    DEFAULT_HL_4_1,
    DEFAULT_HL_4_2,
    FX_8_4_ENDING_FRAMES,
    FX_8_4_FM2_START,
    GoalKind,
    HL_1_1_NATURAL_SETTLE,
    HL_1_1_SETTLE,
    HL_1_2_FM2_START,
    HL_1_2_W4_FRAMES,
    HL_4_1_FM2_START,
    HL_4_1_LEAVE_FRAMES,
    HL_4_2_FM2_START,
    HL_4_2_W8_FRAMES,
    HL_8_1_CTRL_WAIT,
    HL_8_1_FM2_START,
    HL_8_1_LEAVE_FRAMES,
    HL_8_2_CTRL_WAIT,
    HL_8_2_FM2_START,
    HL_8_2_LEAVE_FRAMES,
    HL_8_3_FM2_START,
    HL_8_3_LEAVE_FRAMES,
    HL_8_3_SKILLS_LEAVE,
    HL_8_4_ENDING_FRAMES,
    HL_8_4_FM2_START,
    NAT_8_3_FOR_HL_START,
    NAT_8_3_TO_8_4_CONTROL,
    STAGE_1_2,
    STAGE_4_1,
    STAGE_4_2,
    STAGE_8_1,
    STAGE_8_2,
    STAGE_8_3,
    STAGE_8_4,
    STAGES,
    StageSpec,
    get_stage,
    goal_hit,
    is_4_1_control,
    is_4_2_control,
    is_8_1_control,
    is_8_2_control,
    is_8_3_control,
    is_8_4_control,
    is_dead,
    reached_world_8,
)

# Stage constants re-exported for ``from smb.tas.slice import HL_*`` callers.


# ---------------------------------------------------------------------------
# Probe result
# ---------------------------------------------------------------------------


@dataclass
class SliceProbe:
    """One FM2 start-index trial from a control gate."""

    start_idx: int
    max_x: int = 0
    death: int | None = None
    leave_frame: int | None = None
    ug: int | None = None
    exits: list[dict[str, Any]] = field(default_factory=list)
    leave_prior: int | None = None
    ctrl_wait: int | None = None

    @property
    def w4(self) -> int | None:
        """Legacy alias for :attr:`leave_frame` (historical field name)."""
        return self.leave_frame

    @w4.setter
    def w4(self, value: int | None) -> None:
        self.leave_frame = value

    @property
    def ok(self) -> bool:
        return self.leave_frame is not None and self.death is None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Keep legacy key in JSON reports for existing evidence parsers.
        d["w4"] = self.leave_frame
        return d


# ---------------------------------------------------------------------------
# Generic probe / export / search
# ---------------------------------------------------------------------------


def probe_from_control(
    env: Any,
    fm2_frames: list[list[int]],
    start_idx: int,
    *,
    stage: StageSpec | None = None,
    goal: GoalKind | None = None,
    goal_world: int | None = None,
    goal_level: int | None = None,
    track_ug: tuple[int, int] | None = None,
    max_play: int | None = None,
    start_lives: int | None = None,
    track_ending: bool = False,
) -> SliceProbe:
    """Replay FM2 from ``start_idx`` until goal / death / cap.

    Prefer passing a :class:`StageSpec`. Low-level goal fields remain for
    one-off probes (e.g. continuous tail helpers).
    """
    if stage is not None:
        goal = stage.goal
        goal_world = stage.goal_world
        goal_level = stage.goal_level
        track_ug = stage.track_ug
        if max_play is None:
            max_play = stage.max_play
        if stage.goal is GoalKind.ENDING:
            track_ending = True
    if goal is None:
        raise ValueError("probe_from_control requires stage= or goal=")
    if max_play is None:
        max_play = 4000

    if start_lives is None:
        start_lives = int(read_snapshot(env.get_ram(), 0).lives)
    body = fm2_frames[start_idx:]
    max_x = 0
    death: int | None = None
    leave: int | None = None
    ug: int | None = None
    exits: list[dict[str, Any]] = []
    snap0 = read_snapshot(env.get_ram(), 0)
    last = (int(snap0.world), int(snap0.level))
    ending: int | None = None

    for i in range(min(len(body), max_play)):
        env.step(to_action9(body[i]))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        key = (int(snap.world), int(snap.level))
        if key != last:
            exits.append({"i": i + 1, "from": list(last), "to": list(key), "x": px})
            if track_ug is not None and key == track_ug and ug is None:
                ug = i + 1
            if goal_hit(
                goal,
                snap=snap,
                ram=ram,
                key=key,
                goal_world=goal_world,
                goal_level=goal_level,
                start_lives=start_lives,
            ):
                leave = i + 1
                break
            last = key
        elif goal is GoalKind.ENDING or track_ending:
            if reached_ending(ram, start_lives=start_lives):
                ending = i + 1
                leave = i + 1
                break
        # WORLD goals can also hit without a key change edge if we started mid-transition
        if goal is GoalKind.WORLD and goal_hit(
            goal,
            snap=snap,
            ram=ram,
            key=key,
            goal_world=goal_world,
            goal_level=goal_level,
            start_lives=start_lives,
        ):
            leave = i + 1
            break
        if is_dead(snap, start_lives):
            death = i + 1
            break

    tr = SliceProbe(
        start_idx=start_idx,
        max_x=max_x,
        death=death,
        leave_frame=leave,
        ug=ug,
        exits=exits[:12],
    )
    if ending is not None:
        tr.leave_prior = ending
    return tr


def export_stage_slice(
    stage: StageSpec | str,
    *,
    fm2_path: Path = DEFAULT_FM2,
    start_idx: int | None = None,
    body_frames: int | None = None,
    out_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a stage body seed from FM2 ``[start_idx:start_idx+body_frames]``."""
    if isinstance(stage, str):
        stage = get_stage(stage)
    si = stage.fm2_start if start_idx is None else start_idx
    n = stage.body_frames if body_frames is None else body_frames
    fm2 = parse_fm2(fm2_path).frames
    frames = [list(f) for f in fm2[si : si + n]]
    meta = {
        "level_id": stage.resolved_route_id,
        "start_state": stage.start_state,
        "settle_frames": 0,
        "game_name": "SuperMarioBros-Nes-v0",
        "target": stage.target,
        "body_frames": n,
        "leave_frames": n,
        "fm2": str(fm2_path),
        "fm2_start_index": si,
        "predecessor": stage.predecessor,
        "note": stage.note,
        "stage_id": stage.id,
    }
    # Stage-specific verified flags (legacy seed metadata).
    if stage.id == "1-2":
        meta["verified_w4"] = True
        meta["w4_frames"] = n
    elif stage.id == "4-1":
        meta["verified_leave_4_2"] = True
    elif stage.id == "4-2":
        meta["verified_w8"] = True
        meta["w8_frames"] = n
    elif stage.id == "8-1":
        meta["verified_leave_8_2"] = True
    elif stage.id == "8-2":
        meta["verified_leave_8_3"] = True
    if extra:
        meta.update(extra)
    payload = frames_to_nes9_rle_payload(
        frames,
        route_id=stage.resolved_route_id,
        source=stage.source,
        extra=meta,
    )
    out = out_path or stage.seed_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["_path"] = str(out)
    return payload


def search_stage_offsets(
    stage: StageSpec | str,
    *,
    predecessor: Callable[[Any, list[list[int]]], dict[str, Any]],
    fm2_path: Path = DEFAULT_FM2,
    start_min: int | None = None,
    start_max: int | None = None,
    step: int | None = None,
    max_play: int | None = None,
    progress: Callable[[SliceProbe], None] | None = None,
    use_savestate: bool = True,
    lead_idles: range | None = None,
) -> dict[str, Any]:
    """Grid-search FM2 starts for *stage* after *predecessor(env, fm2)*.

    *predecessor* must leave ``env`` at the stage control gate and return a
    dict with at least ``success`` (bool). Optional ``ctrl_wait`` is attached
    to each probe.
    """
    if isinstance(stage, str):
        stage = get_stage(stage)
    s_min = stage.search_min if start_min is None else start_min
    s_max = stage.search_max if start_max is None else start_max
    s_step = stage.search_step if step is None else step
    m_play = stage.max_play if max_play is None else max_play
    lead_idles = lead_idles or range(0, 1)

    fm2 = parse_fm2(fm2_path).frames
    env = make_level1_env()
    pred = predecessor(env, fm2)
    if not pred.get("success"):
        env.close()
        return {"error": "predecessor_failed", "pred": pred, "stage": stage.id}

    ctrl_wait = pred.get("ctrl_wait")
    lives = int(read_snapshot(env.get_ram(), 0).lives)
    try:
        state = get_state(env) if use_savestate else None
    except AttributeError:
        state = None

    hits: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    n_trials = 0

    for lead in lead_idles:
        for si in range(s_min, s_max + 1, s_step):
            n_trials += 1
            if state is not None:
                set_state(env, state)
            else:
                env.close()
                env = make_level1_env()
                pred_i = predecessor(env, fm2)
                if not pred_i.get("success"):
                    continue
                ctrl_wait = pred_i.get("ctrl_wait")
                lives = int(read_snapshot(env.get_ram(), 0).lives)
            for _ in range(lead):
                env.step(IDLE)
            tr = probe_from_control(
                env,
                fm2,
                si,
                stage=stage,
                max_play=m_play,
                start_lives=lives,
            )
            tr.ctrl_wait = ctrl_wait
            row = {**tr.to_dict(), "lead_idle": lead}
            if progress and (tr.ok or (tr.max_x or 0) > 400 or tr.death):
                progress(tr)
            if tr.ok:
                hits.append(row)
                if best is None or (tr.leave_frame or 10**9) < (
                    best.get("leave_frame") or best.get("w4") or 10**9
                ):
                    best = row
            elif best is None or (
                best.get("leave_frame") is None
                and best.get("w4") is None
                and (tr.max_x or 0) > (best.get("max_x") or 0)
            ):
                if not (best and (best.get("leave_frame") or best.get("w4"))):
                    best = {**row, "_progress_only": True}

    env.close()
    return {
        "fm2": str(fm2_path),
        "stage": stage.id,
        "range": [s_min, s_max, s_step],
        "lead_idles": list(lead_idles),
        "ctrl_wait": ctrl_wait,
        "hits": hits,
        "best": best,
        "n_trials": n_trials,
        "n_hits": len(hits),
        "pred": {k: v for k, v in pred.items() if "snap" not in k and k != "hl"},
    }


# ---------------------------------------------------------------------------
# Named probe / export wrappers (thin StageSpec adapters)
# ---------------------------------------------------------------------------


def _stage_probe(
    stage: StageSpec,
    env: Any,
    fm2_frames: list[list[int]],
    start_idx: int,
    *,
    max_play: int | None = None,
    start_lives: int | None = None,
) -> SliceProbe:
    return probe_from_control(
        env,
        fm2_frames,
        start_idx,
        stage=stage,
        max_play=max_play,
        start_lives=start_lives,
    )


def probe_1_2_from_control(env, fm2_frames, start_idx, *, max_play=2200, start_lives=None):
    return _stage_probe(STAGE_1_2, env, fm2_frames, start_idx, max_play=max_play, start_lives=start_lives)


def probe_4_1_from_control(env, fm2_frames, start_idx, *, max_play=2800, start_lives=None):
    return _stage_probe(STAGE_4_1, env, fm2_frames, start_idx, max_play=max_play, start_lives=start_lives)


def probe_4_2_from_control(env, fm2_frames, start_idx, *, max_play=4000, start_lives=None):
    return _stage_probe(STAGE_4_2, env, fm2_frames, start_idx, max_play=max_play, start_lives=start_lives)


def probe_8_1_from_control(env, fm2_frames, start_idx, *, max_play=3500, start_lives=None):
    return _stage_probe(STAGE_8_1, env, fm2_frames, start_idx, max_play=max_play, start_lives=start_lives)


def probe_8_2_from_control(env, fm2_frames, start_idx, *, max_play=3500, start_lives=None):
    return _stage_probe(STAGE_8_2, env, fm2_frames, start_idx, max_play=max_play, start_lives=start_lives)


def probe_8_3_from_control(env, fm2_frames, start_idx, *, max_play=3500, start_lives=None):
    return _stage_probe(STAGE_8_3, env, fm2_frames, start_idx, max_play=max_play, start_lives=start_lives)


def probe_8_4_from_control(env, fm2_frames, start_idx, *, max_play=6000, start_lives=None):
    return _stage_probe(STAGE_8_4, env, fm2_frames, start_idx, max_play=max_play, start_lives=start_lives)


def probe_level_leave_from_control(
    env: Any,
    fm2_frames: list[list[int]],
    start_idx: int,
    *,
    from_level: int,
    to_level: int | None = None,
    to_world: int | None = None,
    max_play: int = 4000,
    start_lives: int | None = None,
    track_ending: bool = False,
) -> SliceProbe:
    """Generic W8-style leave probe (legacy signature)."""
    if track_ending:
        return probe_from_control(
            env, fm2_frames, start_idx, goal=GoalKind.ENDING,
            max_play=max_play, start_lives=start_lives, track_ending=True,
        )
    if to_world is not None and to_level is None:
        return probe_from_control(
            env, fm2_frames, start_idx, goal=GoalKind.WORLD,
            goal_world=to_world, max_play=max_play, start_lives=start_lives,
        )
    gw = to_world if to_world is not None else 7
    gl = to_level if to_level is not None else from_level + 1
    return probe_from_control(
        env, fm2_frames, start_idx, goal=GoalKind.LEVEL,
        goal_world=gw, goal_level=gl, max_play=max_play, start_lives=start_lives,
    )


def export_1_2_slice(*, fm2_path=DEFAULT_FM2, start_idx=HL_1_2_FM2_START,
                     w4_frames=HL_1_2_W4_FRAMES, out_path=None, extra=None):
    return export_stage_slice(STAGE_1_2, fm2_path=fm2_path, start_idx=start_idx,
                              body_frames=w4_frames, out_path=out_path, extra=extra)


def export_4_1_slice(*, fm2_path=DEFAULT_FM2, start_idx=HL_4_1_FM2_START,
                     leave_frames=HL_4_1_LEAVE_FRAMES, out_path=None, extra=None):
    return export_stage_slice(STAGE_4_1, fm2_path=fm2_path, start_idx=start_idx,
                              body_frames=leave_frames, out_path=out_path, extra=extra)


def export_4_2_slice(*, fm2_path=DEFAULT_FM2, start_idx=HL_4_2_FM2_START,
                     w8_frames=HL_4_2_W8_FRAMES, out_path=None, extra=None):
    return export_stage_slice(STAGE_4_2, fm2_path=fm2_path, start_idx=start_idx,
                              body_frames=w8_frames, out_path=out_path, extra=extra)


def export_8_1_slice(*, fm2_path=DEFAULT_FM2, start_idx=HL_8_1_FM2_START,
                     leave_frames=HL_8_1_LEAVE_FRAMES, out_path=None):
    return export_stage_slice(STAGE_8_1, fm2_path=fm2_path, start_idx=start_idx,
                              body_frames=leave_frames, out_path=out_path)


def export_8_2_slice(*, fm2_path=DEFAULT_FM2, start_idx=HL_8_2_FM2_START,
                     leave_frames=HL_8_2_LEAVE_FRAMES, out_path=None):
    return export_stage_slice(STAGE_8_2, fm2_path=fm2_path, start_idx=start_idx,
                              body_frames=leave_frames, out_path=out_path)


def export_w8_slice(
    *,
    route_id: str,
    start_idx: int,
    n_frames: int,
    start_state: str,
    target: str,
    fm2_path: Path = DEFAULT_FM2,
    out_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a free-form World-8 body seed (legacy)."""
    fm2 = parse_fm2(fm2_path).frames
    frames = [list(f) for f in fm2[start_idx : start_idx + n_frames]]
    meta = {
        "level_id": route_id,
        "start_state": start_state,
        "settle_frames": 0,
        "game_name": "SuperMarioBros-Nes-v0",
        "target": target,
        "body_frames": n_frames,
        "fm2": str(fm2_path),
        "fm2_start_index": start_idx,
        "note": "Control-relative World 8 HL body. Do not sanitize L+R.",
    }
    if extra:
        meta.update(extra)
    payload = frames_to_nes9_rle_payload(
        frames, route_id=route_id,
        source="HappyLee warps #1715M FM2 (HL W8 predecessor)", extra=meta,
    )
    out = out_path or (MODELS_DIR / f"{route_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["_path"] = str(out)
    return payload



# ---------------------------------------------------------------------------
# Search wrappers
# ---------------------------------------------------------------------------


def search_1_2_offsets(
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_min: int = 2080,
    start_max: int = 2140,
    step: int = 1,
    max_play: int = 2000,
    progress: Callable[[SliceProbe], None] | None = None,
) -> dict[str, Any]:
    """Fresh natural HL 1-1 → surface control for each trial; rank W4 clears."""
    from smb.tas.chain import reach_surface_after_hl_1_1

    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    hits: list[SliceProbe] = []
    best: SliceProbe | None = None

    for si in range(start_min, start_max + 1, step):
        env = make_level1_env()
        leave, wait, ctrl = reach_surface_after_hl_1_1(env, hl_frames=hl)
        tr = probe_1_2_from_control(
            env, fm2, si, max_play=max_play, start_lives=int(ctrl.lives)
        )
        tr.leave_prior = leave
        tr.ctrl_wait = wait
        env.close()
        if progress:
            progress(tr)
        if tr.leave_frame is not None:
            hits.append(tr)
            if best is None or (tr.leave_frame or 10**9) < (best.leave_frame or 10**9):
                best = tr

    return {
        "fm2": str(fm2_path),
        "hl_1_1": str(hl_1_1_path),
        "range": [start_min, start_max, step],
        "hits": [h.to_dict() for h in hits],
        "best": best.to_dict() if best else None,
        "n_trials": len(range(start_min, start_max + 1, step)),
    }


def search_4_1_offsets(
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_min: int = 3880,
    start_max: int = 4020,
    step: int = 2,
    max_play: int = 2600,
    progress: Callable[[SliceProbe], None] | None = None,
    use_savestate: bool = True,
) -> dict[str, Any]:
    from smb.tas.chain import reach_4_1_control_after_hl_w4

    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    hits: list[SliceProbe] = []
    best: SliceProbe | None = None

    env = make_level1_env()
    pred = reach_4_1_control_after_hl_w4(
        env, hl_1_1_frames=hl, fm2_frames=fm2, fm2_path=fm2_path
    )
    if not pred.get("success") or pred.get("control_snap") is None:
        env.close()
        return {
            "error": "predecessor_failed",
            "pred": {
                k: v for k, v in pred.items() if k not in ("probe", "control_snap")
            },
        }
    ctrl = pred["control_snap"]
    wait41 = pred["ctrl_wait_4_1"]
    try:
        state = get_state(env) if use_savestate else None
    except AttributeError:
        state = None
    lives = int(ctrl.lives)

    for si in range(start_min, start_max + 1, step):
        if use_savestate and state is not None:
            set_state(env, state)
        else:
            env.close()
            env = make_level1_env()
            pred_i = reach_4_1_control_after_hl_w4(
                env, hl_1_1_frames=hl, fm2_frames=fm2
            )
            wait41 = pred_i["ctrl_wait_4_1"]
            lives = int(pred_i["control_snap"].lives)
        tr = probe_4_1_from_control(env, fm2, si, max_play=max_play, start_lives=lives)
        tr.ctrl_wait = wait41
        tr.leave_prior = pred.get("leave_1_1")
        if progress:
            progress(tr)
        if tr.ok:
            hits.append(tr)
            if best is None or (tr.leave_frame or 10**9) < (best.leave_frame or 10**9):
                best = tr
    env.close()
    return {
        "fm2": str(fm2_path),
        "range": [start_min, start_max, step],
        "ctrl_wait_4_1": wait41,
        "hits": [h.to_dict() for h in hits],
        "best": best.to_dict() if best else None,
        "n_trials": len(range(start_min, start_max + 1, step)),
        "note": "savestate search; re-verify best with verify_4_1_4_2_natural_chain",
    }


def search_4_2_offsets(
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_4_1: int = HL_4_1_FM2_START,
    leave_4_1: int = HL_4_1_LEAVE_FRAMES,
    start_min: int = 6100,
    start_max: int = 6250,
    step: int = 2,
    max_play: int = 4000,
    progress: Callable[[SliceProbe], None] | None = None,
    use_savestate: bool = True,
) -> dict[str, Any]:
    from smb.tas.chain import reach_4_1_control_after_hl_w4

    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    hits: list[SliceProbe] = []
    best: SliceProbe | None = None

    env = make_level1_env()
    pred = reach_4_1_control_after_hl_w4(env, hl_1_1_frames=hl, fm2_frames=fm2)
    if not pred.get("success"):
        env.close()
        return {"error": "4_1_control_failed", "pred": pred}
    play_body(env, fm2, start=start_4_1, n=leave_4_1)
    wait42, snap = idle_until(env, is_4_2_control, max_wait=600)
    if not is_4_2_control(snap):
        env.close()
        return {"error": "4_2_control_timeout", "wait42": wait42}

    try:
        state = get_state(env) if use_savestate else None
    except AttributeError:
        state = None
    lives = int(snap.lives)

    for si in range(start_min, start_max + 1, step):
        if use_savestate and state is not None:
            set_state(env, state)
        tr = probe_4_2_from_control(env, fm2, si, max_play=max_play, start_lives=lives)
        tr.ctrl_wait = wait42
        if progress:
            progress(tr)
        if tr.ok:
            hits.append(tr)
            if best is None or (tr.leave_frame or 10**9) < (best.leave_frame or 10**9):
                best = tr
    env.close()
    return {
        "fm2": str(fm2_path),
        "range": [start_min, start_max, step],
        "ctrl_wait_4_2": wait42,
        "start_4_1": start_4_1,
        "leave_4_1": leave_4_1,
        "hits": [h.to_dict() for h in hits],
        "best": best.to_dict() if best else None,
        "n_trials": len(range(start_min, start_max + 1, step)),
    }


def search_8_3_offsets(
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_8_1: int = HL_8_1_FM2_START,
    start_8_2: int = HL_8_2_FM2_START,
    start_min: int = 13000,
    start_max: int = 13600,
    step: int = 1,
    max_play: int = 3200,
    lead_idles: range | None = None,
    progress: Callable[[SliceProbe], None] | None = None,
) -> dict[str, Any]:
    """After HL 8-2 leave → 8-3 control, grid-search FM2 starts (+ optional lead)."""

    from smb.tas.chain import reach_stage_control

    def _pred(env: Any, fm2: list[list[int]]) -> dict[str, Any]:
        # Rebuild uses full chain; fm2 arg unused (path-based).
        del fm2
        r = reach_stage_control(env, "8-3", fm2_path=fm2_path, hl_1_1_path=hl_1_1_path)
        if r.get("success"):
            r["ctrl_wait"] = r.get("ctrl_wait_8_3")
        return r

    report = search_stage_offsets(
        STAGE_8_3,
        predecessor=_pred,
        fm2_path=fm2_path,
        start_min=start_min,
        start_max=start_max,
        step=step,
        max_play=max_play,
        progress=progress,
        lead_idles=lead_idles,
        use_savestate=True,
    )
    # Preserve legacy key names used by evidence JSON.
    pred = report.get("pred") or {}
    report["ctrl_wait_8_1"] = pred.get("ctrl_wait_8_1")
    report["leave_8_1"] = pred.get("leave_8_1")
    report["ctrl_wait_8_2"] = pred.get("ctrl_wait_8_2")
    report["leave_8_2"] = pred.get("leave_8_2")
    report["ctrl_wait_8_3"] = report.get("ctrl_wait")
    if report.get("error") == "predecessor_failed":
        # Match old error strings when possible.
        stage = (report.get("pred") or {}).get("stage")
        if stage:
            report["error"] = f"{stage}_failed" if not str(stage).endswith("failed") else stage
    return report


