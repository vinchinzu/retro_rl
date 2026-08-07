"""Control-relative FM2 level slices for fceumm (HappyLee warps path).

Power-on full movies desync (longer blackout than FCEUX). Working path:

1. Clear prior stage with a verified body (e.g. HappyLee 1-1 slice).
2. Idle to a named control gate (``is_surface_control``, etc.).
3. Search even/odd FM2 indices near the expected movie offset.
4. Export the W4 / exit body as ``nes9_rle`` — **no L+R sanitize**.

Frame parity matters: after an odd control-wait, **odd** FM2 start indices
often clear while even die (hitbox / enemy phase). Same for even waits → even
starts (4-1 after even ``wait41``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from smb.paths import GAME_DIR, GAME_V0, MODELS_DIR
from smb.policy import expand_nes9_rle, load_nes9_rle_seed
from smb.ram import (
    ADDR_WORLD,
    PLAYER_STATE_DYING,
    WORLD_INDEX_8,
    read_snapshot,
    reached_world_4,
)
from smb.reactive_12 import is_surface_control
from smb.tas.fm2 import frames_to_nes9_rle_payload, parse_fm2

DEFAULT_FM2 = GAME_DIR / "tas" / "ref" / "happylee_warps_1715M.fm2"
DEFAULT_HL_1_1 = MODELS_DIR / "smb_1_1_happylee_slice.json"
DEFAULT_HL_1_2 = MODELS_DIR / "smb_1_2_happylee_slice.json"

# Isolated Level1_1 recipe for HappyLee body (metadata + practice).
HL_1_1_SETTLE = 2
# Natural power-on entry: odd settles clear (1,3,5…); default 1 matches run_1_1.
HL_1_1_NATURAL_SETTLE = 1

# Verified 1-2 W4 body after natural HL 1-1 → surface control (2026-08-07).
HL_1_2_FM2_START = 2109
HL_1_2_W4_FRAMES = 1657

# Verified 4-1 after HL W4 → 4-1 control (2026-08-07). Even start (even wait41).
HL_4_1_FM2_START = 3968
HL_4_1_LEAVE_FRAMES = 2062

# Verified 4-2 → W8 after HL 4-1 → 4-2 control (2026-08-07). Odd start (odd wait42).
HL_4_2_FM2_START = 6207
HL_4_2_W8_FRAMES = 1516


def _act(frame: list[int] | list) -> np.ndarray:
    action = np.zeros(9, dtype=np.int8)
    for j in range(min(9, len(frame))):
        action[j] = int(frame[j])
    return action


@dataclass
class SliceProbe:
    """One FM2 start-index trial from a control gate."""

    start_idx: int
    max_x: int = 0
    death: int | None = None
    w4: int | None = None
    ug: int | None = None
    exits: list[dict[str, Any]] = field(default_factory=list)
    leave_prior: int | None = None
    ctrl_wait: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reach_surface_after_hl_1_1(
    env,
    *,
    hl_frames: list[list[int]] | None = None,
    settle: int = HL_1_1_SETTLE,
) -> tuple[int | None, int, Any]:
    """Play HappyLee 1-1 from ``Level1_1``, idle to ``is_surface_control``.

    Returns ``(leave_1_1_frame, ctrl_wait_frames, control_snapshot)``.
    """
    if hl_frames is None:
        hl_frames = expand_nes9_rle(load_nes9_rle_seed(DEFAULT_HL_1_1))
    idle = np.zeros(9, dtype=np.int8)
    for _ in range(settle):
        env.step(idle)
    max_x = 0
    leave: int | None = None
    for i, fr in enumerate(hl_frames):
        env.step(_act(fr))
        snap = read_snapshot(env.get_ram(), i + 1)
        max_x = max(max_x, int(snap.player_x))
        if leave is None and max_x >= 2500 and int(snap.level_id) != 0:
            leave = i + 1
    wait = 0
    for _ in range(600):
        snap = read_snapshot(env.get_ram(), 0)
        if is_surface_control(snap):
            return leave, wait, snap
        env.step(idle)
        wait += 1
    return leave, wait, read_snapshot(env.get_ram(), 0)


def probe_1_2_from_control(
    env,
    fm2_frames: list[list[int]],
    start_idx: int,
    *,
    max_play: int = 2200,
    start_lives: int | None = None,
) -> SliceProbe:
    """Replay FM2 from ``start_idx`` until W4 / death / cap (env already at control)."""
    if start_lives is None:
        start_lives = int(read_snapshot(env.get_ram(), 0).lives)
    body = fm2_frames[start_idx:]
    max_x = 0
    death: int | None = None
    w4: int | None = None
    ug: int | None = None
    exits: list[dict[str, Any]] = []
    snap0 = read_snapshot(env.get_ram(), 0)
    last = (int(snap0.world), int(snap0.level))
    for i in range(min(len(body), max_play)):
        env.step(_act(body[i]))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        key = (int(snap.world), int(snap.level))
        if key != last:
            exits.append({"i": i + 1, "from": list(last), "to": list(key), "x": px})
            if key == (0, 2) and ug is None:
                ug = i + 1
        last = key
        if reached_world_4(ram):
            w4 = i + 1
            break
        if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
            death = i + 1
            break
    return SliceProbe(
        start_idx=start_idx,
        max_x=max_x,
        death=death,
        w4=w4,
        ug=ug,
        exits=exits[:8],
    )


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
    from retro_harness.env import make_env

    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    hits: list[SliceProbe] = []
    best: SliceProbe | None = None

    for si in range(start_min, start_max + 1, step):
        env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
        env.reset()
        leave, wait, ctrl = reach_surface_after_hl_1_1(env, hl_frames=hl)
        tr = probe_1_2_from_control(
            env, fm2, si, max_play=max_play, start_lives=int(ctrl.lives)
        )
        tr.leave_prior = leave
        tr.ctrl_wait = wait
        env.close()
        if progress:
            progress(tr)
        if tr.w4 is not None:
            hits.append(tr)
            if best is None or (tr.w4 or 10**9) < (best.w4 or 10**9):
                best = tr

    return {
        "fm2": str(fm2_path),
        "hl_1_1": str(hl_1_1_path),
        "range": [start_min, start_max, step],
        "hits": [h.to_dict() for h in hits],
        "best": best.to_dict() if best else None,
        "n_trials": len(range(start_min, start_max + 1, step)),
    }


def export_1_2_slice(
    *,
    fm2_path: Path = DEFAULT_FM2,
    start_idx: int = HL_1_2_FM2_START,
    w4_frames: int = HL_1_2_W4_FRAMES,
    out_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``smb_1_2_happylee_slice.json`` (or *out_path*) from FM2 indices."""
    fm2 = parse_fm2(fm2_path).frames
    frames = [list(f) for f in fm2[start_idx : start_idx + w4_frames]]
    meta = {
        "level_id": "smb_1_2",
        "start_state": "1-2_surface_control_after_happylee_1_1",
        "settle_frames": 0,
        "game_name": "SuperMarioBros-Nes-v0",
        "verified_w4": True,
        "w4_frames": w4_frames,
        "fm2": str(fm2_path),
        "fm2_start_index": start_idx,
        "predecessor": "smb_1_1_happylee_slice Level1_1 settle=2; idle to is_surface_control",
        "note": (
            "Control-relative 1-2 W4 warp. Do not sanitize L+R. "
            "Odd FM2 starts after odd ctrl_wait; re-search if 1-1 body changes."
        ),
    }
    if extra:
        meta.update(extra)
    payload = frames_to_nes9_rle_payload(
        frames,
        route_id="smb_1_2_happylee_slice",
        source="HappyLee warps #1715M FM2 (natural HL 1-1 predecessor)",
        extra=meta,
    )
    out = out_path or (MODELS_DIR / "smb_1_2_happylee_slice.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["_path"] = str(out)
    return payload


def verify_1_2_natural_chain(
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_idx: int = HL_1_2_FM2_START,
    max_play: int = 2200,
) -> dict[str, Any]:
    """One-shot: Level1_1 → HL 1-1 → surface control → FM2 body → W4 report."""
    from retro_harness.env import make_env

    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    leave, wait, ctrl = reach_surface_after_hl_1_1(env, hl_frames=hl)
    tr = probe_1_2_from_control(
        env, fm2, start_idx, max_play=max_play, start_lives=int(ctrl.lives)
    )
    tr.leave_prior = leave
    tr.ctrl_wait = wait
    env.close()
    total = (leave or 0) + wait + (tr.w4 or 0)
    return {
        **tr.to_dict(),
        "success": tr.w4 is not None and tr.death is None,
        "approx_total_to_w4": total,
        "vs_natural_82_w4": 3884,
        "delta_vs_natural_82_w4": 3884 - total if tr.w4 else None,
    }


# ---------------------------------------------------------------------------
# 4-1 / 4-2 (World 4 → World 8 warp)
# ---------------------------------------------------------------------------


def is_4_1_control(snap) -> bool:
    """Controllable 4-1 start after W4 pipe (timer live, low x)."""
    return (
        int(snap.world) == 3
        and int(snap.level) == 0
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and int(snap.timer) > 0
        and int(snap.player_x) < 200
    )


def is_4_2_control(snap) -> bool:
    """Controllable 4-2 surface start after 4-1 castle load.

    Timer is often **0** on the first controllable frame (matches natural
    entry fingerprints) — do not require timer > 0.
    """
    return (
        int(snap.world) == 3
        and int(snap.level) == 1
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and 20 <= int(snap.player_x) <= 80
    )


def reached_world_8(ram) -> bool:
    """True when warp-zone pipe delivered Mario to World 8."""
    return int(ram[ADDR_WORLD]) == WORLD_INDEX_8


def reach_w4_after_hl(
    env,
    *,
    hl_1_1_frames: list[list[int]] | None = None,
    fm2_frames: list[list[int]] | None = None,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_1_2: int = HL_1_2_FM2_START,
    max_1_2: int = HL_1_2_W4_FRAMES + 50,
) -> dict[str, Any]:
    """Level1_1 → HL 1-1 → surface → HL 1-2 → W4. Env ends on W4 entry frame."""
    if hl_1_1_frames is None:
        hl_1_1_frames = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    if fm2_frames is None:
        fm2_frames = parse_fm2(fm2_path).frames
    leave, wait12, ctrl = reach_surface_after_hl_1_1(env, hl_frames=hl_1_1_frames)
    tr = probe_1_2_from_control(
        env, fm2_frames, start_1_2, max_play=max_1_2, start_lives=int(ctrl.lives)
    )
    return {
        "leave_1_1": leave,
        "ctrl_wait_1_2": wait12,
        "w4": tr.w4,
        "death": tr.death,
        "success": tr.w4 is not None and tr.death is None,
        "probe": tr,
    }


def reach_4_1_control_after_hl_w4(
    env,
    *,
    max_wait: int = 800,
    **w4_kwargs: Any,
) -> dict[str, Any]:
    """After :func:`reach_w4_after_hl`, idle to ``is_4_1_control``."""
    idle = np.zeros(9, dtype=np.int8)
    base = reach_w4_after_hl(env, **w4_kwargs)
    if not base["success"]:
        return {**base, "ctrl_wait_4_1": None, "control_snap": None}
    wait = 0
    snap = read_snapshot(env.get_ram(), 0)
    for _ in range(max_wait):
        snap = read_snapshot(env.get_ram(), 0)
        if is_4_1_control(snap):
            return {
                **base,
                "ctrl_wait_4_1": wait,
                "control_snap": snap,
            }
        env.step(idle)
        wait += 1
    return {
        **base,
        "ctrl_wait_4_1": wait,
        "control_snap": snap,
        "success": False,
        "death": base.get("death") or "4_1_control_timeout",
    }


def probe_4_1_from_control(
    env,
    fm2_frames: list[list[int]],
    start_idx: int,
    *,
    max_play: int = 2800,
    start_lives: int | None = None,
) -> SliceProbe:
    """Replay FM2 from ``start_idx`` until 4-2 load / death (env at 4-1 control)."""
    if start_lives is None:
        start_lives = int(read_snapshot(env.get_ram(), 0).lives)
    body = fm2_frames[start_idx:]
    max_x = 0
    death: int | None = None
    leave: int | None = None
    exits: list[dict[str, Any]] = []
    snap0 = read_snapshot(env.get_ram(), 0)
    last = (int(snap0.world), int(snap0.level))
    for i in range(min(len(body), max_play)):
        env.step(_act(body[i]))
        snap = read_snapshot(env.get_ram(), i + 1)
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        key = (int(snap.world), int(snap.level))
        if key != last:
            exits.append({"i": i + 1, "from": list(last), "to": list(key), "x": px})
            if leave is None and key == (3, 1):
                leave = i + 1
                break
            last = key
        if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
            death = i + 1
            break
    return SliceProbe(
        start_idx=start_idx,
        max_x=max_x,
        death=death,
        w4=leave,  # reuse field: frames to leave stage (here → 4-2)
        exits=exits[:8],
    )


def probe_4_2_from_control(
    env,
    fm2_frames: list[list[int]],
    start_idx: int,
    *,
    max_play: int = 4000,
    start_lives: int | None = None,
) -> SliceProbe:
    """Replay FM2 from ``start_idx`` until World 8 / death (env at 4-2 control)."""
    if start_lives is None:
        start_lives = int(read_snapshot(env.get_ram(), 0).lives)
    body = fm2_frames[start_idx:]
    max_x = 0
    death: int | None = None
    w8: int | None = None
    ug: int | None = None
    exits: list[dict[str, Any]] = []
    snap0 = read_snapshot(env.get_ram(), 0)
    last = (int(snap0.world), int(snap0.level))
    for i in range(min(len(body), max_play)):
        env.step(_act(body[i]))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        key = (int(snap.world), int(snap.level))
        if key != last:
            exits.append({"i": i + 1, "from": list(last), "to": list(key), "x": px})
            if key == (3, 2) and ug is None:
                ug = i + 1
            if int(snap.world) == WORLD_INDEX_8:
                w8 = i + 1
                break
            last = key
        if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
            death = i + 1
            break
    return SliceProbe(
        start_idx=start_idx,
        max_x=max_x,
        death=death,
        w4=w8,  # frames to W8 entry
        ug=ug,
        exits=exits[:8],
    )


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
    """Search FM2 starts for 4-1 clear after HL W4 → 4-1 control.

    Default step 2 keeps parity with the measured even ``ctrl_wait_4_1``.
    Savestate mode rebuilds the predecessor once (faster); always re-verify
    winners with a fresh chain before promoting indices.
    """
    from retro_harness.env import make_env

    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    hits: list[SliceProbe] = []
    best: SliceProbe | None = None

    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    pred = reach_4_1_control_after_hl_w4(
        env, hl_1_1_frames=hl, fm2_frames=fm2, fm2_path=fm2_path
    )
    if not pred.get("success") or pred.get("control_snap") is None:
        env.close()
        return {
            "error": "predecessor_failed",
            "pred": {k: v for k, v in pred.items() if k != "probe" and k != "control_snap"},
        }
    ctrl = pred["control_snap"]
    wait41 = pred["ctrl_wait_4_1"]
    state = env.em.get_state() if use_savestate else None
    lives = int(ctrl.lives)

    for si in range(start_min, start_max + 1, step):
        if use_savestate and state is not None:
            env.em.set_state(state)
        else:
            env.close()
            env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
            env.reset()
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
        if tr.w4 is not None and tr.death is None:
            hits.append(tr)
            if best is None or (tr.w4 or 10**9) < (best.w4 or 10**9):
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
    """Search FM2 starts for W8 after HL 4-1 → idle to 4-2 control."""
    from retro_harness.env import make_env

    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    idle = np.zeros(9, dtype=np.int8)
    hits: list[SliceProbe] = []
    best: SliceProbe | None = None

    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    pred = reach_4_1_control_after_hl_w4(
        env, hl_1_1_frames=hl, fm2_frames=fm2
    )
    if not pred.get("success"):
        env.close()
        return {"error": "4_1_control_failed", "pred": pred}
    lives41 = int(pred["control_snap"].lives)
    body41 = fm2[start_4_1 : start_4_1 + leave_4_1]
    for fr in body41:
        env.step(_act(fr))
    wait42 = 0
    snap = read_snapshot(env.get_ram(), 0)
    for _ in range(600):
        snap = read_snapshot(env.get_ram(), 0)
        if is_4_2_control(snap):
            break
        env.step(idle)
        wait42 += 1
    else:
        env.close()
        return {"error": "4_2_control_timeout", "wait42": wait42}

    state = env.em.get_state() if use_savestate else None
    lives = int(snap.lives)

    for si in range(start_min, start_max + 1, step):
        if use_savestate and state is not None:
            env.em.set_state(state)
        tr = probe_4_2_from_control(env, fm2, si, max_play=max_play, start_lives=lives)
        tr.ctrl_wait = wait42
        if progress:
            progress(tr)
        if tr.w4 is not None and tr.death is None:
            hits.append(tr)
            if best is None or (tr.w4 or 10**9) < (best.w4 or 10**9):
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


def export_4_1_slice(
    *,
    fm2_path: Path = DEFAULT_FM2,
    start_idx: int = HL_4_1_FM2_START,
    leave_frames: int = HL_4_1_LEAVE_FRAMES,
    out_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``smb_4_1_happylee_slice.json`` from FM2 indices."""
    fm2 = parse_fm2(fm2_path).frames
    frames = [list(f) for f in fm2[start_idx : start_idx + leave_frames]]
    meta = {
        "level_id": "smb_4_1",
        "start_state": "4-1_control_after_happylee_w4",
        "settle_frames": 0,
        "game_name": "SuperMarioBros-Nes-v0",
        "verified_leave_4_2": True,
        "leave_frames": leave_frames,
        "fm2": str(fm2_path),
        "fm2_start_index": start_idx,
        "predecessor": (
            "HL 1-1 + surface + HL 1-2 W4 + idle to is_4_1_control "
            "(even ctrl_wait → even FM2 start)"
        ),
        "note": (
            "Control-relative 4-1. Do not sanitize L+R. "
            "Re-search if W4 predecessor timing changes."
        ),
    }
    if extra:
        meta.update(extra)
    payload = frames_to_nes9_rle_payload(
        frames,
        route_id="smb_4_1_happylee_slice",
        source="HappyLee warps #1715M FM2 (HL W4 predecessor)",
        extra=meta,
    )
    out = out_path or (MODELS_DIR / "smb_4_1_happylee_slice.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["_path"] = str(out)
    return payload


def export_4_2_slice(
    *,
    fm2_path: Path = DEFAULT_FM2,
    start_idx: int = HL_4_2_FM2_START,
    w8_frames: int = HL_4_2_W8_FRAMES,
    out_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``smb_4_2_happylee_slice.json`` from FM2 indices."""
    fm2 = parse_fm2(fm2_path).frames
    frames = [list(f) for f in fm2[start_idx : start_idx + w8_frames]]
    meta = {
        "level_id": "smb_4_2",
        "start_state": "4-2_control_after_happylee_4_1",
        "settle_frames": 0,
        "game_name": "SuperMarioBros-Nes-v0",
        "verified_w8": True,
        "w8_frames": w8_frames,
        "target": "world_8_entry",
        "fm2": str(fm2_path),
        "fm2_start_index": start_idx,
        "predecessor": (
            "HL 4-1 body + idle to is_4_2_control "
            "(odd ctrl_wait → odd FM2 start; timer often 0 at gate)"
        ),
        "note": (
            "Control-relative 4-2 → W8 warp. Do not sanitize L+R. "
            "Gate does not require timer>0."
        ),
    }
    if extra:
        meta.update(extra)
    payload = frames_to_nes9_rle_payload(
        frames,
        route_id="smb_4_2_happylee_slice",
        source="HappyLee warps #1715M FM2 (HL 4-1 predecessor)",
        extra=meta,
    )
    out = out_path or (MODELS_DIR / "smb_4_2_happylee_slice.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["_path"] = str(out)
    return payload


def verify_4_1_4_2_natural_chain(
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_4_1: int = HL_4_1_FM2_START,
    start_4_2: int = HL_4_2_FM2_START,
    max_4_1: int = 2800,
    max_4_2: int = 4000,
) -> dict[str, Any]:
    """Fresh Level1_1 chain: HL 1-1 → 1-2 W4 → 4-1 → 4-2 → W8."""
    from retro_harness.env import make_env

    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    idle = np.zeros(9, dtype=np.int8)
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()

    pred = reach_4_1_control_after_hl_w4(
        env, hl_1_1_frames=hl, fm2_frames=fm2, fm2_path=fm2_path
    )
    if not pred.get("success") or pred.get("control_snap") is None:
        env.close()
        return {
            "success": False,
            "stage": "4_1_control",
            **{k: v for k, v in pred.items() if k not in ("probe", "control_snap")},
        }

    lives = int(pred["control_snap"].lives)
    tr41 = probe_4_1_from_control(
        env, fm2, start_4_1, max_play=max_4_1, start_lives=lives
    )
    if tr41.w4 is None or tr41.death is not None:
        env.close()
        return {
            "success": False,
            "stage": "4_1_body",
            "leave_1_1": pred.get("leave_1_1"),
            "ctrl_wait_1_2": pred.get("ctrl_wait_1_2"),
            "w4": pred.get("w4"),
            "ctrl_wait_4_1": pred.get("ctrl_wait_4_1"),
            "probe_4_1": tr41.to_dict(),
        }

    wait42 = 0
    snap = read_snapshot(env.get_ram(), 0)
    for _ in range(600):
        snap = read_snapshot(env.get_ram(), 0)
        if is_4_2_control(snap):
            break
        env.step(idle)
        wait42 += 1
    else:
        env.close()
        return {
            "success": False,
            "stage": "4_2_control",
            "leave_4_1": tr41.w4,
            "ctrl_wait_4_1": pred.get("ctrl_wait_4_1"),
            "ctrl_wait_4_2": wait42,
        }

    tr42 = probe_4_2_from_control(
        env, fm2, start_4_2, max_play=max_4_2, start_lives=int(snap.lives)
    )
    env.close()

    leave11 = pred.get("leave_1_1") or 0
    wait12 = pred.get("ctrl_wait_1_2") or 0
    w4 = pred.get("w4") or 0
    wait41 = pred.get("ctrl_wait_4_1") or 0
    leave41 = tr41.w4 or 0
    w8 = tr42.w4 or 0
    total = leave11 + wait12 + w4 + wait41 + leave41 + wait42 + w8
    ok = tr42.w4 is not None and tr42.death is None
    return {
        "success": ok,
        "leave_1_1": leave11,
        "ctrl_wait_1_2": wait12,
        "w4": w4,
        "ctrl_wait_4_1": wait41,
        "si_4_1": start_4_1,
        "leave_4_1": leave41,
        "ctrl_wait_4_2": wait42,
        "si_4_2": start_4_2,
        "w8": w8,
        "ug_4_2": tr42.ug,
        "death_4_1": tr41.death,
        "death_4_2": tr42.death,
        "exits_4_1": tr41.exits,
        "exits_4_2": tr42.exits,
        "approx_total_to_w8": total if ok else None,
        "vs_natural_82_8_1": 12628,
        "delta_vs_natural_82_8_1": 12628 - total if ok else None,
        "vs_natural_82_4_1": 6198,
        "approx_total_to_4_2_load": leave11 + wait12 + w4 + wait41 + leave41,
        "delta_vs_natural_82_4_1": 6198 - (leave11 + wait12 + w4 + wait41 + leave41),
    }


# ---------------------------------------------------------------------------
# World 8 (8-1 … 8-4) after HL W8 entry
# ---------------------------------------------------------------------------

# Probe-verified 2026-08-07 (leave only; 8-3/8-4 open until phase-matched).
HL_8_1_FM2_START = 7930
HL_8_1_LEAVE_FRAMES = 2881
HL_8_1_CTRL_WAIT = 209  # odd wait; even FM2 starts clear

HL_8_2_FM2_START = 10910
HL_8_2_LEAVE_FRAMES = 2209
HL_8_2_CTRL_WAIT = 165

# Placeholders until phase search promotes.
HL_8_3_FM2_START: int | None = None
HL_8_3_LEAVE_FRAMES: int | None = None
HL_8_4_FM2_START: int | None = None
HL_8_4_ENDING_FRAMES: int | None = None

DEFAULT_HL_4_1 = MODELS_DIR / "smb_4_1_happylee_slice.json"
DEFAULT_HL_4_2 = MODELS_DIR / "smb_4_2_happylee_slice.json"


def is_8_1_control(snap) -> bool:
    """Controllable 8-1 after W8 pipe (state 7/8, low x, timer live)."""
    return (
        int(snap.world) == WORLD_INDEX_8
        and int(snap.level) == 0
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and int(snap.timer) > 0
        and int(snap.player_x) < 120
    )


def is_8_2_control(snap) -> bool:
    """Controllable 8-2 start after 8-1 castle load."""
    return (
        int(snap.world) == WORLD_INDEX_8
        and int(snap.level) == 1
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and 20 <= int(snap.player_x) <= 120
    )


def is_8_3_control(snap) -> bool:
    """Controllable 8-3 start after 8-2 castle load."""
    return (
        int(snap.world) == WORLD_INDEX_8
        and int(snap.level) == 2
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and 20 <= int(snap.player_x) <= 120
    )


def is_8_4_control(snap) -> bool:
    """Controllable 8-4 start after 8-3 castle load."""
    return (
        int(snap.world) == WORLD_INDEX_8
        and int(snap.level) == 3
        and int(snap.oper_mode) == 1
        and int(snap.player_state) in (7, 8)
        and not snap.dying
        and 20 <= int(snap.player_x) <= 200
    )


def _snap_fp(snap) -> dict[str, int]:
    return {
        "world": int(snap.world),
        "level": int(snap.level),
        "area_pointer": int(getattr(snap, "area_pointer", -1) or -1),
        "oper_mode": int(snap.oper_mode),
        "player_state": int(snap.player_state),
        "player_x": int(snap.player_x),
        "player_y": int(snap.player_y),
        "timer": int(snap.timer),
        "lives": int(snap.lives),
    }


def reach_w8_after_hl(
    env,
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_4_1: int = HL_4_1_FM2_START,
    start_4_2: int = HL_4_2_FM2_START,
    use_seed_bodies: bool = True,
) -> dict[str, Any]:
    """Level1_1 → HL through 4-2 → W8. Prefer exported seeds when present."""
    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    idle = np.zeros(9, dtype=np.int8)

    pred = reach_4_1_control_after_hl_w4(
        env, hl_1_1_frames=hl, fm2_frames=fm2, fm2_path=fm2_path
    )
    if not pred.get("success") or pred.get("control_snap") is None:
        return {
            "success": False,
            "stage": "4_1_control",
            **{k: v for k, v in pred.items() if k not in ("probe", "control_snap")},
        }

    if use_seed_bodies and DEFAULT_HL_4_1.exists():
        body41 = expand_nes9_rle(load_nes9_rle_seed(DEFAULT_HL_4_1))
        lives = int(pred["control_snap"].lives)
        leave41 = None
        death = None
        for i, fr in enumerate(body41):
            env.step(_act(fr))
            snap = read_snapshot(env.get_ram(), i + 1)
            if int(snap.lives) < lives or int(snap.player_state) == PLAYER_STATE_DYING:
                death = i + 1
                break
            if int(snap.world) == 3 and int(snap.level) == 1:
                leave41 = i + 1
                break
        if death is not None or leave41 is None:
            return {
                "success": False,
                "stage": "4_1_body",
                "leave_4_1": leave41,
                "death": death,
                **{k: pred.get(k) for k in ("leave_1_1", "ctrl_wait_1_2", "w4", "ctrl_wait_4_1")},
            }
    else:
        tr41 = probe_4_1_from_control(
            env, fm2, start_4_1, max_play=2800, start_lives=int(pred["control_snap"].lives)
        )
        if tr41.w4 is None or tr41.death is not None:
            return {
                "success": False,
                "stage": "4_1_body",
                "probe_4_1": tr41.to_dict(),
                **{k: pred.get(k) for k in ("leave_1_1", "ctrl_wait_1_2", "w4", "ctrl_wait_4_1")},
            }
        leave41 = tr41.w4

    wait42 = 0
    snap = read_snapshot(env.get_ram(), 0)
    for _ in range(600):
        snap = read_snapshot(env.get_ram(), 0)
        if is_4_2_control(snap):
            break
        env.step(idle)
        wait42 += 1
    else:
        return {
            "success": False,
            "stage": "4_2_control",
            "leave_4_1": leave41,
            "ctrl_wait_4_2": wait42,
        }

    if use_seed_bodies and DEFAULT_HL_4_2.exists():
        body42 = expand_nes9_rle(load_nes9_rle_seed(DEFAULT_HL_4_2))
        lives = int(snap.lives)
        w8 = None
        death = None
        for i, fr in enumerate(body42):
            env.step(_act(fr))
            ram = env.get_ram()
            snap = read_snapshot(ram, i + 1)
            if int(snap.lives) < lives or int(snap.player_state) == PLAYER_STATE_DYING:
                death = i + 1
                break
            if int(snap.world) == WORLD_INDEX_8:
                w8 = i + 1
                break
        if death is not None or w8 is None:
            return {
                "success": False,
                "stage": "4_2_body",
                "leave_4_1": leave41,
                "ctrl_wait_4_2": wait42,
                "w8": w8,
                "death": death,
            }
    else:
        tr42 = probe_4_2_from_control(
            env, fm2, start_4_2, max_play=4000, start_lives=int(snap.lives)
        )
        if tr42.w4 is None or tr42.death is not None:
            return {
                "success": False,
                "stage": "4_2_body",
                "leave_4_1": leave41,
                "ctrl_wait_4_2": wait42,
                "probe_4_2": tr42.to_dict(),
            }
        w8 = tr42.w4

    return {
        "success": True,
        "leave_1_1": pred.get("leave_1_1"),
        "ctrl_wait_1_2": pred.get("ctrl_wait_1_2"),
        "w4": pred.get("w4"),
        "ctrl_wait_4_1": pred.get("ctrl_wait_4_1"),
        "leave_4_1": leave41,
        "ctrl_wait_4_2": wait42,
        "w8": w8,
        "w8_snap": read_snapshot(env.get_ram(), 0),
    }


def reach_8_1_control_after_hl_w8(
    env,
    *,
    max_wait: int = 800,
    **w8_kwargs: Any,
) -> dict[str, Any]:
    """After :func:`reach_w8_after_hl`, idle to ``is_8_1_control``."""
    idle = np.zeros(9, dtype=np.int8)
    base = reach_w8_after_hl(env, **w8_kwargs)
    if not base.get("success"):
        return {**base, "ctrl_wait_8_1": None, "control_snap": None}
    wait = 0
    snap = read_snapshot(env.get_ram(), 0)
    for _ in range(max_wait):
        snap = read_snapshot(env.get_ram(), 0)
        if is_8_1_control(snap):
            return {**base, "ctrl_wait_8_1": wait, "control_snap": snap}
        env.step(idle)
        wait += 1
    return {
        **base,
        "ctrl_wait_8_1": wait,
        "control_snap": snap,
        "success": False,
        "stage": "8_1_control_timeout",
    }


def probe_level_leave_from_control(
    env,
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
    """Generic W8 stage probe: play FM2 until level/world change, death, or cap.

    ``to_level`` is 0-indexed within world 8 by default. For 8-4 axe use
    ``track_ending=True`` (oper_mode end on 8-4).
    """
    from smb.ram import reached_ending

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
        env.step(_act(body[i]))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        key = (int(snap.world), int(snap.level))
        if key != last:
            exits.append({"i": i + 1, "from": list(last), "to": list(key), "x": px})
            if to_world is not None and key[0] == to_world:
                leave = i + 1
                break
            if (
                to_level is not None
                and key[0] == WORLD_INDEX_8
                and key[1] == to_level
            ):
                leave = i + 1
                break
            if (
                to_level is None
                and to_world is None
                and key[0] == WORLD_INDEX_8
                and key[1] == from_level + 1
            ):
                leave = i + 1
                break
            last = key
        if track_ending and reached_ending(ram, start_lives=start_lives):
            ending = i + 1
            leave = i + 1
            break
        if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
            death = i + 1
            break
    tr = SliceProbe(
        start_idx=start_idx,
        max_x=max_x,
        death=death,
        w4=leave,
        ug=ug,
        exits=exits[:12],
    )
    if ending is not None:
        tr.leave_prior = ending  # stash ending frame
    return tr


def probe_8_1_from_control(
    env,
    fm2_frames: list[list[int]],
    start_idx: int,
    *,
    max_play: int = 3500,
    start_lives: int | None = None,
) -> SliceProbe:
    return probe_level_leave_from_control(
        env,
        fm2_frames,
        start_idx,
        from_level=0,
        to_level=1,
        max_play=max_play,
        start_lives=start_lives,
    )


def probe_8_2_from_control(
    env,
    fm2_frames: list[list[int]],
    start_idx: int,
    *,
    max_play: int = 3500,
    start_lives: int | None = None,
) -> SliceProbe:
    return probe_level_leave_from_control(
        env,
        fm2_frames,
        start_idx,
        from_level=1,
        to_level=2,
        max_play=max_play,
        start_lives=start_lives,
    )


def probe_8_3_from_control(
    env,
    fm2_frames: list[list[int]],
    start_idx: int,
    *,
    max_play: int = 3500,
    start_lives: int | None = None,
) -> SliceProbe:
    return probe_level_leave_from_control(
        env,
        fm2_frames,
        start_idx,
        from_level=2,
        to_level=3,
        max_play=max_play,
        start_lives=start_lives,
    )


def probe_8_4_from_control(
    env,
    fm2_frames: list[list[int]],
    start_idx: int,
    *,
    max_play: int = 6000,
    start_lives: int | None = None,
) -> SliceProbe:
    """Play until ending (axe) or death."""
    return probe_level_leave_from_control(
        env,
        fm2_frames,
        start_idx,
        from_level=3,
        to_level=None,
        max_play=max_play,
        start_lives=start_lives,
        track_ending=True,
    )


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
    """Write a World-8 body seed from FM2 [start_idx:start_idx+n_frames]."""
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
        "note": (
            "Control-relative World 8 HL body. Do not sanitize L+R. "
            "Re-search if predecessor timing/phase changes."
        ),
    }
    if extra:
        meta.update(extra)
    payload = frames_to_nes9_rle_payload(
        frames,
        route_id=route_id,
        source="HappyLee warps #1715M FM2 (HL W8 predecessor)",
        extra=meta,
    )
    out = out_path or (MODELS_DIR / f"{route_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["_path"] = str(out)
    return payload


def export_8_1_slice(
    *,
    fm2_path: Path = DEFAULT_FM2,
    start_idx: int = HL_8_1_FM2_START,
    leave_frames: int = HL_8_1_LEAVE_FRAMES,
    out_path: Path | None = None,
) -> dict[str, Any]:
    return export_w8_slice(
        route_id="smb_8_1_happylee_slice",
        start_idx=start_idx,
        n_frames=leave_frames,
        start_state="8-1_control_after_happylee_w8",
        target="8_2_load",
        fm2_path=fm2_path,
        out_path=out_path or (MODELS_DIR / "smb_8_1_happylee_slice.json"),
        extra={
            "verified_leave_8_2": True,
            "leave_frames": leave_frames,
            "predecessor": "HL chain to W8 + idle to is_8_1_control (wait≈209 odd; even FM2)",
        },
    )


def export_8_2_slice(
    *,
    fm2_path: Path = DEFAULT_FM2,
    start_idx: int = HL_8_2_FM2_START,
    leave_frames: int = HL_8_2_LEAVE_FRAMES,
    out_path: Path | None = None,
) -> dict[str, Any]:
    return export_w8_slice(
        route_id="smb_8_2_happylee_slice",
        start_idx=start_idx,
        n_frames=leave_frames,
        start_state="8-2_control_after_happylee_8_1",
        target="8_3_load",
        fm2_path=fm2_path,
        out_path=out_path or (MODELS_DIR / "smb_8_2_happylee_slice.json"),
        extra={
            "verified_leave_8_3": True,
            "leave_frames": leave_frames,
            "predecessor": "HL 8-1 + idle to is_8_2_control (wait≈165)",
        },
    )


def verify_8_1_8_2_natural_chain(
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_8_1: int = HL_8_1_FM2_START,
    start_8_2: int = HL_8_2_FM2_START,
    max_8_1: int = 3500,
    max_8_2: int = 3500,
) -> dict[str, Any]:
    """Fresh Level1_1: HL → W8 → 8-1 → 8-2 → 8-3 load."""
    from retro_harness.env import make_env

    fm2 = parse_fm2(fm2_path).frames
    idle = np.zeros(9, dtype=np.int8)
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()

    pred = reach_8_1_control_after_hl_w8(env, fm2_path=fm2_path, hl_1_1_path=hl_1_1_path)
    if not pred.get("success") or pred.get("control_snap") is None:
        env.close()
        return {
            "success": False,
            "stage": pred.get("stage") or "8_1_control",
            **{k: v for k, v in pred.items() if k not in ("probe", "control_snap", "w8_snap")},
        }

    lives = int(pred["control_snap"].lives)
    tr81 = probe_8_1_from_control(
        env, fm2, start_8_1, max_play=max_8_1, start_lives=lives
    )
    if tr81.w4 is None or tr81.death is not None:
        env.close()
        return {
            "success": False,
            "stage": "8_1_body",
            "ctrl_wait_8_1": pred.get("ctrl_wait_8_1"),
            "probe_8_1": tr81.to_dict(),
            "approx_total_to_w8": _sum_w8_pred(pred),
        }

    wait82 = 0
    snap = read_snapshot(env.get_ram(), 0)
    for _ in range(600):
        snap = read_snapshot(env.get_ram(), 0)
        if is_8_2_control(snap):
            break
        env.step(idle)
        wait82 += 1
    else:
        env.close()
        return {
            "success": False,
            "stage": "8_2_control",
            "leave_8_1": tr81.w4,
            "ctrl_wait_8_1": pred.get("ctrl_wait_8_1"),
            "ctrl_wait_8_2": wait82,
        }

    tr82 = probe_8_2_from_control(
        env, fm2, start_8_2, max_play=max_8_2, start_lives=int(snap.lives)
    )
    env.close()

    leave81 = tr81.w4 or 0
    leave82 = tr82.w4 or 0
    wait81 = pred.get("ctrl_wait_8_1") or 0
    base = _sum_w8_pred(pred)
    total = base + wait81 + leave81 + wait82 + leave82
    ok = tr82.w4 is not None and tr82.death is None
    return {
        "success": ok,
        **{k: pred.get(k) for k in (
            "leave_1_1", "ctrl_wait_1_2", "w4", "ctrl_wait_4_1",
            "leave_4_1", "ctrl_wait_4_2", "w8",
        )},
        "ctrl_wait_8_1": wait81,
        "si_8_1": start_8_1,
        "leave_8_1": leave81,
        "ctrl_wait_8_2": wait82,
        "si_8_2": start_8_2,
        "leave_8_2": leave82,
        "death_8_1": tr81.death,
        "death_8_2": tr82.death,
        "exits_8_1": tr81.exits,
        "exits_8_2": tr82.exits,
        "approx_total_to_w8": base,
        "approx_total_to_8_3_load": total if ok else None,
        "vs_natural_82_8_2": 15779,
        "delta_vs_natural_82_8_2": 15779 - total if ok else None,
        "control_8_1_fp": _snap_fp(pred["control_snap"]) if pred.get("control_snap") else None,
    }


def _sum_w8_pred(pred: dict[str, Any]) -> int:
    keys = (
        "leave_1_1",
        "ctrl_wait_1_2",
        "w4",
        "ctrl_wait_4_1",
        "leave_4_1",
        "ctrl_wait_4_2",
        "w8",
    )
    return sum(int(pred.get(k) or 0) for k in keys)


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
    """After HL 8-2 leave → 8-3 control, grid-search FM2 starts (+ optional lead idle).

    Rebuilds predecessor once (savestate), then each trial reloads that state.
    """
    from retro_harness.env import make_env

    fm2 = parse_fm2(fm2_path).frames
    idle = np.zeros(9, dtype=np.int8)
    lead_idles = lead_idles or range(0, 1)

    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    pred = reach_8_1_control_after_hl_w8(env, fm2_path=fm2_path, hl_1_1_path=hl_1_1_path)
    if not pred.get("success") or pred.get("control_snap") is None:
        env.close()
        return {"error": "8_1_control_failed", "pred": {k: v for k, v in pred.items() if k not in ("control_snap", "w8_snap")}}

    tr81 = probe_8_1_from_control(
        env, fm2, start_8_1, start_lives=int(pred["control_snap"].lives)
    )
    if tr81.w4 is None or tr81.death is not None:
        env.close()
        return {"error": "8_1_body_failed", "probe": tr81.to_dict()}

    wait82 = 0
    snap = read_snapshot(env.get_ram(), 0)
    for _ in range(600):
        snap = read_snapshot(env.get_ram(), 0)
        if is_8_2_control(snap):
            break
        env.step(idle)
        wait82 += 1
    else:
        env.close()
        return {"error": "8_2_control_timeout", "wait82": wait82}

    tr82 = probe_8_2_from_control(env, fm2, start_8_2, start_lives=int(snap.lives))
    if tr82.w4 is None or tr82.death is not None:
        env.close()
        return {"error": "8_2_body_failed", "probe": tr82.to_dict(), "wait82": wait82}

    wait83 = 0
    snap = read_snapshot(env.get_ram(), 0)
    for _ in range(600):
        snap = read_snapshot(env.get_ram(), 0)
        if is_8_3_control(snap):
            break
        env.step(idle)
        wait83 += 1
    else:
        env.close()
        return {
            "error": "8_3_control_timeout",
            "wait82": wait82,
            "wait83": wait83,
            "leave_8_2": tr82.w4,
        }

    ctrl_fp = _snap_fp(snap)
    lives = int(snap.lives)
    state_bytes = env.em.get_state() if hasattr(env, "em") and hasattr(env.em, "get_state") else None
    if state_bytes is None and hasattr(env, "unwrapped"):
        uw = env.unwrapped
        if hasattr(uw, "em") and hasattr(uw.em, "get_state"):
            state_bytes = uw.em.get_state()

    hits: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    n_trials = 0

    for lead in lead_idles:
        for si in range(start_min, start_max + 1, step):
            n_trials += 1
            if state_bytes is not None:
                if hasattr(env, "em") and hasattr(env.em, "set_state"):
                    env.em.set_state(state_bytes)
                else:
                    uw = env.unwrapped
                    uw.em.set_state(state_bytes)
            else:
                # expensive rebuild
                env.close()
                env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
                env.reset()
                reach_8_1_control_after_hl_w8(env, fm2_path=fm2_path, hl_1_1_path=hl_1_1_path)
                # This path is too slow for grid search without state — break
                env.close()
                return {
                    "error": "no_savestate",
                    "note": "emulator get_state required for 8-3 grid search",
                    "ctrl_wait_8_3": wait83,
                    "control_fp": ctrl_fp,
                }

            for _ in range(lead):
                env.step(idle)
            tr = probe_8_3_from_control(env, fm2, si, max_play=max_play, start_lives=lives)
            tr.ctrl_wait = wait83
            row = {**tr.to_dict(), "lead_idle": lead}
            if progress and (tr.w4 is not None or (tr.max_x or 0) > 400 or tr.death):
                progress(tr)
            if tr.w4 is not None and tr.death is None:
                hits.append(row)
                if best is None or (tr.w4 or 10**9) < (best.get("w4") or 10**9):
                    best = row
            elif best is None or (tr.max_x or 0) > (best.get("max_x") or 0):
                # keep best progress for diagnostics even without leave
                if tr.w4 is None:
                    if best is None or best.get("w4") is None:
                        if best is None or (tr.max_x or 0) > (best.get("max_x") or 0):
                            best = {**row, "_progress_only": True}

    env.close()
    return {
        "fm2": str(fm2_path),
        "range": [start_min, start_max, step],
        "lead_idles": list(lead_idles),
        "ctrl_wait_8_1": pred.get("ctrl_wait_8_1"),
        "leave_8_1": tr81.w4,
        "ctrl_wait_8_2": wait82,
        "leave_8_2": tr82.w4,
        "ctrl_wait_8_3": wait83,
        "control_8_3_fp": ctrl_fp,
        "hits": hits,
        "best": best,
        "n_trials": n_trials,
        "n_hits": len(hits),
    }


def verify_continuous_tail_from_8_1(
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_idx: int = HL_8_1_FM2_START,
    max_play: int | None = None,
) -> dict[str, Any]:
    """From 8-1 control, play FM2 continuously until ending/death (no re-gate)."""
    from retro_harness.env import make_env
    from smb.ram import reached_ending

    fm2 = parse_fm2(fm2_path).frames
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    pred = reach_8_1_control_after_hl_w8(env, fm2_path=fm2_path, hl_1_1_path=hl_1_1_path)
    if not pred.get("success") or pred.get("control_snap") is None:
        env.close()
        return {"success": False, "stage": "8_1_control", "pred": {k: v for k, v in pred.items() if "snap" not in k}}

    lives = int(pred["control_snap"].lives)
    body = fm2[start_idx:]
    limit = len(body) if max_play is None else min(len(body), max_play)
    exits: list[dict[str, Any]] = []
    snap0 = read_snapshot(env.get_ram(), 0)
    last = (int(snap0.world), int(snap0.level))
    death = None
    ending = None
    max_x = 0
    for i in range(limit):
        env.step(_act(body[i]))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        key = (int(snap.world), int(snap.level))
        if key != last:
            exits.append({"i": i + 1, "from": list(last), "to": list(key), "x": px, "t": int(snap.timer)})
            last = key
        if reached_ending(ram, start_lives=lives):
            ending = i + 1
            break
        if int(snap.lives) < lives or int(snap.player_state) == PLAYER_STATE_DYING:
            death = i + 1
            death_info = {
                "i": death,
                "w": int(snap.world) + 1,
                "l": int(snap.level) + 1,
                "x": px,
                "t": int(snap.timer),
                "ps": int(snap.player_state),
            }
            env.close()
            base = _sum_w8_pred(pred) + (pred.get("ctrl_wait_8_1") or 0)
            return {
                "success": False,
                "start_idx": start_idx,
                "death": death,
                "death_info": death_info,
                "exits": exits,
                "max_x": max_x,
                "ctrl_wait_8_1": pred.get("ctrl_wait_8_1"),
                "approx_frames_at_death": base + death,
            }

    env.close()
    base = _sum_w8_pred(pred) + (pred.get("ctrl_wait_8_1") or 0)
    return {
        "success": ending is not None,
        "start_idx": start_idx,
        "ending_body_frame": ending,
        "death": death,
        "exits": exits,
        "max_x": max_x,
        "ctrl_wait_8_1": pred.get("ctrl_wait_8_1"),
        "approx_total_to_ending": base + ending if ending else None,
        "vs_natural_82": 21559,
        "delta_vs_natural_82": (21559 - (base + ending)) if ending else None,
    }
