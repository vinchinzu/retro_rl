"""HL warps chain builders: Level1_1 → stage control gates and verify runs.

Probe/export/search stay in :mod:`smb.tas.slice`. This module owns the single
navigation path (reach_* / verify_*) so slice stays focused on FM2 slices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from smb.policy import expand_nes9_rle, load_nes9_rle_seed
from smb.ram import PLAYER_STATE_DYING, read_snapshot, reached_ending
from smb.reactive_12 import is_surface_control
from smb.tas.fm2 import parse_fm2
from smb.tas.replay import IDLE, idle_until, make_level1_env, to_action9
from smb.tas.slice import (
    SliceProbe,
    probe_1_2_from_control,
    probe_4_1_from_control,
    probe_4_2_from_control,
    probe_8_1_from_control,
    probe_8_2_from_control,
)
from smb.tas.stages import (
    DEFAULT_FM2,
    DEFAULT_HL_1_1,
    DEFAULT_HL_4_1,
    DEFAULT_HL_4_2,
    HL_1_1_SETTLE,
    HL_1_2_FM2_START,
    HL_1_2_W4_FRAMES,
    HL_4_1_FM2_START,
    HL_4_2_FM2_START,
    HL_8_1_FM2_START,
    HL_8_2_FM2_START,
    NAT82_TO_4_1_LEAVE,
    NAT82_TO_8_1_ENTRY,
    NAT82_TO_8_2_LEAVE,
    NAT82_TO_ENDING,
    NAT82_TO_W4,
    is_4_1_control,
    is_4_2_control,
    is_8_1_control,
    is_8_2_control,
    is_8_3_control,
    is_dead,
    snap_fingerprint,
)


def reach_surface_after_hl_1_1(
    env: Any,
    *,
    hl_frames: list[list[int]] | None = None,
    settle: int = HL_1_1_SETTLE,
) -> tuple[int | None, int, Any]:
    """Play HappyLee 1-1 from ``Level1_1``, idle to ``is_surface_control``.

    Returns ``(leave_1_1_frame, ctrl_wait_frames, control_snapshot)``.
    """
    if hl_frames is None:
        hl_frames = expand_nes9_rle(load_nes9_rle_seed(DEFAULT_HL_1_1))
    for _ in range(settle):
        env.step(IDLE)
    max_x = 0
    leave: int | None = None
    for i, fr in enumerate(hl_frames):
        env.step(to_action9(fr))
        snap = read_snapshot(env.get_ram(), i + 1)
        max_x = max(max_x, int(snap.player_x))
        if leave is None and max_x >= 2500 and int(snap.level_id) != 0:
            leave = i + 1
    wait, snap = idle_until(env, is_surface_control, max_wait=600)
    return leave, wait, snap


def reach_w4_after_hl(
    env: Any,
    *,
    hl_1_1_frames: list[list[int]] | None = None,
    fm2_frames: list[list[int]] | None = None,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_1_2: int = HL_1_2_FM2_START,
    max_1_2: int = HL_1_2_W4_FRAMES + 50,
) -> dict[str, Any]:
    """Level1_1 → HL 1-1 → surface → HL 1-2 → W4."""
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
        "w4": tr.leave_frame,
        "death": tr.death,
        "success": tr.ok,
        "probe": tr,
    }


def reach_4_1_control_after_hl_w4(
    env: Any,
    *,
    max_wait: int = 800,
    **w4_kwargs: Any,
) -> dict[str, Any]:
    """After :func:`reach_w4_after_hl`, idle to ``is_4_1_control``."""
    base = reach_w4_after_hl(env, **w4_kwargs)
    if not base["success"]:
        return {**base, "ctrl_wait_4_1": None, "control_snap": None}
    wait, snap = idle_until(env, is_4_1_control, max_wait=max_wait)
    ok = is_4_1_control(snap)
    return {
        **base,
        "ctrl_wait_4_1": wait,
        "control_snap": snap,
        "success": ok,
        "death": base.get("death") if ok else (base.get("death") or "4_1_control_timeout"),
    }


def _play_seed_or_probe(
    env: Any,
    *,
    seed_path: Path,
    fm2: list[list[int]],
    start_idx: int,
    max_play: int,
    start_lives: int,
    use_seed: bool,
    probe_fn: Callable[..., SliceProbe],
    leave_key_check: Callable[[Any], bool],
) -> tuple[int | None, int | None]:
    """Play exported seed body or FM2 probe; return (leave, death)."""
    if use_seed and seed_path.exists():
        body = expand_nes9_rle(load_nes9_rle_seed(seed_path))
        leave = None
        death = None
        for i, fr in enumerate(body):
            env.step(to_action9(fr))
            snap = read_snapshot(env.get_ram(), i + 1)
            if is_dead(snap, start_lives):
                death = i + 1
                break
            if leave_key_check(snap):
                leave = i + 1
                break
        return leave, death
    tr = probe_fn(env, fm2, start_idx, max_play=max_play, start_lives=start_lives)
    return tr.leave_frame, tr.death


def reach_w8_after_hl(
    env: Any,
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

    pred = reach_4_1_control_after_hl_w4(
        env, hl_1_1_frames=hl, fm2_frames=fm2, fm2_path=fm2_path
    )
    if not pred.get("success") or pred.get("control_snap") is None:
        return {
            "success": False,
            "stage": "4_1_control",
            **{k: v for k, v in pred.items() if k not in ("probe", "control_snap")},
        }

    leave41, death41 = _play_seed_or_probe(
        env,
        seed_path=DEFAULT_HL_4_1,
        fm2=fm2,
        start_idx=start_4_1,
        max_play=2800,
        start_lives=int(pred["control_snap"].lives),
        use_seed=use_seed_bodies,
        probe_fn=probe_4_1_from_control,
        leave_key_check=lambda s: int(s.world) == 3 and int(s.level) == 1,
    )
    if death41 is not None or leave41 is None:
        return {
            "success": False,
            "stage": "4_1_body",
            "leave_4_1": leave41,
            "death": death41,
            **{
                k: pred.get(k)
                for k in ("leave_1_1", "ctrl_wait_1_2", "w4", "ctrl_wait_4_1")
            },
        }

    wait42, snap42 = idle_until(env, is_4_2_control, max_wait=600)
    if not is_4_2_control(snap42):
        return {
            "success": False,
            "stage": "4_2_control",
            "leave_4_1": leave41,
            "ctrl_wait_4_2": wait42,
        }

    leave_w8, death42 = _play_seed_or_probe(
        env,
        seed_path=DEFAULT_HL_4_2,
        fm2=fm2,
        start_idx=start_4_2,
        max_play=4000,
        start_lives=int(snap42.lives),
        use_seed=use_seed_bodies,
        probe_fn=probe_4_2_from_control,
        leave_key_check=lambda s: int(s.world) == 7,  # WORLD_INDEX_8
    )
    if death42 is not None or leave_w8 is None:
        return {
            "success": False,
            "stage": "4_2_body",
            "leave_4_1": leave41,
            "ctrl_wait_4_2": wait42,
            "w8": leave_w8,
            "death": death42,
        }

    return {
        "success": True,
        "leave_1_1": pred.get("leave_1_1"),
        "ctrl_wait_1_2": pred.get("ctrl_wait_1_2"),
        "w4": pred.get("w4"),
        "ctrl_wait_4_1": pred.get("ctrl_wait_4_1"),
        "leave_4_1": leave41,
        "ctrl_wait_4_2": wait42,
        "w8": leave_w8,
        "w8_snap": read_snapshot(env.get_ram(), 0),
    }


def reach_8_1_control_after_hl_w8(
    env: Any,
    *,
    max_wait: int = 800,
    **w8_kwargs: Any,
) -> dict[str, Any]:
    """After :func:`reach_w8_after_hl`, idle to ``is_8_1_control``."""
    base = reach_w8_after_hl(env, **w8_kwargs)
    if not base.get("success"):
        return {**base, "ctrl_wait_8_1": None, "control_snap": None}
    wait, snap = idle_until(env, is_8_1_control, max_wait=max_wait)
    ok = is_8_1_control(snap)
    return {
        **base,
        "ctrl_wait_8_1": wait,
        "control_snap": snap,
        "success": ok,
        "stage": None if ok else "8_1_control_timeout",
    }


def reach_stage_control(
    env: Any,
    stage_id: str,
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
) -> dict[str, Any]:
    """Drive env from Level1_1 HL chain to the named stage control gate.

    Supported: ``1-2``, ``4-1``, ``4-2``, ``8-1``, ``8-2``, ``8-3``.
    """
    sid = stage_id.strip().lower().replace("_", "-")
    if sid == "1-2":
        leave, wait, snap = reach_surface_after_hl_1_1(env)
        return {
            "success": is_surface_control(snap),
            "ctrl_wait": wait,
            "leave_1_1": leave,
            "control_snap": snap,
        }
    if sid == "4-1":
        return reach_4_1_control_after_hl_w4(
            env, fm2_path=fm2_path, hl_1_1_path=hl_1_1_path
        )
    if sid in ("8-1", "8-2", "8-3"):
        pred = reach_8_1_control_after_hl_w8(
            env, fm2_path=fm2_path, hl_1_1_path=hl_1_1_path
        )
        if sid == "8-1":
            return pred
        if not pred.get("success") or pred.get("control_snap") is None:
            return pred
        fm2 = parse_fm2(fm2_path).frames
        lives = int(pred["control_snap"].lives)
        tr81 = probe_8_1_from_control(
            env, fm2, HL_8_1_FM2_START, start_lives=lives
        )
        if not tr81.ok:
            return {
                "success": False,
                "stage": "8_1_body",
                "probe": tr81.to_dict(),
                **{k: pred.get(k) for k in pred if "snap" not in k},
            }
        wait82, snap82 = idle_until(env, is_8_2_control)
        if not is_8_2_control(snap82):
            return {
                "success": False,
                "stage": "8_2_control",
                "leave_8_1": tr81.leave_frame,
                "ctrl_wait_8_2": wait82,
            }
        if sid == "8-2":
            return {
                **{k: pred.get(k) for k in pred if "snap" not in k},
                "success": True,
                "leave_8_1": tr81.leave_frame,
                "ctrl_wait_8_2": wait82,
                "control_snap": snap82,
                "ctrl_wait": wait82,
            }
        tr82 = probe_8_2_from_control(
            env, fm2, HL_8_2_FM2_START, start_lives=int(snap82.lives)
        )
        if not tr82.ok:
            return {
                "success": False,
                "stage": "8_2_body",
                "leave_8_1": tr81.leave_frame,
                "probe": tr82.to_dict(),
            }
        wait83, snap83 = idle_until(env, is_8_3_control)
        return {
            **{k: pred.get(k) for k in pred if "snap" not in k},
            "success": is_8_3_control(snap83),
            "leave_8_1": tr81.leave_frame,
            "ctrl_wait_8_2": wait82,
            "leave_8_2": tr82.leave_frame,
            "ctrl_wait_8_3": wait83,
            "control_snap": snap83,
            "ctrl_wait": wait83,
            "stage": None if is_8_3_control(snap83) else "8_3_control",
        }
    if sid == "4-2":
        pred = reach_4_1_control_after_hl_w4(
            env, fm2_path=fm2_path, hl_1_1_path=hl_1_1_path
        )
        if not pred.get("success"):
            return pred
        fm2 = parse_fm2(fm2_path).frames
        tr41 = probe_4_1_from_control(
            env,
            fm2,
            HL_4_1_FM2_START,
            start_lives=int(pred["control_snap"].lives),
        )
        if not tr41.ok:
            return {"success": False, "stage": "4_1_body", "probe": tr41.to_dict()}
        wait42, snap42 = idle_until(env, is_4_2_control)
        return {
            **{k: pred.get(k) for k in pred if "snap" not in k},
            "success": is_4_2_control(snap42),
            "leave_4_1": tr41.leave_frame,
            "ctrl_wait_4_2": wait42,
            "control_snap": snap42,
            "ctrl_wait": wait42,
        }
    raise KeyError(f"reach_stage_control: unsupported stage {stage_id!r}")


def verify_1_2_natural_chain(
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_idx: int = HL_1_2_FM2_START,
    max_play: int = 2200,
) -> dict[str, Any]:
    """One-shot: Level1_1 → HL 1-1 → surface control → FM2 body → W4 report."""
    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    env = make_level1_env()
    leave, wait, ctrl = reach_surface_after_hl_1_1(env, hl_frames=hl)
    tr = probe_1_2_from_control(
        env, fm2, start_idx, max_play=max_play, start_lives=int(ctrl.lives)
    )
    tr.leave_prior = leave
    tr.ctrl_wait = wait
    env.close()
    total = (leave or 0) + wait + (tr.leave_frame or 0)
    return {
        **tr.to_dict(),
        "success": tr.ok,
        "approx_total_to_w4": total,
        "vs_natural_82_w4": NAT82_TO_W4,
        "delta_vs_natural_82_w4": NAT82_TO_W4 - total if tr.ok else None,
    }


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
    hl = expand_nes9_rle(load_nes9_rle_seed(hl_1_1_path))
    fm2 = parse_fm2(fm2_path).frames
    env = make_level1_env()

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
    if not tr41.ok:
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

    wait42, snap42 = idle_until(env, is_4_2_control, max_wait=600)
    if not is_4_2_control(snap42):
        env.close()
        return {
            "success": False,
            "stage": "4_2_control",
            "leave_4_1": tr41.leave_frame,
            "ctrl_wait_4_2": wait42,
        }

    tr42 = probe_4_2_from_control(
        env, fm2, start_4_2, max_play=max_4_2, start_lives=int(snap42.lives)
    )
    env.close()

    leave11 = pred.get("leave_1_1") or 0
    wait12 = pred.get("ctrl_wait_1_2") or 0
    w4 = pred.get("w4") or 0
    wait41 = pred.get("ctrl_wait_4_1") or 0
    leave41 = tr41.leave_frame or 0
    w8 = tr42.leave_frame or 0
    total = leave11 + wait12 + w4 + wait41 + leave41 + wait42 + w8
    ok = tr42.ok
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
        "vs_natural_82_8_1": NAT82_TO_8_1_ENTRY,
        "delta_vs_natural_82_8_1": NAT82_TO_8_1_ENTRY - total if ok else None,
        "vs_natural_82_4_1": NAT82_TO_4_1_LEAVE,
        "approx_total_to_4_2_load": leave11 + wait12 + w4 + wait41 + leave41,
        "delta_vs_natural_82_4_1": NAT82_TO_4_1_LEAVE
        - (leave11 + wait12 + w4 + wait41 + leave41),
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
    fm2 = parse_fm2(fm2_path).frames
    env = make_level1_env()

    pred = reach_8_1_control_after_hl_w8(
        env, fm2_path=fm2_path, hl_1_1_path=hl_1_1_path
    )
    if not pred.get("success") or pred.get("control_snap") is None:
        env.close()
        return {
            "success": False,
            "stage": pred.get("stage") or "8_1_control",
            **{
                k: v
                for k, v in pred.items()
                if k not in ("probe", "control_snap", "w8_snap")
            },
        }

    lives = int(pred["control_snap"].lives)
    tr81 = probe_8_1_from_control(
        env, fm2, start_8_1, max_play=max_8_1, start_lives=lives
    )
    if not tr81.ok:
        env.close()
        return {
            "success": False,
            "stage": "8_1_body",
            "ctrl_wait_8_1": pred.get("ctrl_wait_8_1"),
            "probe_8_1": tr81.to_dict(),
            "approx_total_to_w8": _sum_w8_pred(pred),
        }

    wait82, snap82 = idle_until(env, is_8_2_control, max_wait=600)
    if not is_8_2_control(snap82):
        env.close()
        return {
            "success": False,
            "stage": "8_2_control",
            "leave_8_1": tr81.leave_frame,
            "ctrl_wait_8_1": pred.get("ctrl_wait_8_1"),
            "ctrl_wait_8_2": wait82,
        }

    tr82 = probe_8_2_from_control(
        env, fm2, start_8_2, max_play=max_8_2, start_lives=int(snap82.lives)
    )
    env.close()

    leave81 = tr81.leave_frame or 0
    leave82 = tr82.leave_frame or 0
    wait81 = pred.get("ctrl_wait_8_1") or 0
    base = _sum_w8_pred(pred)
    total = base + wait81 + leave81 + wait82 + leave82
    ok = tr82.ok
    return {
        "success": ok,
        **{
            k: pred.get(k)
            for k in (
                "leave_1_1",
                "ctrl_wait_1_2",
                "w4",
                "ctrl_wait_4_1",
                "leave_4_1",
                "ctrl_wait_4_2",
                "w8",
            )
        },
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
        "vs_natural_82_8_2": NAT82_TO_8_2_LEAVE,
        "delta_vs_natural_82_8_2": NAT82_TO_8_2_LEAVE - total if ok else None,
        "control_8_1_fp": snap_fingerprint(pred["control_snap"])
        if pred.get("control_snap")
        else None,
    }


def verify_continuous_tail_from_8_1(
    *,
    fm2_path: Path = DEFAULT_FM2,
    hl_1_1_path: Path = DEFAULT_HL_1_1,
    start_idx: int = HL_8_1_FM2_START,
    max_play: int | None = None,
) -> dict[str, Any]:
    """From 8-1 control, play FM2 continuously until ending/death (no re-gate)."""
    fm2 = parse_fm2(fm2_path).frames
    env = make_level1_env()
    pred = reach_8_1_control_after_hl_w8(
        env, fm2_path=fm2_path, hl_1_1_path=hl_1_1_path
    )
    if not pred.get("success") or pred.get("control_snap") is None:
        env.close()
        return {
            "success": False,
            "stage": "8_1_control",
            "pred": {k: v for k, v in pred.items() if "snap" not in k},
        }

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
        env.step(to_action9(body[i]))
        ram = env.get_ram()
        snap = read_snapshot(ram, i + 1)
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        key = (int(snap.world), int(snap.level))
        if key != last:
            exits.append(
                {
                    "i": i + 1,
                    "from": list(last),
                    "to": list(key),
                    "x": px,
                    "t": int(snap.timer),
                }
            )
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
        "vs_natural_82": NAT82_TO_ENDING,
        "delta_vs_natural_82": (NAT82_TO_ENDING - (base + ending)) if ending else None,
    }


__all__ = [
    "reach_surface_after_hl_1_1",
    "reach_w4_after_hl",
    "reach_4_1_control_after_hl_w4",
    "reach_w8_after_hl",
    "reach_8_1_control_after_hl_w8",
    "reach_stage_control",
    "verify_1_2_natural_chain",
    "verify_4_1_4_2_natural_chain",
    "verify_8_1_8_2_natural_chain",
    "verify_continuous_tail_from_8_1",
]
