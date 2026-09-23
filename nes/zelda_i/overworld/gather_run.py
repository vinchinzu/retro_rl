"""Dev pins and a last-heart runner for gathering geometry.

The coast bomb farm does not use this. A ``$0670`` refill hides the
10-kill 5-rupee, so ``--through pre-l1`` stays assist-off. Later stops
load a ``BFS_*`` pin, stamp only the count they are allowed to stamp,
and walk under :class:`zelda_i.assist.LastHeartAssist`.

Entry pokes, and only on the pin before the walk: bomb count, rupee
count, blue-candle byte, B-slot select. Never the item the segment is
there to earn, never a heart container, never a sword. A saved state is
development evidence. The route is still the natural predecessor.
"""

from __future__ import annotations

import json
from typing import Any

from retro_harness.env import save_state, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import save_rgb_png
from zelda_i.assist import LastHeartAssist, UnlimitedHealthAssist
from zelda_i.dungeon.ops import (
    B_ITEM_CANDLE,
    ensure_bomb,
    mem_write,
    poke_bombs,
    poke_rupees,
)
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_SELECTED_ITEM,
    full_health_byte,
    health_byte_is_coherent,
    read_snapshot,
)
from zelda_i.runner import open_env

# Blue candle. Red is 2 and is not this prefix.
BLUE_CANDLE = 1
_SETTLE_FRAMES = 8


def entry_health_byte(health: int) -> int | None:
    """Coherent full-heart byte for an entry pin, or None if already legal.

    The ``BFS_*`` overworld pins were saved at ``$066F = 0x2F``: low nibble
    15 in 3 containers. ``whole_hearts`` then reads 16, last-heart assist
    never engages, and every later heart count is against a fake budget.
    Clamping rewrites that byte to the owned container max. It does not
    add a container.
    """
    if health_byte_is_coherent(health):
        return None
    return full_health_byte(health)


def gather_glance(snap: Any) -> dict[str, Any]:
    """Leave fields a gathering hop is graded on, plus the compact census."""
    glance = compact_snapshot(snap)
    glance.update(
        {
            "screen_hex": f"0x{int(snap.screen):02X}",
            "sword": int(snap.sword),
            "candle": int(snap.candle),
            "ring": int(snap.ring),
            "whole_hearts": int(snap.whole_hearts),
            "heart_containers": int(snap.heart_containers),
        }
    )
    return glance


def write_entry_pin(
    from_state: str,
    save_as: str,
    *,
    segment: str,
    bombs: int | None = None,
    rupees: int | None = None,
    candle: int | None = None,
    select: str | None = None,
) -> dict[str, Any]:
    """Load ``from_state``, apply the disclosed count pokes, save ``save_as``.

    ``select`` is ``"bomb"`` or ``"candle"``. Counts only rise. A candle
    write is the blue candle byte, for a segment whose predecessor has
    not bought it yet.
    """
    if select not in (None, "bomb", "candle"):
        raise ValueError(f"select must be bomb, candle, or omitted, not {select!r}")
    if candle not in (None, BLUE_CANDLE):
        raise ValueError("candle entry poke is blue (1) or omitted")

    env = open_env(from_state=from_state)
    pokes: list[dict[str, Any]] = []
    try:
        before = read_snapshot(env.get_ram())
        coherent = entry_health_byte(int(before.health))
        if coherent is not None:
            data = env.unwrapped.data
            data.set_value("health", coherent)
            data.set_value("heart_partial", 0xFF)
            pokes.append(
                {
                    "field": "health",
                    "from": int(before.health),
                    "to": coherent,
                    "msg": "clamped incoherent heart byte to owned containers",
                }
            )
        if bombs is not None and int(before.bombs) < int(bombs):
            pokes.append(
                {
                    "field": "bombs",
                    "from": int(before.bombs),
                    "to": int(bombs),
                    "msg": poke_bombs(env, int(bombs)),
                }
            )
        if rupees is not None and int(before.rupees) < int(rupees):
            pokes.append(
                {
                    "field": "rupees",
                    "from": int(before.rupees),
                    "to": int(rupees),
                    "msg": poke_rupees(env, int(rupees)),
                }
            )
        if candle is not None and int(before.candle) < int(candle):
            pokes.append(
                {
                    "field": "candle",
                    "from": int(before.candle),
                    "to": int(candle),
                    "msg": mem_write(env, ADDR_CANDLE, int(candle)),
                }
            )
        if select == "bomb":
            pokes.append({"field": "selected_item", "msg": ensure_bomb(env)})
        elif select == "candle":
            pokes.append(
                {
                    "field": "selected_item",
                    "to": B_ITEM_CANDLE,
                    "msg": mem_write(env, ADDR_SELECTED_ITEM, B_ITEM_CANDLE),
                }
            )
        obs = None
        for _ in range(_SETTLE_FRAMES):
            obs, *_rest = env.step(nes_idle_action())
        after = read_snapshot(env.get_ram())
        path = save_state(env, GAME_DIR, GAME, save_as)
        glance = gather_glance(after)
        write_state_provenance(
            path,
            source_state_path=state_path(GAME_DIR, GAME, from_state),
            request={
                "segment": segment,
                "start_state": from_state,
                "save_as": save_as,
                "track": "last_heart",
                "pokes": pokes,
            },
            selected_trial={"ok": True, "glance": glance, "pokes": pokes},
            natural_entry=False,
        )
        if obs is not None:
            save_rgb_png(obs, RECORDINGS_DIR / f"{segment}_entry.png")
        payload = {
            "ok": True,
            "state": save_as,
            "path": str(path),
            "pokes": pokes,
            "glance": glance,
        }
    finally:
        env.close()
    _print_glance("entry", payload["glance"], pokes=pokes)
    return payload


def written_leave(
    save_as: str | None,
    ok: bool,
    saved: str | None,
    *,
    save_red: bool = True,
) -> str | None:
    """Return the pose path when this stop may keep a file.

    Other gathering stops still write a red pose. A heart leave is kept
    only when the container count rose (``ok``).
    """
    if not save_as or not (bool(ok) or save_red):
        return None
    return saved


def leave_payload(
    controller: Any,
    glance: dict[str, Any],
    frames: int,
    save_as: str | None,
    saved: str | None,
) -> dict[str, Any]:
    """Grade a leave: ``ok`` is the stop predicate; ``saved`` is independent."""
    return {
        "ok": bool(getattr(controller, "success", False)),
        "frames": int(frames),
        "saved": None if save_as is None else saved,
        "glance": glance,
    }


def run_segment(
    controller: Any,
    *,
    from_state: str,
    segment: str,
    save_as: str | None = None,
    max_frames: int | None = None,
    save_red: bool = True,
    engage_hearts: int = 1,
) -> dict[str, Any]:
    """Walk ``controller`` from ``from_state`` under last-heart assist.

    ``engage_hearts`` above 1 refills earlier, for a screen whose single
    hit is bigger than one heart (the 0x0A Lynel took 0x21/0x7E to 0).

    Assist is applied once before the first step, so a pin that loaded
    already on its last heart is refilled before it can eat a killing
    blow, then after every step. When ``save_as`` is set, the last frame
    is written even on a red stop unless ``save_red`` is false. ``ok``
    still follows the controller either way.
    """
    if engage_hearts <= 1:
        assist = LastHeartAssist(enabled=True)
    else:
        assist = UnlimitedHealthAssist(enabled=True, engage_at_whole_hearts=engage_hearts)
    env = open_env(from_state=from_state)
    limit = max_frames
    if limit is None:
        limit = int(getattr(controller, "max_frames", 8000) or 8000)
    obs = None
    frame = -1
    try:
        assist.apply_env(env, frame=0)
        for frame in range(limit):
            snap = read_snapshot(env.get_ram())
            action = controller.step(snap)
            obs, *_rest = env.step(action.action)
            assist.apply_env(env, frame=frame)
            if _stopped(controller):
                break
        snap = read_snapshot(env.get_ram())
        glance = gather_glance(snap)
        ok = bool(getattr(controller, "success", False))
        saved = None
        if save_as and (save_red or ok):
            path = save_state(env, GAME_DIR, GAME, save_as)
            saved = written_leave(save_as, ok, str(path), save_red=save_red)
        payload = leave_payload(
            controller,
            glance,
            frame + 1,
            save_as if saved else None,
            saved,
        )
        payload["segment"] = segment
        payload["from_state"] = from_state
        payload["assist"] = assist.report()
        if hasattr(controller, "report"):
            payload["controller"] = controller.report()
        if saved is not None:
            write_state_provenance(
                path,
                source_state_path=state_path(GAME_DIR, GAME, from_state),
                request={
                    "segment": segment,
                    "start_state": from_state,
                    "save_as": save_as,
                    "track": "last_heart",
                },
                selected_trial={
                    "ok": payload["ok"],
                    "frames": payload["frames"],
                    "glance": glance,
                    "controller": payload.get("controller", {}),
                    "assist": payload["assist"],
                },
                natural_entry=False,
            )
        if obs is not None:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            save_rgb_png(obs, RECORDINGS_DIR / f"{segment}_final.png")
    finally:
        env.close()
    _print_glance("leave", payload["glance"], ok=payload["ok"], frames=payload["frames"])
    out = RECORDINGS_DIR / f"{segment}_report.json"
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    payload["report_path"] = str(out)
    return payload


def run_chain(
    stages: list[tuple[str, Any]],
    *,
    from_state: str,
    chain: str,
    engage_hearts: int = 1,
    rupee_topups: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run ``stages`` back to back in one env. Stop at the first red stage.

    Each green stage writes ``<chain>_<name>`` so a later sitting can start
    from the last good boundary. ``ok`` is every stage green.
    """
    if engage_hearts <= 1:
        assist = LastHeartAssist(enabled=True)
    else:
        assist = UnlimitedHealthAssist(enabled=True, engage_at_whole_hearts=engage_hearts)
    env = open_env(from_state=from_state)
    results: list[dict[str, Any]] = []
    total = 0
    inventory_assist: list[dict[str, Any]] = []
    obs = None
    try:
        assist.apply_env(env, frame=0)
        for name, controller in stages:
            target = (rupee_topups or {}).get(name)
            if target is not None:
                before = read_snapshot(env.get_ram()).rupees
                if before < target:
                    note = poke_rupees(env, target)
                    inventory_assist.append(
                        {
                            "stage": name,
                            "field": "rupees",
                            "from": before,
                            "to": target,
                            "note": note,
                        }
                    )
            if hasattr(controller, "bind_env"):
                controller.bind_env(env)
            limit = int(getattr(controller, "max_frames", 8000) or 8000)
            frames = 0
            for frames in range(1, limit + 1):
                snap = read_snapshot(env.get_ram())
                action = controller.step(snap)
                obs, *_rest = env.step(action.action)
                total += 1
                assist.apply_env(env, frame=total)
                if _stopped(controller):
                    break
            snap = read_snapshot(env.get_ram())
            ok = bool(getattr(controller, "success", False))
            row: dict[str, Any] = {
                "stage": name,
                "ok": ok,
                "frames": frames,
                "end_frame": total,
                "glance": gather_glance(snap),
            }
            if hasattr(controller, "report"):
                row["controller"] = controller.report()
            if ok:
                path = save_state(env, GAME_DIR, GAME, f"{chain}_{name}")
                row["saved"] = str(path)
            results.append(row)
            _print_glance(name, row["glance"], ok=ok, frames=frames, total=total)
            if not ok:
                break
        if obs is not None:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            save_rgb_png(obs, RECORDINGS_DIR / f"{chain}_final.png")
    finally:
        env.close()
    payload = {
        "chain": chain,
        "from_state": from_state,
        "ok": len(results) == len(stages) and all(r["ok"] for r in results),
        "frames": total,
        "stages": results,
        "assist": assist.report(),
        "inventory_assist": inventory_assist,
    }
    out = RECORDINGS_DIR / f"{chain}_report.json"
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    payload["report_path"] = str(out)
    return payload


PRE_L1_LEAVE = "PreL1BombLeave"


def pin_pre_l1(save_as: str = PRE_L1_LEAVE) -> dict[str, Any]:
    """Power-on ``--through pre-l1`` (assist off), then pin the bomb-shop leave.

    Saved only when Link is in the ``0x6F`` cave holding bombs. The spine's
    own disclosed write (``$066D`` up to the 20R price when short) is the
    one RAM write on this pose.
    """
    from retro_harness.env import make_env, reset_obs
    from zelda_i.spine.survival import run_survival_spine

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        run = run_survival_spine(env, obs, assist=None, through="pre-l1", allow_pokes=False)
        snap = read_snapshot(env.get_ram())
        glance = gather_glance(snap)
        ok = snap.level == 0 and snap.screen == 0x6F and snap.mode == 11 and snap.bombs >= 1
        saved = None
        if ok:
            path = save_state(env, GAME_DIR, GAME, save_as)
            write_state_provenance(
                path,
                source_state_path=None,
                request={
                    "segment": "pre_l1",
                    "start_state": "power_on",
                    "save_as": save_as,
                    "track": "pre_l1_default",
                    "disclosed_writes": ["$066D up to 20 before bomb_topup when short"],
                },
                selected_trial={"ok": True, "glance": glance, "spine": run.report()},
                natural_entry=False,
            )
            saved = str(path)
    finally:
        env.close()
    _print_glance("pre_l1", glance, ok=ok)
    return {"ok": ok, "saved": saved, "glance": glance}


def _stopped(controller: Any) -> bool:
    if getattr(controller, "success", False) or getattr(controller, "failed", False):
        return True
    phase = getattr(controller, "phase", None)
    if phase is None:
        return False
    name = getattr(phase, "name", None)
    token = str(name if name is not None else phase)
    return token.upper() in {"FAILED", "DONE"}


def _print_glance(kind: str, glance: dict[str, Any], **extra: Any) -> None:
    line = {
        "kind": kind,
        "screen": glance.get("screen_hex"),
        "mode": glance.get("mode"),
        "x": glance.get("x"),
        "y": glance.get("y"),
        "sword": glance.get("sword"),
        "bombs": glance.get("bombs"),
        "rupees": glance.get("rupees"),
        "candle": glance.get("candle"),
        "hearts": f"{glance.get('whole_hearts')}/{glance.get('heart_containers')}",
        "health": glance.get("health"),
        **extra,
    }
    print(json.dumps(line, default=str), flush=True)


__all__ = [
    "BLUE_CANDLE",
    "B_ITEM_CANDLE",
    "entry_health_byte",
    "gather_glance",
    "leave_payload",
    "PRE_L1_LEAVE",
    "pin_pre_l1",
    "run_chain",
    "run_segment",
    "write_entry_pin",
    "written_leave",
]
