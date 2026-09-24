"""The gather chain's development runner: stages back to back in one env.

The coast bomb farm does not use this. A ``$0670`` refill hides the
10-kill 5-rupee, so ``--through pre-l1`` stays assist-off. The chain runs
from the power-on ``PreL1BombLeave`` pin under a health refill and writes
no inventory; each green stage saves ``GatherChain_<stage>`` for
``stage_replay.py``. A saved state is development evidence. The route is
still the natural predecessor.
"""

from __future__ import annotations

import json
from typing import Any

from retro_harness.env import save_state
from retro_harness.segment_runner import save_rgb_png
from zelda_i.assist import LastHeartAssist, UnlimitedHealthAssist
from zelda_i.dungeon.ops import B_ITEM_CANDLE
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import open_env

# Blue candle. Red is 2 and is not this prefix.
BLUE_CANDLE = 1


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

def run_chain(
    stages: list[tuple[str, Any]],
    *,
    from_state: str,
    chain: str,
    engage_hearts: int = 1,
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
    obs = None
    try:
        assist.apply_env(env, frame=0)
        for name, controller in stages:
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
    "gather_glance",
    "PRE_L1_LEAVE",
    "pin_pre_l1",
    "run_chain",
]
