"""Compose 0x50 east leftover → 0x01 well → 0x72.

Natural-entry evidence for ``room_01_down_to_0x72``. Not continuous.
0x50-east leftover lands 0x01 ~(560,120), west of the well; corridor walk
is east on y=120 to x≈760, then (760,99), hold UP. Independent settle
past submodule 14 (do not trust 240f settle_control).

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_compose_stair.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from alttp.opening_route.room_engine import run_room_edge
from alttp.paths import RECORDINGS_DIR
from alttp.primitives import Waypoint, move_to
from alttp.ram import EQUIP_SWORD, FOLLOWER, NUM_KEYS, wram_index
from alttp.room_sense import detect_edge, load_room_map
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames

OUT_DIR = RECORDINGS_DIR / "probe_compose_stair"
STAIR_SUBMODULE = 14
SETTLE_MAX = 400
PUSH_MAX = 240


def glance(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    ram = env.get_ram()  # type: ignore[attr-defined]
    boot = leftover_from_snapshot(snap)
    boot.update(
        {
            "room_hex": f"0x{int(boot['room']) & 0xFF:02X}",
            "module_hex": f"0x{int(boot['module']) & 0xFF:02X}",
            "F359_sword": int(ram[wram_index(EQUIP_SWORD)]),
            "F3CC_follower": int(ram[wram_index(FOLLOWER)]),
            "F36F_keys": int(ram[wram_index(NUM_KEYS)]),
            "has_control": bool(snap.has_control),
        }
    )
    return boot


def settle_past_stair(env: object, *, max_frames: int = SETTLE_MAX) -> dict[str, Any]:
    """Idle until submodule 14 ends and control returns. Not settle_control."""
    rec: dict[str, Any] = {
        "ok": False,
        "frames": 0,
        "saw_sub14": False,
        "sub14_last_frame": None,
        "control_frame": None,
    }
    frames = 0
    while frames < max_frames:
        snap = snapshot_env(env)
        if snap.submodule == STAIR_SUBMODULE:
            rec["saw_sub14"] = True
            rec["sub14_last_frame"] = frames
        if (
            snap.game_mode == 0x07
            and snap.submodule != STAIR_SUBMODULE
            and snap.has_control
            and not snap.is_text_mode
            and not snap.is_hold_up_item
        ):
            rec["ok"] = True
            rec["frames"] = frames
            rec["control_frame"] = frames
            rec["leftover"] = glance(env)
            return rec
        if snap.game_mode == 0x12:
            rec["frames"] = frames
            rec["blocker"] = "Link died"
            rec["leftover"] = glance(env)
            return rec
        step_frames(env, no_action(), 4)
        frames += 4
    rec["frames"] = frames
    rec["blocker"] = "stair settle timeout"
    rec["leftover"] = glance(env)
    return rec


def hold_up_until_leave(env: object, *, room: int, max_frames: int = PUSH_MAX) -> dict[str, Any]:
    start = snapshot_env(env)
    rec: dict[str, Any] = {
        "dir": "UP",
        "fromXy": [start.link_x, start.link_y],
        "frames": 0,
    }
    before = start
    frames = 0
    while frames < max_frames:
        step_frames(env, action_for("UP"), 4)
        frames += 4
        after = snapshot_env(env)
        edge = detect_edge(
            before,
            after,
            expected_room=room,
            frames=frames,
            label="UP",
            preferred_direction="UP",
        )
        if edge is not None:
            rec.update(
                {
                    "frames": frames,
                    "toRoom": f"0x{edge.to_room:02X}",
                    "toXy": list(edge.to_xy),
                    "approachXy": list(edge.from_xy),
                    "outdoors": edge.outdoors,
                    "submodule": int(after.submodule),
                }
            )
            rec["glance_at_edge"] = glance(env)
            return rec
        before = after
    rec["frames"] = frames
    rec["timeout"] = True
    rec["endXy"] = [snapshot_env(env).link_x, snapshot_env(env).link_y]
    rec["leftover"] = glance(env)
    return rec


def compose() -> dict[str, Any]:
    room_map = load_room_map("room_01")
    door = room_map.door("down_to_0x72")
    if door is None:
        return {"ok": False, "blocker": "maps/room_01.json missing down_to_0x72"}
    corridor = next(p for p in room_map.points if p.label == "stair_west_corridor")
    rec: dict[str, Any] = {
        "when": datetime.now(timezone.utc).isoformat(),
        "state": "CastleRoom50",
        "ok": False,
        "walks": [],
        "mapApproach": list(door.approach_xy),
        "mapLanding": list(door.landing_xy) if door.landing_xy else None,
    }
    env = build_boot_env("CastleRoom50")
    try:
        env.reset()  # type: ignore[attr-defined]
        rec["spawnLeftover"] = glance(env)
        edge = run_room_edge(
            env, "room_50", "east_to_0x01", clear=True, source="state_load_dev"
        )
        rec["eastTo01"] = {
            "ok": edge.ok,
            "phase": edge.phase,
            "frames": edge.frames,
            "blocker": edge.blocker,
            "leftover": glance(env),
        }
        snap = snapshot_env(env)
        if snap.room_base_id != 1:
            rec["blocker"] = "0x50 east did not land in 0x01"
            rec["finalLeftover"] = glance(env)
            return rec
        rec["eastLandXy"] = [snap.link_x, snap.link_y]
        # 0x50 leftover ~(560,120) is west of the well; walk east on y=120.
        for label, x, y, tol in (
            ("west_corridor", corridor.x, corridor.y, 4),
            ("stair_approach", door.approach_xy[0], door.approach_xy[1], 2),
        ):
            walk = move_to(
                env,
                Waypoint(x, y, tolerance=tol, room=1, label=label),
                max_frames=700,
            )
            rec["walks"].append(
                {
                    "label": label,
                    "ok": walk.ok,
                    "reason": walk.reason,
                    "frames": walk.frames,
                    "leftover": glance(env),
                }
            )
            if snapshot_env(env).room_base_id != 1:
                rec["blocker"] = f"left 0x01 during {label}"
                rec["finalLeftover"] = glance(env)
                return rec
            if not walk.ok:
                rec["blocker"] = walk.reason
                rec["finalLeftover"] = glance(env)
                return rec
        rec["approachLeftover"] = glance(env)
        up = hold_up_until_leave(env, room=1)
        rec["up"] = up
        if up.get("toRoom") != "0x72":
            rec["blocker"] = f"UP from approach did not reach 0x72: {up}"
            rec["finalLeftover"] = glance(env)
            return rec
        rec["edgeLeftover"] = up.get("glance_at_edge")
        rec["settle"] = settle_past_stair(env)
        rec["destLeftover"] = rec["settle"].get("leftover") or glance(env)
        dest = rec["destLeftover"]
        landing = door.landing_xy
        rec["ok"] = bool(
            rec["settle"].get("ok")
            and dest.get("room") == 0x72
            and dest.get("submodule") == 0
            and dest.get("has_control")
            and dest.get("F359_sword") == 1
            and dest.get("F3CC_follower") == 0
            and dest.get("F36F_keys") == 0
        )
        if landing is not None and dest.get("xy"):
            rec["landingDelta"] = [
                int(dest["xy"][0]) - int(landing[0]),
                int(dest["xy"][1]) - int(landing[1]),
            ]
        if not rec["ok"] and "blocker" not in rec:
            rec["blocker"] = rec["settle"].get("blocker") or "dest glance miss"
        return rec
    finally:
        env.close()  # type: ignore[attr-defined]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = compose()
    path = OUT_DIR / "compose.json"
    path.write_text(json.dumps(rec, indent=2))
    dest = rec.get("destLeftover") or rec.get("finalLeftover") or rec.get("spawnLeftover")
    print(f"Wrote {path}", flush=True)
    print(
        f"ok={rec.get('ok')} leftover={dest} blocker={rec.get('blocker')!r}",
        flush=True,
    )
    return 0 if rec.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
