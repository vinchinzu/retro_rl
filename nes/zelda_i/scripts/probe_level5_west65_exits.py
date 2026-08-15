"""Probe 0x66 exits after the east-key return (no key/door pokes).

Walks Level5EastKey -> 0x76 -> 0x66 with the west65 policy, then tries
natural UP and bomb walls (inventory bombs only).
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import bomb_stand, exit_door, room_fields
from zelda_i.level5_path import make_west65_controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot


def _walk_to_66(env, assist) -> dict:
    controller = make_west65_controller()
    controller.max_frames = 4000
    last = -1
    trail = []
    for frame in range(controller.max_frames):
        if assist is not None:
            assist.apply_env(env, frame=frame)
        snap = read_snapshot(env.get_ram())
        action = controller.step(snap)
        env.step(action.action)
        after = read_snapshot(env.get_ram())
        if after.screen != last:
            trail.append(
                {
                    "frame": frame + 1,
                    "room": f"0x{after.screen:02x}",
                    "mode": after.mode,
                    "x": after.link_x,
                    "y": after.link_y,
                    "keys": after.keys,
                    "reason": action.reason,
                }
            )
            last = after.screen
        if (
            after.level == 5
            and after.screen == 0x66
            and after.mode == PLAY_MODE
            and not after.transitioning
        ):
            return {
                "ok": True,
                "frames": frame + 1,
                "trail": trail,
                "at": room_fields(after, env.get_ram()),
                "controller": controller.report(),
            }
        if controller.success or controller.failed:
            break
    snap = read_snapshot(env.get_ram())
    return {
        "ok": False,
        "frames": controller.frames,
        "trail": trail,
        "at": room_fields(snap, env.get_ram()),
        "controller": controller.report(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default="Level5EastKey")
    parser.add_argument("--infinite-life", action="store_true")
    args = parser.parse_args(argv)
    configure_headless()
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if args.infinite_life else None
    total = [0]
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        walked = _walk_to_66(env, assist)
        probes = {}
        if walked["ok"]:
            probes["UP"] = exit_door(env, assist, total, "UP")
            # return to 0x66 if we left
            snap = read_snapshot(env.get_ram())
            if snap.screen != 0x66:
                probes["UP_left_room"] = room_fields(snap, env.get_ram())
            else:
                probes["LEFT"] = exit_door(env, assist, total, "LEFT")
                snap = read_snapshot(env.get_ram())
                if snap.screen == 0x66:
                    probes["BOMB_UP"] = bomb_stand(
                        env, assist, total, "UP", 120, 101
                    )
                    snap = read_snapshot(env.get_ram())
                    if snap.screen == 0x66:
                        probes["BOMB_LEFT"] = bomb_stand(
                            env, assist, total, "LEFT", 48, 141
                        )
        snap = read_snapshot(env.get_ram())
        obs, *_ = env.step(nes_idle_action())
        shot = RECORDINGS_DIR / "l5_west65_exits.png"
        save_rgb_png(obs, shot)
        report = {
            "segment": "level5_west65_exit_probe",
            "start_state": args.from_state,
            "runtime_class": "bronze",
            "track": "assisted" if args.infinite_life else "clean",
            "key_poke": False,
            "door_poke": False,
            "walk": walked,
            "probes": {
                name: {
                    "direction": row.get("direction") or row.get("face"),
                    "result": row.get("result"),
                    "changed_room": row.get("changed_room"),
                    "before": {
                        "sc": row.get("before", {}).get("sc"),
                        "x": row.get("before", {}).get("x"),
                        "y": row.get("before", {}).get("y"),
                        "keys": row.get("before", {}).get("keys"),
                        "bombs": row.get("before", {}).get("bombs"),
                        "doors": row.get("before", {}).get("cur_opened_doors"),
                    },
                    "after": {
                        "sc": row.get("after", {}).get("sc"),
                        "x": row.get("after", {}).get("x"),
                        "y": row.get("after", {}).get("y"),
                        "keys": row.get("after", {}).get("keys"),
                        "bombs": row.get("after", {}).get("bombs"),
                        "doors": row.get("after", {}).get("cur_opened_doors"),
                        "mode": row.get("after", {}).get("mode"),
                        "type_counts": row.get("after", {}).get("type_counts"),
                    },
                }
                for name, row in probes.items()
                if isinstance(row, dict) and "result" in row
            },
            "extra": {k: v for k, v in probes.items() if "result" not in v},
            "final": room_fields(snap, env.get_ram()),
            "screenshot": str(shot),
            "assist": assist.report() if assist else None,
        }
        out = RECORDINGS_DIR / "l5_west65_exits.json"
        write_json_report(out, report)
        print(f"walk_ok={walked['ok']} room={walked['at'].get('sc')} probes={list(probes)}")
        for name, row in report["probes"].items():
            print(
                f"  {name}: {row['result']} "
                f"{row['before'].get('sc')}->{row['after'].get('sc')} "
                f"keys {row['before'].get('keys')}->{row['after'].get('keys')}"
            )
        print(f"wrote {out}")
        return 0 if walked["ok"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
