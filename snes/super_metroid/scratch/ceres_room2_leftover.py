"""Compare live Ceres 2 vs Sniq gs=8 series. Scratch only."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.paths import GAME_DIR
from super_metroid.ram import parse_env_state
from super_metroid.routes.kpdr.ceres.outbound import (
    play_ceres_falling_to_magnet,
    play_ceres_first_room_moonfall,
)

PIN = GAME_DIR / "scratch" / "ceres_first_control.state"
SERIES = GAME_DIR / "recordings" / "tas_oracle" / "sniq_100_lsnes" / "series.jsonl"
OUT = GAME_DIR / "scratch" / "ceres_room2_leftover.json"
TAS0 = 8950
TAS_DOOR = 9071


def main() -> None:
    tas = []
    with SERIES.open() as fh:
        for line in fh:
            rec = json.loads(line)
            f = int(rec["frame"])
            if TAS0 <= f <= TAS_DOOR + 2:
                tas.append(rec)
            if f > TAS_DOOR + 2:
                break
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)

        class Sess:
            def __init__(self) -> None:
                self.env = env
                self.frame = 0
                self.state = parse_env_state(env, mode="nav")
                self.trace = []

            def step(self, action, reason: str = ""):
                env.step(action)
                self.frame += 1
                self.state = parse_env_state(env, frame=self.frame, mode="nav")
                assist.apply(env.data, self.state)
                self.trace.append(
                    {
                        "f": self.frame,
                        "reason": reason,
                        "x": int(self.state.samus_x),
                        "y": int(self.state.samus_y),
                        "pose": int(self.state.pose),
                        "gs": int(self.state.game_state),
                        "room": f"0x{int(self.state.room_id):04X}",
                        "mt": int(self.state.movement_type),
                    }
                )
                return self.state

            def wait_until(self, predicate, *, timeout: int, reason: str) -> int:
                for waited in range(timeout + 1):
                    if predicate(self.state):
                        return waited
                    self.step(idle_action(), reason)
                raise TimeoutError(reason)

        sess = Sess()
        play_ceres_first_room_moonfall(sess)
        start = sess.frame
        sess.trace.clear()
        play_ceres_falling_to_magnet(sess)
        live = sess.trace
        diffs = []
        for rec in tas:
            rel = int(rec["frame"]) - TAS0
            hit = next((r for r in live if r["f"] - start == rel + 1), None)
            if hit is None:
                continue
            dx = hit["x"] - int(rec["x"])
            dy = hit["y"] - int(rec["y"])
            if abs(dx) >= 8 or abs(dy) >= 8 or hit["pose"] != int(rec["pose"]):
                diffs.append(
                    {
                        "rel": rel,
                        "tas": {
                            "x": rec["x"],
                            "y": rec["y"],
                            "pose": rec["pose"],
                            "gs": rec["gs"],
                        },
                        "live": hit,
                        "dx": dx,
                        "dy": dy,
                    }
                )
        report = {
            "room1_frames": start,
            "room2_frames": sess.frame - start,
            "first_div": diffs[:12],
            "div_count": len(diffs),
            "live_sel": [
                r
                for r in live
                if r["reason"]
                in (
                    "ceres_falling_floor_hop",
                    "ceres_falling_magnet_feet",
                    "ceres_falling_exit_hop",
                    "ceres_falling_door_kb",
                    "ceres_falling_exit",
                    "ceres_falling_door",
                )
                or r["gs"] in (9, 11)
            ][:40],
        }
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"room2 {report['room2_frames']}f divs={len(diffs)}")
        if diffs:
            d = diffs[0]
            print(f"first_div rel={d['rel']} tas={d['tas']} live={d['live']}")
        for r in report["live_sel"][:15]:
            print(f"  {r['f']-start:3d} {r['reason']:28s} x={r['x']} y={r['y']} p={r['pose']} gs={r['gs']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
