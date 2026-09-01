"""Dump TAS Ceres 1 vs live hop. Scratch, not STATUS."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from retro_harness.controls import SNES_BUTTON_NAMES
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.ram import parse_env_state, set_moonwalk
from super_metroid.routes.kpdr.ceres.outbound import play_ceres_first_room_moonfall
from super_metroid.tas.lsmv import parse_lsmv

GAME_DIR = Path(__file__).resolve().parents[1]
PIN = GAME_DIR / "scratch" / "ceres_first_control.state"
SERIES = GAME_DIR / "recordings" / "tas_oracle" / "sniq_100_lsnes" / "series.jsonl"
LSMV = GAME_DIR / "tas" / "ref" / "sniq_100_4010M.lsmv"
OUT = GAME_DIR / "scratch" / "ceres_first_moonfall_dump.json"
TAS_CTRL = 8538
TAS_FALLING = 8832


def _row(st, extra: dict | None = None) -> dict:
    d = {
        "x": int(st.samus_x),
        "y": int(st.samus_y),
        "pose": int(st.pose),
        "gs": int(st.game_state),
        "room": f"0x{int(st.room_id):04X}",
        "mw": int(st.moonwalk),
        "vy": int(st.velocity_y),
        "vd": int(st.vertical_direction),
        "mt": int(st.movement_type),
        "facing": int(st.facing),
        "air": int(st.movement_type) in (2, 3, 6, 23),
    }
    if extra:
        d.update(extra)
    return d


def _tas_series() -> list[dict]:
    rows = []
    with SERIES.open() as fh:
        for line in fh:
            rec = json.loads(line)
            f = int(rec["frame"])
            if TAS_CTRL <= f <= TAS_FALLING + 5:
                rows.append(
                    {
                        "f": f,
                        "rel": f - TAS_CTRL,
                        "x": rec["x"],
                        "y": rec["y"],
                        "pose": rec["pose"],
                        "gs": rec["gs"],
                        "room": rec["room_id"],
                    }
                )
            if f > TAS_FALLING + 5:
                break
    return rows


def _tas_buttons() -> list[dict]:
    movie = parse_lsmv(LSMV)
    out = []
    for i in range(TAS_CTRL, min(TAS_CTRL + 180, movie.num_frames)):
        names = [
            SNES_BUTTON_NAMES[k]
            for k, v in enumerate(movie.frames[i])
            if v
        ]
        out.append({"rel": i - TAS_CTRL, "names": names, "raw": movie.raw_p1[i]})
    return out


def _live_current() -> dict:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        set_moonwalk(env, True)

        class Sess:
            def __init__(self) -> None:
                self.env = env
                self.frame = 0
                self.state = parse_env_state(env, mode="nav")
                self.trace: list[dict] = []

            def step(self, action, reason: str = ""):
                env.step(action)
                self.frame += 1
                self.state = parse_env_state(env, frame=self.frame, mode="nav")
                assist.apply(env.data, self.state)
                st = self.state
                self.trace.append(
                    _row(st, {"f": self.frame, "reason": reason})
                )
                return self.state

            def wait_until(self, predicate, *, timeout: int, reason: str) -> int:
                for waited in range(timeout + 1):
                    if predicate(self.state):
                        return waited
                    self.step(idle_action(), reason)
                raise TimeoutError(reason)

        sess = Sess()
        play_ceres_first_room_moonfall(sess, restore_moonwalk=False)
        first_ground = next(
            (
                r
                for r in sess.trace
                if r["y"] >= 80 and not r["air"] and r["vd"] == 0
            ),
            None,
        )
        vd0 = [r for r in sess.trace if r["vd"] == 0 and r["air"]]
        ledges = [
            r
            for r in sess.trace
            if abs(r["y"] - 75) <= 4
            or abs(r["y"] - 267) <= 8
            or abs(r["y"] - 475) <= 8
            or abs(r["y"] - 571) <= 8
        ]
        return {
            "frames": sess.frame,
            "end": _row(sess.state),
            "first_ground_after_80": first_ground,
            "vd0_count": len(vd0),
            "vd0_first": vd0[:8],
            "ledge_hits": ledges[:40],
            "trace_head": sess.trace[:80],
            "y_at": {
                str(f): next((r for r in sess.trace if r["f"] == f), None)
                for f in (18, 28, 40, 80, 120, 200, 267, 329, 342)
            },
        }
    finally:
        env.close()


def _try_script(name: str, spans: list[tuple[tuple[str, ...], int]], n: int = 220) -> dict:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        set_moonwalk(env, True)
        frame = 0
        trace: list[dict] = []
        for names, hold in spans:
            act = buttons(*names) if names else idle_action()
            for _ in range(hold):
                env.step(act)
                frame += 1
                st = parse_env_state(env, frame=frame, mode="nav")
                assist.apply(env.data, st)
                if frame <= n:
                    trace.append(_row(st, {"f": frame, "in": list(names)}))
                if int(st.room_id) != 0xDF45 and int(st.game_state) == 8:
                    break
        st = parse_env_state(env, frame=frame, mode="nav")
        vd0 = [r for r in trace if r["vd"] == 0 and r["air"]]
        grounded = [
            r
            for r in trace
            if not r["air"] and r["y"] >= 70 and r["gs"] == 8
        ]
        return {
            "name": name,
            "frames": frame,
            "end": _row(st),
            "vd0_count": len(vd0),
            "vd0_first": vd0[:6],
            "first_ground": grounded[0] if grounded else None,
            "min_y_air_after_20": min(
                (r["y"] for r in trace if r["f"] > 20 and r["air"]),
                default=None,
            ),
            "max_vy": max((r["vy"] for r in trace), default=None),
            "trace_sel": [
                r
                for r in trace
                if r["f"] <= 40 or r["f"] % 10 == 0 or (r["vd"] == 0 and r["air"])
            ][:60],
        }
    finally:
        env.close()


def main() -> None:
    tas_xy = _tas_series()
    tas_btn = _tas_buttons()
    live = _live_current()
    # TAS-like from settled pad: short hop then air-turn moonfall recipe.
    tas_like = _try_script(
        "tas_like_from_pad",
        [
            (("RIGHT", "B"), 1),
            (("RIGHT", "A", "B"), 1),
            (("RIGHT", "B"), 12),
            (("LEFT", "B"), 1),
            (("L", "B"), 1),
            (("B",), 1),
            (("RIGHT", "X", "B"), 1),
            (("RIGHT", "A", "B"), 1),
            (("B",), 1),
            (("RIGHT", "B"), 8),
            (("RIGHT",), 8),
            ((), 4),
            (("LEFT",), 1),
            ((), 7),
            (("LEFT",), 15),
            ((), 1),
            (("RIGHT",), 1),
            ((), 80),
        ],
    )
    # Longer hop past first platform (x~164) then same recipe.
    late_turn = _try_script(
        "late_turn_x164",
        [
            (("RIGHT", "A", "B"), 2),
            (("RIGHT", "B"), 18),
            (("LEFT", "B"), 1),
            (("L", "B"), 1),
            (("B",), 1),
            (("RIGHT", "X", "B"), 1),
            (("RIGHT", "A", "B"), 1),
            (("B",), 1),
            ((), 80),
        ],
    )
    # Ground moonfall: hop, land, then wiki initiate.
    ground = _try_script(
        "ground_moonfall",
        [
            (("RIGHT", "A", "B"), 2),
            (("RIGHT", "B"), 8),
            (("LEFT",), 8),
            ((), 6),
            (("RIGHT", "X", "L"), 8),
            (("RIGHT", "X", "L", "A"), 2),
            (("RIGHT", "A"), 2),
            (("RIGHT",), 80),
        ],
    )
    report = {
        "tas_ctrl": TAS_CTRL,
        "tas_xy_head": tas_xy[:40],
        "tas_xy_turn": [
            r
            for r in tas_xy
            if r["rel"] <= 40 or r["pose"] in (25, 26, 166, 7, 9)
        ][:50],
        "tas_buttons_head": tas_btn[:40],
        "live_current": live,
        "scripts": [tas_like, late_turn, ground],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tas_n": len(tas_xy),
                "tas_btn0": tas_btn[:12],
                "live_frames": live["frames"],
                "live_vd0": live["vd0_count"],
                "live_first_ground": live["first_ground_after_80"],
                "scripts": [
                    {
                        k: s[k]
                        for k in (
                            "name",
                            "frames",
                            "vd0_count",
                            "first_ground",
                            "max_vy",
                            "end",
                        )
                    }
                    for s in report["scripts"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
