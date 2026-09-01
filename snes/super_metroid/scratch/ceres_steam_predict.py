#!/usr/bin/env python3
"""Predict Ceres elevator steam/debris burns, then try a debris d-boost.

Scratch only. Claims are graded one frame at a time from the 1611 pin's
fast window. Do not STATUS.
"""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.paths import GAME_DIR
from super_metroid.ram import parse_env_state, probe_pin
from super_metroid.routes.controller_common import POSE_WALL_LATCH
from super_metroid.routes.kpdr.ceres.elev_escape import (
    _ceres_checkpoint_hop,
    _ceres_checkpoint_shaft,
    _ceres_reactive_elev_climb,
    _ceres_try_fast_elev_climb,
)
from super_metroid.routes.runtime import ActionSpan
from super_metroid.routes.skills.knockback import is_knockback

GAME_DIR_SCRATCH = GAME_DIR / "scratch"
PIN_FAST = GAME_DIR_SCRATCH / "ceres_elev_fast_window.state"
PIN_475 = GAME_DIR_SCRATCH / "ceres_elev_475_scan.state"
OUT = GAME_DIR_SCRATCH / "ceres_steam_predict.json"

ENEMY_BASE = 0x0F78
ENEMY_STRIDE = 0x40
MAX_SLOTS = 32
OFF_MAP = 0xFE00
# PJBoy $0F86 bit 2: intangible (no Samus/enemy reaction).
INTANGIBLE_BIT = 0x04
POSE_WALL_LATCH_LEFT = 131


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def _slot(ram, slot: int) -> dict | None:
    base = ENEMY_BASE + slot * ENEMY_STRIDE
    enemy_id = _u16(ram, base)
    if enemy_id == 0:
        return None
    hp = _u16(ram, base + 0x14)
    if hp <= 0:
        return None
    x = _u16(ram, base + 0x02)
    y = _u16(ram, base + 0x06)
    if x >= OFF_MAP or y >= OFF_MAP:
        return None
    props = _u16(ram, base + 0x0E)  # $0F86
    extra = _u16(ram, base + 0x10)  # $0F88
    return {
        "slot": slot,
        "id": enemy_id,
        "id_hex": f"0x{enemy_id:04X}",
        "x": x,
        "y": y,
        "hp": hp,
        "x_rad": _u16(ram, base + 0x0A),
        "y_rad": _u16(ram, base + 0x0C),
        "props": props,
        "extra": extra,
        "intangible": bool((props | extra) & INTANGIBLE_BIT),
        "spritemap": _u16(ram, base + 0x16),  # $0F8E
        "inst_timer": _u16(ram, base + 0x12),  # $0F8A
        "inst_list": _u16(ram, base + 0x24),
        "freeze": _u16(ram, base + 0x26),
    }


def _scan(ram) -> list[dict]:
    out = []
    for slot in range(MAX_SLOTS):
        row = _slot(ram, slot)
        if row is not None:
            out.append(row)
    return out


def _burning(row: dict, *, sx: int, sy: int, dx: int = 24, dy: int = 24) -> bool:
    if row["intangible"]:
        return False
    return abs(int(row["x"]) - sx) <= dx and abs(int(row["y"]) - sy) <= dy


class _Sess:
    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        return self.state

    def span(self, span: ActionSpan) -> None:
        action = buttons(*span.names) if span.names else idle_action()
        for _ in range(span.frames):
            self.step(action, span.reason)

    def wait_until(self, predicate, *, timeout: int, reason: str) -> int:
        for waited in range(timeout + 1):
            if predicate(self.state):
                return waited
            self.step(idle_action(), reason)
        raise TimeoutError(reason)


def _snap(sess: _Sess) -> dict:
    st = sess.state
    ram = sess.env.get_ram()
    enemies = _scan(ram)
    sx, sy = int(st.samus_x), int(st.samus_y)
    return {
        "f": sess.frame,
        "pin": probe_pin(st),
        "x": sx,
        "y": sy,
        "pose": int(st.pose),
        "vy": int(st.velocity_y),
        "mom": int(st.momentum_x),
        "health": int(st.health),
        "kb": int(is_knockback(st)),
        "latch": int(st.pose) in (POSE_WALL_LATCH, POSE_WALL_LATCH_LEFT),
        "enemies": enemies,
        "near_burning": [
            e for e in enemies if _burning(e, sx=sx, sy=sy, dx=40, dy=40)
        ],
    }


def _boot(path: Path) -> tuple[object, _Sess]:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    assist.attach_env(env)
    boot_from_state(env, path, settle_frames=0)
    return env, _Sess(env, assist)


def _reload(sess: _Sess, path: Path) -> None:
    boot_from_state(sess.env, path, settle_frames=0)
    sess.frame = 0
    sess.state = parse_env_state(sess.env, mode="nav")


def _grade(claim: str, predicted, observed, ok: bool) -> dict:
    return {
        "claim": claim,
        "predicted": predicted,
        "observed": observed,
        "ok": bool(ok),
    }


def dump_idle(sess: _Sess, path: Path, frames: int) -> dict:
    """Claim: steam x/y stay put; intangible/spritemap may cycle."""
    _reload(sess, path)
    start = _snap(sess)
    grades = []
    samples = [start]
    prev = start
    first_miss = None
    for _ in range(frames):
        sx, sy = int(sess.state.samus_x), int(sess.state.samus_y)
        prev_by_slot = {e["slot"]: e for e in prev["enemies"]}
        # Predict: each live slot keeps x/y next idle frame.
        sess.step(idle_action(), "idle")
        now = _snap(sess)
        samples.append(now)
        now_by_slot = {e["slot"]: e for e in now["enemies"]}
        xy_hold = all(
            slot in now_by_slot
            and now_by_slot[slot]["x"] == row["x"]
            and now_by_slot[slot]["y"] == row["y"]
            for slot, row in prev_by_slot.items()
        )
        grade = _grade(
            "idle 1f: every live enemy keeps x/y",
            {"xy_hold": True, "from": (sx, sy)},
            {
                "xy_hold": xy_hold,
                "pose": now["pose"],
                "health": now["health"],
                "kb": now["kb"],
            },
            xy_hold,
        )
        grades.append(grade)
        if not grade["ok"] and first_miss is None:
            first_miss = {"frame": now["f"], "grade": grade, "now": now, "prev": prev}
            break
        prev = now
    cycle = []
    for e0, e1 in zip(start["enemies"], samples[-1]["enemies"]):
        cycle.append(
            {
                "slot": e0["slot"],
                "id_hex": e0["id_hex"],
                "start": {
                    k: e0[k]
                    for k in (
                        "x",
                        "y",
                        "intangible",
                        "spritemap",
                        "props",
                        "extra",
                        "inst_list",
                        "inst_timer",
                    )
                },
                "end": {
                    k: e1[k]
                    for k in (
                        "x",
                        "y",
                        "intangible",
                        "spritemap",
                        "props",
                        "extra",
                        "inst_list",
                        "inst_timer",
                    )
                },
                "xy_moved": (e0["x"], e0["y"]) != (e1["x"], e1["y"]),
                "intangible_flipped": e0["intangible"] != e1["intangible"],
                "sprite_flipped": e0["spritemap"] != e1["spritemap"],
            }
        )
    return {
        "start": {k: start[k] for k in start if k != "enemies"},
        "enemies0": start["enemies"],
        "frames_run": len(grades),
        "first_miss": first_miss,
        "cycle": cycle,
        "grades_ok": sum(1 for g in grades if g["ok"]),
        "grades_n": len(grades),
        "samples_every_8": samples[::8],
    }


def run_buttons(sess: _Sess, path: Path, tape: list[tuple[tuple[str, ...], int]]) -> dict:
    _reload(sess, path)
    start = _snap(sess)
    trace = []
    burns = []
    min_y = start["y"]
    for names, n in tape:
        for _ in range(n):
            before = _snap(sess)
            sess.step(buttons(*names) if names else idle_action(), "act")
            now = _snap(sess)
            min_y = min(min_y, now["y"])
            hit = now["kb"] and not before["kb"]
            health_drop = now["health"] < before["health"]
            if hit or health_drop or now["near_burning"]:
                burns.append(
                    {
                        "f": now["f"],
                        "act": names,
                        "before": {
                            k: before[k]
                            for k in ("x", "y", "pose", "health", "kb", "latch")
                        },
                        "after": {
                            k: now[k]
                            for k in ("x", "y", "pose", "health", "kb", "latch")
                        },
                        "near_burning": now["near_burning"],
                    }
                )
            if now["f"] <= 48 or now["kb"] or now["latch"] or hit:
                trace.append(
                    {
                        "f": now["f"],
                        "act": names,
                        "x": now["x"],
                        "y": now["y"],
                        "pose": now["pose"],
                        "health": now["health"],
                        "kb": now["kb"],
                        "latch": now["latch"],
                        "vy": now["vy"],
                        "burning": now["near_burning"],
                    }
                )
    end = _snap(sess)
    return {
        "tape": tape,
        "start": {k: start[k] for k in ("x", "y", "pose", "health", "kb")},
        "end": {
            k: end[k]
            for k in ("x", "y", "pose", "health", "kb", "latch")
        },
        "min_y": min_y,
        "planted_475": abs(end["y"] - 475) <= 18 and end["pose"] not in (25, 26, 137, 138),
        "planted_363": abs(end["y"] - 363) <= 18,
        "burns": burns,
        "trace": trace,
    }


def dboost_on_hit(sess: _Sess, path: Path, approach: list[tuple[tuple[str, ...], int]]) -> dict:
    """Approach until first knockback, then hold A+LEFT (Absorb)."""
    _reload(sess, path)
    start = _snap(sess)
    for names, n in approach:
        for _ in range(n):
            sess.step(buttons(*names) if names else idle_action(), "approach")
            if is_knockback(sess.state):
                hit = _snap(sess)
                for _ in range(24):
                    sess.step(buttons("LEFT", "A"), "dboost")
                    if not is_knockback(sess.state) and int(sess.state.pose) not in (
                        25,
                        26,
                        129,
                        130,
                    ):
                        break
                end = _snap(sess)
                return {
                    "start": {k: start[k] for k in ("x", "y", "pose", "health")},
                    "hit": {
                        k: hit[k]
                        for k in ("f", "x", "y", "pose", "health", "near_burning")
                    },
                    "end": {
                        k: end[k]
                        for k in ("x", "y", "pose", "health", "kb", "latch")
                    },
                    "min_y_after": end["y"],
                    "planted_475": abs(end["y"] - 475) <= 18,
                    "dumped_floor": end["y"] >= 640,
                }
    end = _snap(sess)
    return {
        "start": {k: start[k] for k in ("x", "y", "pose", "health")},
        "hit": None,
        "end": {k: end[k] for k in ("x", "y", "pose", "health", "kb")},
        "miss": "no knockback",
    }


def tas_475_left_wj(sess: _Sess, path: Path) -> dict:
    """Walk to TAS x=163, then LEFT wall-jump toward y=363. No idle."""
    from super_metroid.routes.kpdr.ceres.elev_escape import _ceres_475_to_363

    _reload(sess, path)
    start = _snap(sess)
    for _ in range(40):
        x = int(sess.state.samus_x)
        if x >= 161:
            break
        sess.step(buttons("RIGHT", "B"), "walk_163")
    walked = _snap(sess)
    reached = _ceres_475_to_363(sess)
    end = _snap(sess)
    return {
        "start": {k: start[k] for k in ("x", "y", "pose")},
        "walked": {k: walked[k] for k in ("x", "y", "pose")},
        "end": {k: end[k] for k in ("x", "y", "pose", "kb")},
        "reached_363": bool(reached),
        "frames": sess.frame,
        "dumped_floor": end["y"] >= 640,
    }


def product_shaft(sess: _Sess, path: Path) -> dict:
    _reload(sess, path)
    start = _snap(sess)
    ok = False
    err = None
    try:
        ok = _ceres_checkpoint_shaft(sess)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
    end = _snap(sess)
    return {
        "start": {k: start[k] for k in ("x", "y", "pose")},
        "end": {k: end[k] for k in ("x", "y", "pose", "health", "kb")},
        "ok": bool(ok),
        "err": err,
        "frames": sess.frame,
        "top": abs(end["y"] - 171) <= 18,
        "dumped_floor": end["y"] >= 640,
    }


def product_hop_475(sess: _Sess, path: Path) -> dict:
    _reload(sess, path)
    start = _snap(sess)
    landed = _ceres_checkpoint_hop(
        sess, side="RIGHT", runup=4, target_y=363, start_x=None
    )
    end = _snap(sess)
    return {
        "start": {k: start[k] for k in ("x", "y", "pose", "health")},
        "end": {k: end[k] for k in ("x", "y", "pose", "health", "kb")},
        "landed": bool(landed),
        "frames": sess.frame,
        "planted_363": abs(end["y"] - 363) <= 18,
        "dumped_floor": end["y"] >= 640,
    }


def product_full_climb(sess: _Sess, path: Path) -> dict:
    _reload(sess, path)
    start = _snap(sess)
    err = None
    try:
        _ceres_reactive_elev_climb(sess)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
    end = _snap(sess)
    return {
        "start": {k: start[k] for k in ("x", "y", "pose", "vy")},
        "end": {k: end[k] for k in ("x", "y", "pose", "health", "kb")},
        "gs": int(sess.state.game_state),
        "err": err,
        "frames": sess.frame,
        "ship": int(sess.state.game_state) in (32, 33, 34) or end["y"] <= 80,
        "dumped_floor": end["y"] >= 640,
    }


def product_fast_wj(sess: _Sess, path: Path) -> dict:
    _reload(sess, path)
    start = _snap(sess)
    ok = False
    err = None
    try:
        ok = _ceres_try_fast_elev_climb(sess)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
    end = _snap(sess)
    return {
        "start": {k: start[k] for k in ("x", "y", "pose", "health", "vy")},
        "end": {k: end[k] for k in ("x", "y", "pose", "health", "kb", "latch")},
        "ok": bool(ok),
        "err": err,
        "frames": sess.frame,
        "planted_475": abs(end["y"] - 475) <= 18,
        "dumped_floor": end["y"] >= 640,
    }


def main() -> None:
    env, sess = _boot(PIN_FAST)
    try:
        report = {
            "kind": "ceres_steam_predict",
            "fast_pin": str(PIN_FAST),
            "pin_475": str(PIN_475),
            "idle_fast": dump_idle(sess, PIN_FAST, 64),
            "hold_a": run_buttons(sess, PIN_FAST, [(("A",), 24)]),
            "tas_air_turn": run_buttons(
                sess,
                PIN_FAST,
                [(("A",), 3), (("LEFT", "A"), 8), (("RIGHT", "A"), 16)],
            ),
            "into_right": run_buttons(sess, PIN_FAST, [(("RIGHT", "A"), 22)]),
            "dboost_into_right": dboost_on_hit(
                sess, PIN_FAST, [(("RIGHT", "A"), 22)]
            ),
            "dboost_hold_a": dboost_on_hit(sess, PIN_FAST, [(("A",), 24)]),
        }
        if PIN_475.exists():
            report["idle_475"] = dump_idle(sess, PIN_475, 48)
            report["475_right_hop"] = run_buttons(
                sess, PIN_475, [(("RIGHT", "B"), 4), (("RIGHT", "B", "A"), 40)]
            )
            report["475_dboost"] = dboost_on_hit(
                sess,
                PIN_475,
                [(("RIGHT", "B"), 4), (("RIGHT", "B", "A"), 40)],
            )
            report["475_product_hop"] = product_hop_475(sess, PIN_475)
            report["475_short_hop"] = run_buttons(
                sess,
                PIN_475,
                [
                    (("RIGHT", "B"), 4),
                    (("RIGHT", "B", "A"), 12),
                    (("RIGHT",), 40),
                ],
            )
            report["475_jump40_coast"] = run_buttons(
                sess,
                PIN_475,
                [
                    (("RIGHT", "B"), 4),
                    (("RIGHT", "B", "A"), 40),
                    ((), 80),
                ],
            )
            report["475_jump40_left"] = run_buttons(
                sess,
                PIN_475,
                [
                    (("RIGHT", "B"), 4),
                    (("RIGHT", "B", "A"), 40),
                    (("LEFT",), 80),
                ],
            )
            report["475_tas_left_wj"] = tas_475_left_wj(sess, PIN_475)
            report["475_shaft"] = product_shaft(sess, PIN_475)
        report["fast_product_wj"] = product_fast_wj(sess, PIN_FAST)
        report["fast_full_climb"] = product_full_climb(sess, PIN_FAST)
        OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        idle = report["idle_fast"]
        print(
            f"idle_fast grades={idle['grades_ok']}/{idle['grades_n']} "
            f"first_miss={idle['first_miss'] is not None}"
        )
        ids = sorted({e['id_hex'] for e in idle['enemies0']})
        print(f"enemy ids: {ids} n={len(idle['enemies0'])}")
        for row in idle["cycle"]:
            if row["xy_moved"] or row["intangible_flipped"] or row["sprite_flipped"]:
                print(
                    f"  slot {row['slot']} {row['id_hex']} "
                    f"xy_moved={row['xy_moved']} "
                    f"intangible={row['start']['intangible']}->{row['end']['intangible']} "
                    f"sprite={row['start']['spritemap']:04X}->{row['end']['spritemap']:04X} "
                    f"@({row['start']['x']},{row['start']['y']})"
                )
        for name in (
            "hold_a",
            "tas_air_turn",
            "into_right",
            "dboost_into_right",
            "dboost_hold_a",
            "475_product_hop",
            "475_short_hop",
            "475_jump40_coast",
            "475_jump40_left",
            "475_tas_left_wj",
            "475_shaft",
            "fast_product_wj",
            "fast_full_climb",
        ):
            blob = report[name]
            end = blob.get("end", {})
            print(
                f"{name}: end=({end.get('x')},{end.get('y')}) p{end.get('pose')} "
                f"kb={end.get('kb')} burns={len(blob.get('burns', []))} "
                f"hit={blob.get('hit') is not None} "
                f"planted_475={blob.get('planted_475')} "
                f"dumped={blob.get('dumped_floor')}"
            )
        print(f"report: {OUT}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
