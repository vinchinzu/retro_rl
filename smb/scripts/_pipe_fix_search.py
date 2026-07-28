"""One-off search: land on 1-1 first pipe top then DOWN-enter (continuous seed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from retro_harness.env import make_env
from retro_harness.nes import nes_idle_action
from snes_oneshot.segment_runner import configure_headless
from smb.paths import GAME_DIR, GAME_V0
from smb.policy import (
    DEFAULT_CONTINUOUS_SEED,
    expand_nes9_rle,
    load_nes9_rle_seed,
)
from smb.ram import read_snapshot

B, Y, SEL, ST, UP, DN, LF, RT, A = range(9)
PREFIX = 200
SETTLE = 14


def mf(*idxs: int) -> list[int]:
    f = [0] * 9
    for i in idxs:
        f[i] = 1
    return f


def main() -> int:
    configure_headless()
    print("load seed", flush=True)
    base = expand_nes9_rle(load_nes9_rle_seed(DEFAULT_CONTINUOUS_SEED))
    print("make_env", flush=True)
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    idle = np.asarray(nes_idle_action(), dtype=np.int8)
    asz = int(env.action_space.shape[0])

    def step(buttons: list[int]) -> None:
        act = np.zeros(asz, dtype=np.int8)
        n = min(9, asz, len(buttons))
        act[:n] = np.asarray(buttons[:n], dtype=np.int8)
        env.step(act)

    print("build prefix", flush=True)
    env.reset()
    for _ in range(SETTLE):
        env.step(idle)
    for i in range(PREFIX):
        step(base[i])
    prefix_state = env.em.get_state()
    snap = read_snapshot(env.get_ram(), PREFIX)
    print(f"prefix x={snap.player_x} y={snap.player_y}", flush=True)

    def eval_from(js: int, jh: int, dd: int = 8, dlen: int = 70):
        env.em.set_state(prefix_state)
        lives0 = int(env.get_ram()[0x75A])
        log = []
        for abs_i in range(PREFIX, 520):
            if abs_i < js:
                buttons = mf(B, RT)
            elif abs_i < js + jh:
                buttons = mf(B, RT, A)
            elif abs_i < js + jh + dd:
                buttons = mf(B, RT)
            elif abs_i < js + jh + dd + dlen:
                buttons = mf(DN)
            else:
                buttons = mf(B, RT)
            step(buttons)
            ram = env.get_ram()
            snap = read_snapshot(ram, abs_i + 1)
            xs = int(ram[0x57])
            xs = xs - 256 if xs >= 128 else xs
            area = int(ram[0x750])
            f = abs_i + 1
            log.append((f, snap.player_x, snap.player_y, xs, snap.player_state, area))
            if int(ram[0x75A]) < lives0 or snap.player_state == 0x0B:
                return "death", log
            if snap.player_state == 3 or area == 0xA5:
                return "enter", log
            if snap.player_x > 1050:
                return "overshoot", log
        return "timeout", log

    results: list = []
    for js in range(260, 330, 3):
        for jh in (28, 34, 40, 46, 52):
            out, log = eval_from(js, jh)
            stalls = sum(
                1 for e in log if 880 <= e[1] <= 935 and e[3] == 0 and e[2] > 110
            )
            clean = [
                e
                for e in log
                if e[3] >= 20 and 905 <= e[1] <= 955 and 100 <= e[2] <= 122
            ]
            enter = [e for e in log if e[4] == 3 or e[5] == 0xA5]
            score = 100 + max(e[3] for e in clean) if clean else 0
            if enter:
                score += 150 - max(0, (enter[0][0] - 350) // 2)
            score -= stalls
            results.append(
                (
                    score,
                    js,
                    jh,
                    out,
                    stalls,
                    len(clean),
                    enter[0][0] if enter else None,
                    log,
                )
            )
        print(f"js={js} best={max(r[0] for r in results)}", flush=True)

    results.sort(key=lambda x: -x[0])
    print("TOP 12", flush=True)
    for row in results[:12]:
        score, js, jh, out, st, nc, ef, _log = row
        print(
            f"score={score:4d} js={js} jh={jh} out={out:9s} "
            f"stalls={st:3d} clean={nc:2d} enter_f={ef}",
            flush=True,
        )

    for row in results[:3]:
        score, js, jh, out, st, nc, ef, log = row
        print(f"\nDETAIL js={js} jh={jh} out={out}", flush=True)
        for e in log:
            if e[1] >= 780:
                print(
                    f"  f={e[0]:4d} x={e[1]:4d} y={e[2]:3d} "
                    f"xs={e[3]:3d} ps={e[4]} a={e[5]:02x}",
                    flush=True,
                )

    best_js, best_jh = results[0][1], results[0][2]
    print(f"\nREFINE js~{best_js} jh~{best_jh}", flush=True)
    ref: list = []
    for js in range(best_js - 3, best_js + 4):
        for jh in range(best_jh - 4, best_jh + 5):
            for dd in (4, 8, 12):
                out, log = eval_from(js, jh, dd=dd)
                stalls = sum(
                    1
                    for e in log
                    if 880 <= e[1] <= 935 and e[3] == 0 and e[2] > 110
                )
                clean = [
                    e
                    for e in log
                    if e[3] >= 20 and 905 <= e[1] <= 955 and 100 <= e[2] <= 122
                ]
                enter = [e for e in log if e[4] == 3 or e[5] == 0xA5]
                score = 100 + max(e[3] for e in clean) if clean else 0
                if enter:
                    score += 150 - max(0, (enter[0][0] - 350) // 2)
                score -= stalls
                ref.append(
                    (
                        score,
                        js,
                        jh,
                        dd,
                        out,
                        stalls,
                        len(clean),
                        enter[0][0] if enter else None,
                        log,
                    )
                )
    ref.sort(key=lambda x: -x[0])
    for row in ref[:10]:
        score, js, jh, dd, out, st, nc, ef, _log = row
        print(
            f"score={score:4d} js={js} jh={jh} dd={dd} out={out:9s} "
            f"stalls={st} clean={nc} enter_f={ef}",
            flush=True,
        )

    score, js, jh, dd, out, st, nc, ef, log = ref[0]
    print("\nBEST DETAIL", flush=True)
    for e in log:
        if e[1] >= 780:
            print(
                f"  f={e[0]:4d} x={e[1]:4d} y={e[2]:3d} "
                f"xs={e[3]:3d} ps={e[4]} a={e[5]:02x}",
                flush=True,
            )

    outp = Path("smb/optimizer/runs/smb_1_1/pipe_fix_best.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(
        json.dumps(
            {
                "js": js,
                "jh": jh,
                "dd": dd,
                "score": score,
                "out": out,
                "stalls": st,
                "clean": nc,
                "enter_f": ef,
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", outp, flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
