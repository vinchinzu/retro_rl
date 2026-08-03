#!/usr/bin/env python3
"""Micro-sweep double-WJ / lip-clear after proven human r21/s83 launch.

Boots bubble_human_runway.state. Launch fixed. Vary only approach + WJ chain
to maximize (min_y improvement, max_x at y≤200, max_x at y≤170).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402

PIN = (
    ROOT
    / "super_metroid"
    / "custom_integrations"
    / "SuperMetroid-Snes"
    / "scratch"
    / "bubble_human_runway.state"
)
OUT = ROOT / "super_metroid" / "debug" / "bubble_r15_wj_micro.json"
ROOM_BUBBLE = 0xACB3


def snap(st: Any) -> dict[str, Any]:
    return {
        "x": int(st.samus_x),
        "y": int(st.samus_y),
        "p": int(st.pose),
        "vx": int(st.velocity_x),
        "vy": int(st.velocity_y),
    }


def rep(n: int, *b: str) -> list[tuple[str, ...]]:
    return [b] * n


def seq(*cs: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    o: list[tuple[str, ...]] = []
    for c in cs:
        o.extend(c)
    return o


LAUNCH = seq(rep(21, "RIGHT", "B"), rep(83, "RIGHT", "B", "A"))


class P:
    def __init__(self) -> None:
        self.env = make_dev_env()
        self.assist = UnlimitedResourcesAssist()
        self.frame = 0

    def close(self) -> None:
        self.env.close()

    def boot(self) -> Any:
        boot_from_state(self.env, PIN)
        self.frame = 0
        st = parse_env_state(self.env, frame=0, mode="full")
        self.assist.apply(self.env.data, st)
        for _ in range(2):
            st = self.step()
        return st

    def step(self, *names: str) -> Any:
        self.env.step(buttons(*names) if names else idle_action())
        st = parse_env_state(self.env, frame=self.frame, mode="full")
        self.assist.apply(self.env.data, st)
        self.frame += 1
        return st

    def run(self, name: str, script: list[tuple[str, ...]]) -> dict[str, Any]:
        st = self.boot()
        min_y = int(st.samus_y)
        max_x = int(st.samus_x)
        mx200 = mx170 = mx150 = mx142 = 0
        peak = (int(st.samus_x), int(st.samus_y))
        top = False
        first_top = None
        # track first wall contact-ish (pose 132 walljump)
        first_wj = None
        trail: list[dict[str, Any]] = []
        for i, btns in enumerate(script[:400]):
            st = self.step(*btns)
            if int(st.room_id) not in (ROOM_BUBBLE, 0xB07A):
                break
            x, y = int(st.samus_x), int(st.samus_y)
            p = int(st.pose)
            if y < min_y:
                min_y = y
                peak = (x, y)
            max_x = max(max_x, x)
            if y <= 200:
                mx200 = max(mx200, x)
            if y <= 170:
                mx170 = max(mx170, x)
            if y <= 150:
                mx150 = max(mx150, x)
            if y <= 142:
                mx142 = max(mx142, x)
            if p == 132 and first_wj is None:
                first_wj = {"i": i, "x": x, "y": y}
            if y <= 200 and x >= 300 and not top:
                top = True
                first_top = snap(st)
                break
            if y <= 220 and (i % 4 == 0 or p in (132, 83, 25)):
                trail.append({"i": i, "x": x, "y": y, "p": p, "vy": int(st.velocity_y)})
        return {
            "name": name,
            "min_y": min_y,
            "max_x": max_x,
            "mx200": mx200,
            "mx170": mx170,
            "mx150": mx150,
            "mx142": mx142,
            "peak": list(peak),
            "top": top,
            "first_top": first_top,
            "first_wj": first_wj,
            "end": snap(st),
            "trail": trail[-20:],
        }


def wj_chain(
    *,
    coast_ba: int = 4,
    idle: int = 2,
    turn: int = 2,
    la1: int = 23,
    amid: int = 6,
    ra1: int = 16,
    la2: int = 0,
    amid2: int = 3,
    ra2: int = 0,
    la3: int = 0,
    finish: str = "right_spin",
    finish_n: int = 40,
) -> list[tuple[str, ...]]:
    parts: list[list[tuple[str, ...]]] = [
        rep(coast_ba, "B", "A") if coast_ba else [],
        rep(1, "B") if coast_ba else [],
        rep(idle, ) if idle else [],
        rep(turn, "LEFT") if turn else [],
        rep(la1, "LEFT", "A"),
    ]
    if amid:
        parts.append(rep(amid, "A"))
    if ra1:
        parts.append(rep(ra1, "RIGHT", "A"))
    if la2:
        parts.append(rep(la2, "LEFT", "A"))
        if amid2:
            parts.append(rep(amid2, "A"))
        if ra2:
            parts.append(rep(ra2, "RIGHT", "A"))
    if la3:
        parts.append(rep(la3, "LEFT", "A"))
    if finish == "right_spin":
        parts.append(rep(finish_n, "RIGHT", "B", "A"))
    elif finish == "period":
        for i in range(finish_n):
            ph = i % 6
            if ph < 1:
                parts.append([("RIGHT", "B")])
            elif ph < 3:
                parts.append([("LEFT", "A")])
            else:
                parts.append([("RIGHT", "B", "A")])
    elif finish == "right_up":
        parts.append(rep(finish_n, "RIGHT", "UP", "A"))
    elif finish == "alt_wj":
        for i in range(finish_n):
            if i % 8 < 3:
                parts.append([("LEFT", "A")])
            elif i % 8 < 5:
                parts.append([("RIGHT", "A")])
            else:
                parts.append([("RIGHT", "B", "A")])
    return seq(*[p for p in parts if p])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    probe = P()
    results: list[dict[str, Any]] = []

    def add(r: dict[str, Any]) -> None:
        results.append(r)
        tag = "TOP" if r["top"] else "—"
        print(
            f"{r['name']:56} min_y={r['min_y']:4} mx={r['max_x']:3} "
            f"@200={r['mx200']:3} @170={r['mx170']:3} @150={r['mx150']:3} "
            f"@142={r['mx142']:3} [{tag}] peak={r['peak']} "
            f"wj={r['first_wj']}"
        )

    try:
        # Baseline human
        add(probe.run("baseline", seq(LAUNCH, wj_chain())))

        # Extra single WJ (known min_y~145)
        add(
            probe.run(
                "extra_wj14",
                seq(
                    LAUNCH,
                    wj_chain(),
                    rep(14, "LEFT", "A"),
                    rep(4, "A"),
                    rep(12, "RIGHT", "A"),
                    rep(16, "LEFT", "A"),
                    rep(30, "RIGHT", "B", "A"),
                ),
            )
        )

        # Coast / approach timing (coarse)
        for coast in (0, 2, 4, 6):
            for idle in (0, 2):
                for turn in (1, 2):
                    name = f"approach_c{coast}_i{idle}_t{turn}"
                    add(
                        probe.run(
                            name,
                            seq(
                                LAUNCH,
                                wj_chain(coast_ba=coast, idle=idle, turn=turn),
                            ),
                        )
                    )

        # First WJ length (coarse)
        for la1 in (16, 20, 23, 26, 30):
            for ra1 in (0, 8, 16, 24):
                for amid in (0, 6):
                    name = f"wj1_L{la1}_a{amid}_R{ra1}"
                    add(
                        probe.run(
                            name,
                            seq(
                                LAUNCH,
                                wj_chain(la1=la1, amid=amid, ra1=ra1),
                            ),
                        )
                    )

        # Double WJ grid (focused)
        for la1 in (20, 23, 26):
            for ra1 in (8, 12, 16):
                for la2 in (12, 16, 20, 24):
                    for ra2 in (0, 8, 14):
                        for la3 in (0, 12, 18):
                            name = f"dwj_L{la1}R{ra1}L{la2}R{ra2}L{la3}"
                            add(
                                probe.run(
                                    name,
                                    seq(
                                        LAUNCH,
                                        wj_chain(
                                            la1=la1,
                                            amid=4,
                                            ra1=ra1,
                                            la2=la2,
                                            amid2=2,
                                            ra2=ra2,
                                            la3=la3,
                                            finish="alt_wj",
                                            finish_n=48,
                                        ),
                                    ),
                                )
                            )

        # Finish style after human WJ
        for fin, n in (
            ("right_spin", 50),
            ("period", 60),
            ("right_up", 40),
            ("alt_wj", 60),
        ):
            add(
                probe.run(
                    f"fin_{fin}",
                    seq(LAUNCH, wj_chain(finish=fin, finish_n=n)),
                )
            )

        # Early wall seek: less spin then WJ
        for spin in (70, 75, 78, 80, 83, 86, 88, 92):
            launch = seq(rep(21, "RIGHT", "B"), rep(spin, "RIGHT", "B", "A"))
            add(
                probe.run(
                    f"spin{spin}_dwj",
                    seq(
                        launch,
                        wj_chain(
                            la1=24,
                            amid=3,
                            ra1=10,
                            la2=18,
                            ra2=12,
                            la3=14,
                            finish="alt_wj",
                            finish_n=50,
                        ),
                    ),
                )
            )

        # Damage-boost style: hold right into enemy after peak
        add(
            probe.run(
                "dmgboost_right",
                seq(
                    LAUNCH,
                    wj_chain(la1=23, ra1=16),
                    rep(8, "RIGHT"),
                    rep(40, "RIGHT", "B"),
                    rep(20, "RIGHT", "B", "A"),
                ),
            )
        )

        # Walljump with charge beam held (canWallJumpWithCharge style)
        add(
            probe.run(
                "charge_wj",
                seq(
                    LAUNCH,
                    rep(4, "B", "A", "Y"),
                    rep(2, "LEFT", "Y"),
                    rep(24, "LEFT", "A", "Y"),
                    rep(4, "A", "Y"),
                    rep(14, "RIGHT", "A", "Y"),
                    rep(18, "LEFT", "A", "Y"),
                    rep(30, "RIGHT", "B", "A"),
                ),
            )
        )

    finally:
        probe.close()

    ranked = sorted(
        results,
        key=lambda r: (
            r["top"],
            r["mx200"] >= 300,
            r["mx170"] >= 280,
            r["mx150"] >= 260,
            -r["min_y"],
            r["mx200"],
            r["mx170"],
            r["mx150"],
            r["mx142"],
            r["max_x"],
        ),
        reverse=True,
    )
    summary = {
        "n": len(results),
        "best20": [
            {
                k: r[k]
                for k in (
                    "name",
                    "min_y",
                    "max_x",
                    "mx200",
                    "mx170",
                    "mx150",
                    "mx142",
                    "top",
                    "peak",
                    "first_wj",
                    "end",
                )
            }
            for r in ranked[:20]
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print("\n=== BEST 20 ===")
    for b in summary["best20"]:
        print(b)
    print(f"wrote {args.output} n={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
