#!/usr/bin/env python3
"""R10 recon: mid-high approach into right air band for Bubble → Bat.

Development-only. Reuses one env + place_samus (not pure proof).
Target: hit air band x≥340 y∈[280,340] or shelf/top from natural-capable
starts (lip / air place / left wall).
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
from super_metroid.dev.common import boot_from_state, make_dev_env, place_samus  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402

ROOM_BUBBLE = 0xACB3
DEFAULT_SOURCE = (
    ROOT
    / "super_metroid"
    / "custom_integrations"
    / "SuperMetroid-Snes"
    / "scratch"
    / "post_rising_tide_to_bubble_pure.state"
)
DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "bubble_r10_midhigh_recon.json"
LIP = (79, 427)
AIR_X0, AIR_Y0, AIR_X1, AIR_Y1 = 340, 280, 395, 340
TOP_X, TOP_Y = 300, 200
SHELF_X, SHELF_Y = 300, 390
STAND = frozenset({1, 2, 9, 10, 25, 26, 27, 28})


def snap(st: Any) -> dict[str, int]:
    return {
        "x": int(st.samus_x),
        "y": int(st.samus_y),
        "p": int(st.pose),
        "vx": int(st.velocity_x),
        "vy": int(st.velocity_y),
    }


class Probe:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.env = make_dev_env()
        self.assist = UnlimitedResourcesAssist()
        self.frame = 0
        self._boot()

    def _boot(self) -> None:
        boot_from_state(self.env, self.source)
        self.frame = 0
        for _ in range(6):
            self.idle()

    def close(self) -> None:
        self.env.close()

    def rearm(self) -> None:
        # Full reboot is expensive; prefer place after reload when room drifts.
        self.env.close()
        self.env = make_dev_env()
        self.assist = UnlimitedResourcesAssist()
        self._boot()

    def idle(self) -> Any:
        self.env.step(idle_action())
        st = parse_env_state(self.env, frame=self.frame, mode="full")
        self.assist.apply(self.env.data, st)
        self.frame += 1
        return st

    def press(self, *names: str) -> Any:
        self.env.step(buttons(*names) if names else idle_action())
        st = parse_env_state(self.env, frame=self.frame, mode="full")
        self.assist.apply(self.env.data, st)
        self.frame += 1
        return st

    def place(self, x: int, y: int, settle: int = 12) -> Any:
        if int(parse_env_state(self.env, mode="full").room_id) != ROOM_BUBBLE:
            self.rearm()
        place_samus(self.env, x, y)
        st = self.idle()
        for _ in range(settle):
            st = self.idle()
        return st

    def run(
        self,
        name: str,
        start: tuple[int, int],
        script: list[tuple[str, ...]],
        *,
        settle: int = 10,
        cap: int = 280,
    ) -> dict[str, Any]:
        st = self.place(start[0], start[1], settle=settle)
        start_s = snap(st)
        min_y = int(st.samus_y)
        max_x = int(st.samus_x)
        hit_air = hit_shelf = hit_top = False
        first_air = first_shelf = first_top = None
        # Sparse events: start + milestones + end
        events: list[str] = [f"start {start_s}"]
        # Track first time x crosses thresholds at height
        milestones: list[str] = []
        for i, btns in enumerate(script[:cap]):
            if int(st.room_id) != ROOM_BUBBLE:
                events.append(f"left_room {snap(st)}")
                break
            st = self.press(*btns)
            x, y = int(st.samus_x), int(st.samus_y)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            air = AIR_X0 <= x <= AIR_X1 and AIR_Y0 <= y <= AIR_Y1
            shelf = (
                abs(int(st.velocity_y)) <= 1
                and int(st.pose) in STAND
                and x >= SHELF_X
                and 200 <= y <= SHELF_Y
            )
            top = y <= TOP_Y and x >= TOP_X
            if air and not hit_air:
                hit_air = True
                first_air = snap(st)
                events.append(f"AIR f={i} {first_air}")
            if shelf and not hit_shelf:
                hit_shelf = True
                first_shelf = snap(st)
                events.append(f"SHELF f={i} {first_shelf}")
            if top and not hit_top:
                hit_top = True
                first_top = snap(st)
                events.append(f"TOP f={i} {first_top}")
                break
            if i % 20 == 0 and i > 0:
                milestones.append(f"f{i} {snap(st)}")
        events.extend(milestones[-4:])
        events.append(f"end {snap(st)}")
        return {
            "name": name,
            "start": list(start),
            "start_final": start_s,
            "min_y": min_y,
            "max_x": max_x,
            "air": hit_air,
            "shelf": hit_shelf,
            "top": hit_top,
            "first_air": first_air,
            "first_shelf": first_shelf,
            "first_top": first_top,
            "end": snap(st),
            "frames": min(len(script), cap),
            "events": events,
        }


def sequence(*chunks: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for c in chunks:
        out.extend(c)
    return out


def rep(n: int, *btns: str) -> list[tuple[str, ...]]:
    return [btns] * n


def period_wj(
    n: int,
    period: int = 8,
    into: int = 2,
    bounce: int = 2,
    *,
    face: str = "RIGHT",
) -> list[tuple[str, ...]]:
    opp = "LEFT" if face == "RIGHT" else "RIGHT"
    out: list[tuple[str, ...]] = []
    for i in range(n):
        ph = i % period
        if ph < into:
            out.append((face, "B"))
        elif ph < into + bounce:
            out.append((opp, "A"))
        else:
            out.append((face, "B", "A"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    p = Probe(args.source)
    results: list[dict[str, Any]] = []

    def add(r: dict[str, Any]) -> None:
        results.append(r)
        tag = (
            "TOP"
            if r["top"]
            else ("SHELF" if r["shelf"] else ("AIR" if r["air"] else "—"))
        )
        print(
            f"{r['name']:48} min_y={r['min_y']:4} max_x={r['max_x']:4} "
            f"[{tag}] end=({r['end']['x']},{r['end']['y']}) p={r['end']['p']}"
        )

    try:
        # ========== 1. Lip launch variants (solid lip place) ==========
        lip_variants: list[tuple[str, list[tuple[str, ...]]]] = [
            (
                "lip_r9",
                sequence(
                    rep(12, "A"),
                    rep(44, "RIGHT", "B", "A"),
                    rep(70, "RIGHT", "B", "A"),
                    period_wj(100, 8, 2, 2),
                ),
            ),
            (
                "lip_charge16_spin120",
                sequence(rep(16, "A"), rep(120, "RIGHT", "B", "A"), period_wj(80)),
            ),
            (
                "lip_up8_then_right",
                sequence(
                    rep(12, "A"),
                    rep(8, "UP", "A"),
                    rep(100, "RIGHT", "B", "A"),
                    period_wj(80),
                ),
            ),
            # After peak (~frame 40 of HJ), push LEFT for wall contact then WJ up, then cross
            (
                "lip_peak_left_wj_cross",
                sequence(
                    rep(12, "A"),
                    rep(40, "RIGHT", "B", "A"),  # to peak
                    rep(16, "LEFT", "B", "A"),  # into left wall
                    period_wj(60, 8, 2, 2, face="LEFT"),
                    rep(20, "RIGHT", "B", "A"),
                    period_wj(100, 8, 2, 2, face="RIGHT"),
                ),
            ),
            (
                "lip_peak_left_wj_longer",
                sequence(
                    rep(14, "A"),
                    rep(36, "RIGHT", "B", "A"),
                    rep(10, "LEFT", "B", "A"),
                    period_wj(90, 8, 2, 2, face="LEFT"),
                    rep(30, "RIGHT", "B", "A"),
                    period_wj(100, 8, 2, 2, face="RIGHT"),
                ),
            ),
            # Double hop: short first HJ, land, immediate second hop
            (
                "lip_short_rejump",
                sequence(
                    rep(10, "A"),
                    rep(28, "RIGHT", "B", "A"),
                    rep(3, ),  # release
                    rep(4, "A"),
                    rep(56, "RIGHT", "B", "A"),
                    period_wj(100),
                ),
            ),
            # Morph bomb at mid of first arc
            (
                "lip_mid_bomb",
                sequence(
                    rep(10, "A"),
                    rep(32, "RIGHT", "B", "A"),
                    rep(3, "DOWN"),
                    rep(1, "A"),  # bomb
                    rep(20, ),
                    rep(4, "UP"),
                    rep(80, "RIGHT", "B", "A"),
                    period_wj(80),
                ),
            ),
            # Hold A longer on second phase with diagonal
            (
                "lip_spin_then_wj_early",
                sequence(
                    rep(12, "A"),
                    rep(50, "RIGHT", "B", "A"),
                    period_wj(140, 8, 2, 2),  # WJ from mid-cavity if wall
                ),
            ),
            (
                "lip_spin_wj_p6",
                sequence(
                    rep(12, "A"),
                    rep(50, "RIGHT", "B", "A"),
                    period_wj(140, 6, 2, 2),
                ),
            ),
            (
                "lip_spin_wj_p10",
                sequence(
                    rep(12, "A"),
                    rep(50, "RIGHT", "B", "A"),
                    period_wj(140, 10, 2, 2),
                ),
            ),
        ]
        for name, script in lip_variants:
            add(p.run(name, LIP, script, settle=14))

        # ========== 2. Left wall WJ up from solid lip / mid-left, then cross ==========
        for start in ((79, 427), (70, 400), (90, 380), (85, 350), (100, 320)):
            script = sequence(
                rep(8, "A"),
                rep(20, "LEFT", "B", "A"),
                period_wj(80, 8, 2, 2, face="LEFT"),
                rep(30, "RIGHT", "B", "A"),
                period_wj(100, 8, 2, 2, face="RIGHT"),
            )
            add(p.run(f"left_up_cross_{start[0]}_{start[1]}", start, script, settle=12))

        # ========== 3. Right air / low WJ — map which heights can still top ==========
        for sy in (280, 300, 320, 340, 360, 380, 400, 420, 450, 480):
            for sx in (320, 340, 360, 370, 380):
                script = period_wj(140, 8, 2, 2)
                add(p.run(f"rwj8_{sx}_{sy}", (sx, sy), script, settle=3, cap=160))

        # period variants at critical y=360/380/400
        for sy in (350, 360, 370, 380, 390, 400):
            for period, into, bounce in (
                (6, 2, 2),
                (8, 2, 2),
                (8, 3, 2),
                (8, 2, 3),
                (10, 2, 2),
                (10, 3, 2),
                (12, 2, 2),
            ):
                script = period_wj(140, period, into, bounce)
                add(
                    p.run(
                        f"rwj_p{period}i{into}b{bounce}_{360}_{sy}",
                        (360, sy),
                        script,
                        settle=3,
                        cap=160,
                    )
                )

        # ========== 4. Air drop no-A into shelf from high right ==========
        for sy in (250, 270, 290, 310, 330):
            for sx in (350, 360, 370, 380):
                script = sequence(rep(40, "RIGHT", "B"), rep(40, "LEFT", "B"))
                add(p.run(f"drop_{sx}_{sy}", (sx, sy), script, settle=2, cap=90))

        # ========== 5. Mid-cavity air hop chain (not grounded) ==========
        for start in (
            (150, 280),
            (180, 300),
            (200, 320),
            (220, 300),
            (250, 320),
            (280, 300),
            (300, 320),
        ):
            script = sequence(
                rep(60, "RIGHT", "B", "A"),
                period_wj(100, 8, 2, 2),
            )
            add(p.run(f"aircross_{start[0]}_{start[1]}", start, script, settle=2, cap=180))

        # ========== 6. Bomb boost from lip with better timing ==========
        # Place lip, jump, morph near apex, bomb, unmorph into spin
        for bomb_at in (24, 28, 32, 36, 40, 44):
            script = sequence(
                rep(12, "A"),
                rep(bomb_at, "RIGHT", "B", "A"),
                rep(2, "DOWN"),
                rep(1, "A"),
                rep(18, ),
                rep(3, "UP"),
                rep(90, "RIGHT", "B", "A"),
                period_wj(80),
            )
            add(p.run(f"lip_bj_f{bomb_at}", LIP, script, settle=14))

    finally:
        p.close()

    tops = [r for r in results if r["top"]]
    airs = [r for r in results if r["air"] and not r["top"]]
    shelves = [r for r in results if r["shelf"] and not r["top"]]
    lips = [r for r in results if r["name"].startswith("lip_")]
    best_lip = sorted(
        lips,
        key=lambda r: (
            not r["top"],
            not r["air"],
            not r["shelf"],
            r["min_y"],
            -r["max_x"],
        ),
    )[:8]
    # Best right WJ that hits top from highest y (lowest start)
    rwj_tops = [r for r in tops if r["name"].startswith("rwj")]
    max_start_y_top = max((r["start"][1] for r in rwj_tops), default=None)
    summary = {
        "n": len(results),
        "tops": len(tops),
        "airs": len(airs),
        "shelves": len(shelves),
        "top_names": [r["name"] for r in tops[:40]],
        "air_names": [r["name"] for r in airs[:30]],
        "shelf_names": [r["name"] for r in shelves[:30]],
        "best_lip": [
            {
                "name": r["name"],
                "min_y": r["min_y"],
                "max_x": r["max_x"],
                "air": r["air"],
                "shelf": r["shelf"],
                "top": r["top"],
                "end": r["end"],
                "events": r["events"][:8],
            }
            for r in best_lip
        ],
        "max_start_y_for_rwj_top": max_start_y_top,
        "rwj_tops_by_start_y": sorted(
            [
                {
                    "name": r["name"],
                    "start": r["start"],
                    "min_y": r["min_y"],
                    "max_x": r["max_x"],
                }
                for r in rwj_tops
            ],
            key=lambda r: -r["start"][1],
        )[:25],
    }
    out = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2)[:4000])
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
