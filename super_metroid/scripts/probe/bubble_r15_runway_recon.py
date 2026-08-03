#!/usr/bin/env python3
"""R15 recon: max-runway + arm-pump + double-WJ lip clear from save door.

Development-only place probe (not pure proof). Goal: beat human peak class
min_y≈158 max_x≈264 and/or clear Phase D (x≥300 y≤200) / right air band.

Human baseline (bubble_jump_try attempt 2):
  seat (27,395) → run 21f RIGHT+B → spin 83f RIGHT+B+A → coast → WJ LEFT+A
  peak (237,160); ceiling pocket blocks right at height.

Maprando 2→7: Running Jump into Right Side Walljump Climb (no Speed).
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
DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "bubble_r15_runway_recon.json"

# Human seat / phase gates
SEAT = (27, 395)
TOP_X, TOP_Y = 300, 200
PHASE_D = lambda x, y: x >= TOP_X and y <= TOP_Y  # noqa: E731
RIGHT_HIGH = lambda x, y: x >= 280 and y <= 220  # noqa: E731
LIP_BAND = lambda x, y: 220 <= x <= 280 and y <= 170  # noqa: E731


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

    def place(self, x: int, y: int, settle: int = 14) -> Any:
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
        settle: int = 12,
        cap: int = 320,
    ) -> dict[str, Any]:
        st = self.place(start[0], start[1], settle=settle)
        start_s = snap(st)
        min_y = int(st.samus_y)
        max_x = int(st.samus_x)
        peak_xy: tuple[int, int] | None = None
        hit_top = hit_right_high = hit_lip = False
        first_top = first_rh = first_lip = None
        max_x_at_y200 = 0
        max_x_at_y170 = 0
        events: list[str] = [f"start {start_s}"]
        for i, btns in enumerate(script[:cap]):
            if int(st.room_id) != ROOM_BUBBLE:
                events.append(f"left_room {snap(st)}")
                break
            st = self.press(*btns)
            x, y = int(st.samus_x), int(st.samus_y)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            if y <= min_y + 1:
                peak_xy = (x, y)
            if y <= 200:
                max_x_at_y200 = max(max_x_at_y200, x)
            if y <= 170:
                max_x_at_y170 = max(max_x_at_y170, x)
            if PHASE_D(x, y) and not hit_top:
                hit_top = True
                first_top = snap(st)
                events.append(f"TOP f={i} {first_top}")
                break
            if RIGHT_HIGH(x, y) and not hit_right_high:
                hit_right_high = True
                first_rh = snap(st)
                events.append(f"RHIGH f={i} {first_rh}")
            if LIP_BAND(x, y) and not hit_lip:
                hit_lip = True
                first_lip = snap(st)
                events.append(f"LIP f={i} {first_lip}")
        events.append(f"end {snap(st)} peak={peak_xy}")
        return {
            "name": name,
            "start": list(start),
            "start_final": start_s,
            "min_y": min_y,
            "max_x": max_x,
            "max_x_at_y200": max_x_at_y200,
            "max_x_at_y170": max_x_at_y170,
            "peak": list(peak_xy) if peak_xy else None,
            "top": hit_top,
            "right_high": hit_right_high,
            "lip_band": hit_lip,
            "first_top": first_top,
            "first_rh": first_rh,
            "first_lip": first_lip,
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


def arm_pump_run(n: int) -> list[tuple[str, ...]]:
    """Classic SM arm-pump: alternate angle-up while dashing RIGHT+B."""
    out: list[tuple[str, ...]] = []
    for i in range(n):
        if i % 2 == 0:
            out.append(("RIGHT", "B", "X"))  # aim-up shoulder-ish via X+dir
        else:
            out.append(("RIGHT", "B", "R"))  # R = angle up on default map
    return out


def arm_pump_run_lr(n: int) -> list[tuple[str, ...]]:
    """Alternate L/R shoulder arm pumps with RIGHT+B."""
    out: list[tuple[str, ...]] = []
    for i in range(n):
        if i % 2 == 0:
            out.append(("RIGHT", "B", "L"))
        else:
            out.append(("RIGHT", "B", "R"))
    return out


def plain_run(n: int) -> list[tuple[str, ...]]:
    return rep(n, "RIGHT", "B")


def human_wj_prefix() -> list[tuple[str, ...]]:
    """Exact human open-loop after spin (residual R14)."""
    return sequence(
        rep(4, "B", "A"),
        rep(1, "B"),
        rep(2, ),
        rep(2, "LEFT"),
        rep(23, "LEFT", "A"),
        rep(6, "A"),
        rep(16, "RIGHT", "A"),
    )


def double_wj_climb(
    *,
    first_left: int = 18,
    a_only: int = 4,
    right_a: int = 10,
    second_left: int = 14,
    right_spin: int = 12,
    third_left: int = 12,
) -> list[tuple[str, ...]]:
    """Two-to-three consecutive walljumps aiming for ceiling lip clear."""
    return sequence(
        rep(2, "LEFT"),
        rep(first_left, "LEFT", "A"),
        rep(a_only, "A"),
        rep(right_a, "RIGHT", "A"),
        rep(2, "RIGHT"),
        rep(second_left, "LEFT", "A"),
        rep(a_only, "A"),
        rep(right_spin, "RIGHT", "B", "A"),
        rep(third_left, "LEFT", "A"),
        rep(20, "RIGHT", "B", "A"),
    )


def period_wj(n: int, period: int = 6, into: int = 1, bounce: int = 2) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for i in range(n):
        ph = i % period
        if ph < into:
            out.append(("RIGHT", "B"))
        elif ph < into + bounce:
            out.append(("LEFT", "A"))
        else:
            out.append(("RIGHT", "B", "A"))
    return out


def build_script(
    *,
    clear: bool,
    run_kind: str,
    run_f: int,
    spin_f: int,
    wj: str,
) -> list[tuple[str, ...]]:
    chunks: list[list[tuple[str, ...]]] = []
    if clear:
        # Brief beam spray right then re-seat left on runway (bug clear).
        chunks.append(rep(8, "RIGHT", "Y"))
        chunks.append(rep(4, "RIGHT", "B", "Y"))
        chunks.append(rep(10, "LEFT", "B"))
        chunks.append(rep(4, ))
    if run_kind == "plain":
        chunks.append(plain_run(run_f))
    elif run_kind == "pump_x":
        chunks.append(arm_pump_run(run_f))
    elif run_kind == "pump_lr":
        chunks.append(arm_pump_run_lr(run_f))
    else:
        raise ValueError(run_kind)
    chunks.append(rep(spin_f, "RIGHT", "B", "A"))
    if wj == "human":
        chunks.append(human_wj_prefix())
        chunks.append(period_wj(80, 8, 2, 2))
    elif wj == "double":
        chunks.append(double_wj_climb())
        chunks.append(period_wj(60, 6, 1, 2))
    elif wj == "double_tight":
        chunks.append(
            double_wj_climb(
                first_left=20,
                a_only=3,
                right_a=8,
                second_left=16,
                right_spin=8,
                third_left=14,
            )
        )
        chunks.append(period_wj(60, 5, 1, 2))
    elif wj == "double_early":
        # Contact earlier: less spin then WJ thrash
        chunks.append(
            sequence(
                rep(2, "B", "A"),
                rep(1, ),
                rep(1, "LEFT"),
                rep(28, "LEFT", "A"),
                rep(4, "A"),
                rep(10, "RIGHT", "A"),
                rep(18, "LEFT", "A"),
                rep(8, "RIGHT", "B", "A"),
                rep(16, "LEFT", "A"),
                rep(24, "RIGHT", "B", "A"),
            )
        )
        chunks.append(period_wj(50, 5, 1, 2))
    elif wj == "right_face":
        # Maprando right-side WJ: face right wall, bounce left then right climb
        chunks.append(
            sequence(
                rep(6, "RIGHT", "B", "A"),
                rep(4, "RIGHT"),
                rep(2, "LEFT"),
                rep(20, "LEFT", "A"),
                rep(3, "A"),
                rep(12, "RIGHT", "A"),
                rep(3, "RIGHT"),
                rep(16, "LEFT", "A"),
                rep(3, "A"),
                rep(14, "RIGHT", "B", "A"),
                rep(14, "LEFT", "A"),
                rep(30, "RIGHT", "B", "A"),
            )
        )
        chunks.append(period_wj(60, 6, 1, 2))
    else:
        raise ValueError(wj)
    return sequence(*chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--quick", action="store_true", help="fewer grid cells")
    args = ap.parse_args()

    p = Probe(args.source)
    results: list[dict[str, Any]] = []

    def add(r: dict[str, Any]) -> None:
        results.append(r)
        tags = []
        if r["top"]:
            tags.append("TOP")
        if r["right_high"]:
            tags.append("RH")
        if r["lip_band"]:
            tags.append("LIP")
        tag = "+".join(tags) if tags else "—"
        print(
            f"{r['name']:52} min_y={r['min_y']:4} max_x={r['max_x']:4} "
            f"x@y200={r['max_x_at_y200']:3} x@y170={r['max_x_at_y170']:3} "
            f"[{tag}] peak={r['peak']} end=({r['end']['x']},{r['end']['y']})"
        )

    starts = [(24, 395), (27, 395), (32, 395), (40, 395)]
    run_fs = [18, 21, 24, 28, 32, 36] if not args.quick else [21, 28, 36]
    spin_fs = [70, 83, 95] if not args.quick else [83]
    run_kinds = ["plain", "pump_lr", "pump_x"] if not args.quick else ["plain", "pump_lr"]
    wjs = ["human", "double", "double_tight", "right_face"] if not args.quick else [
        "human",
        "double",
        "right_face",
    ]

    try:
        # 1) Exact human replay baseline from seat
        add(
            p.run(
                "baseline_human_r14",
                SEAT,
                build_script(
                    clear=False, run_kind="plain", run_f=21, spin_f=83, wj="human"
                ),
            )
        )

        # 2) Clear-bug prefix + human
        add(
            p.run(
                "clear_then_human",
                SEAT,
                build_script(
                    clear=True, run_kind="plain", run_f=21, spin_f=83, wj="human"
                ),
            )
        )

        # 3) Grid: start x × run × spin × pump × wj (pruned)
        for sx, sy in starts:
            for rk in run_kinds:
                for rf in run_fs:
                    for sf in spin_fs:
                        for wj in wjs:
                            # skip obviously redundant cells in full mode
                            if not args.quick:
                                if rk == "pump_x" and wj not in ("double", "human"):
                                    continue
                                if sx == 40 and rf < 24:
                                    continue
                                if sx == 24 and rf > 32 and wj == "human":
                                    pass
                            name = f"s{sx}_{rk}_r{rf}_sp{sf}_{wj}"
                            add(
                                p.run(
                                    name,
                                    (sx, sy),
                                    build_script(
                                        clear=False,
                                        run_kind=rk,
                                        run_f=rf,
                                        spin_f=sf,
                                        wj=wj,
                                    ),
                                )
                            )

        # 4) Clear + best-looking combos (max left + pump + double WJ)
        for rf in (24, 28, 32):
            for wj in ("double", "double_tight", "right_face"):
                name = f"clear_s27_pump_r{rf}_{wj}"
                add(
                    p.run(
                        name,
                        SEAT,
                        build_script(
                            clear=True,
                            run_kind="pump_lr",
                            run_f=rf,
                            spin_f=83,
                            wj=wj,
                        ),
                    )
                )

        # 5) Extra: longer consecutive WJ thrash after max runway
        for rf in (28, 36):
            script = sequence(
                arm_pump_run_lr(rf),
                rep(80, "RIGHT", "B", "A"),
                rep(2, "LEFT"),
                rep(22, "LEFT", "A"),
                rep(3, "A"),
                rep(8, "RIGHT", "A"),
                rep(18, "LEFT", "A"),
                rep(3, "A"),
                rep(10, "RIGHT", "B", "A"),
                rep(16, "LEFT", "A"),
                rep(3, "A"),
                rep(12, "RIGHT", "B", "A"),
                rep(14, "LEFT", "A"),
                rep(40, "RIGHT", "B", "A"),
            )
            add(p.run(f"triple_wj_r{rf}", (24, 395), script))

    finally:
        p.close()

    # Rank by (top, right_high, -min_y, max_x_at_y200, max_x)
    ranked = sorted(
        results,
        key=lambda r: (
            r["top"],
            r["right_high"],
            r["lip_band"],
            -r["min_y"],
            r["max_x_at_y200"],
            r["max_x_at_y170"],
            r["max_x"],
        ),
        reverse=True,
    )
    summary = {
        "n": len(results),
        "best10": [
            {
                "name": r["name"],
                "min_y": r["min_y"],
                "max_x": r["max_x"],
                "max_x_at_y200": r["max_x_at_y200"],
                "max_x_at_y170": r["max_x_at_y170"],
                "top": r["top"],
                "right_high": r["right_high"],
                "lip_band": r["lip_band"],
                "peak": r["peak"],
            }
            for r in ranked[:10]
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print("\n=== BEST 10 ===")
    for b in summary["best10"]:
        print(b)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
