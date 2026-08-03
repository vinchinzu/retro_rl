#!/usr/bin/env python3
"""R15 recon from human pin states (runway / peak / ceiling).

Not pure proof. Boots ``scratch/bubble_human_*.state`` so velocity/pose match
the human demo, then sweeps max-runway, arm-pump, bug-clear, and double-WJ
lip-clear open-loops.

Human attempt-2:
  runway seat ~(27,395) → run 21 + spin 83 → WJ → peak (237,160)
  ceiling pocket ~y142 x240–250 blocks right at height.
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

SCRATCH = (
    ROOT
    / "super_metroid"
    / "custom_integrations"
    / "SuperMetroid-Snes"
    / "scratch"
)
DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "bubble_r15_human_pin_recon.json"
ROOM_BUBBLE = 0xACB3
TOP_X, TOP_Y = 300, 200


def snap(st: Any) -> dict[str, Any]:
    return {
        "x": int(st.samus_x),
        "y": int(st.samus_y),
        "p": int(st.pose),
        "vx": int(st.velocity_x),
        "vy": int(st.velocity_y),
        "room": f"0x{int(st.room_id):04X}",
    }


class PinProbe:
    def __init__(self) -> None:
        self.env = make_dev_env()
        self.assist = UnlimitedResourcesAssist()
        self.frame = 0

    def close(self) -> None:
        self.env.close()

    def boot(self, state_path: Path) -> Any:
        boot_from_state(self.env, state_path)
        self.frame = 0
        st = parse_env_state(self.env, frame=0, mode="full")
        self.assist.apply(self.env.data, st)
        for _ in range(2):
            st = self.idle()
        return st

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

    def run_script(
        self,
        name: str,
        pin: Path,
        script: list[tuple[str, ...]],
        *,
        cap: int = 360,
    ) -> dict[str, Any]:
        st = self.boot(pin)
        start = snap(st)
        min_y = int(st.samus_y)
        max_x = int(st.samus_x)
        max_x_y200 = 0
        max_x_y170 = 0
        max_x_y150 = 0
        peak = (int(st.samus_x), int(st.samus_y))
        hit_top = False
        first_top = None
        path_hi: list[dict[str, Any]] = []
        for i, btns in enumerate(script[:cap]):
            if int(st.room_id) != ROOM_BUBBLE and int(st.room_id) != 0xB07A:
                break
            if int(st.room_id) == 0xB07A:
                hit_top = True
                first_top = snap(st)
                break
            st = self.press(*btns)
            x, y = int(st.samus_x), int(st.samus_y)
            if y < min_y:
                min_y = y
                peak = (x, y)
            max_x = max(max_x, x)
            if y <= 200:
                max_x_y200 = max(max_x_y200, x)
            if y <= 170:
                max_x_y170 = max(max_x_y170, x)
            if y <= 150:
                max_x_y150 = max(max_x_y150, x)
            if y <= 200 and x >= TOP_X and not hit_top:
                hit_top = True
                first_top = snap(st)
                path_hi.append({"i": i, **snap(st)})
                break
            if y <= 220 and i % 5 == 0:
                path_hi.append({"i": i, "x": x, "y": y, "p": int(st.pose)})
        return {
            "name": name,
            "pin": pin.name,
            "start": start,
            "min_y": min_y,
            "max_x": max_x,
            "max_x_y200": max_x_y200,
            "max_x_y170": max_x_y170,
            "max_x_y150": max_x_y150,
            "peak": list(peak),
            "top": hit_top,
            "first_top": first_top,
            "end": snap(st),
            "path_hi": path_hi[-12:],
            "frames": min(len(script), cap, self.frame),
        }


def rep(n: int, *btns: str) -> list[tuple[str, ...]]:
    return [btns] * n


def seq(*chunks: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for c in chunks:
        out.extend(c)
    return out


def pump_lr(n: int) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for i in range(n):
        out.append(("RIGHT", "B", "L" if i % 2 == 0 else "R"))
    return out


def pump_angle(n: int) -> list[tuple[str, ...]]:
    """Angle-up / release arm pump (X = aim-up often mapped; R shoulder)."""
    out: list[tuple[str, ...]] = []
    for i in range(n):
        if i % 2 == 0:
            out.append(("RIGHT", "B", "R"))
        else:
            out.append(("RIGHT", "B"))
    return out


def human_launch(
    run_f: int = 21,
    spin_f: int = 83,
    *,
    run: str = "plain",
    clear: bool = False,
) -> list[tuple[str, ...]]:
    chunks: list[list[tuple[str, ...]]] = []
    if clear:
        # Shoot while walking right a bit then reset left to door edge.
        chunks.append(rep(6, "RIGHT", "Y"))
        chunks.append(rep(4, "RIGHT", "B", "Y"))
        chunks.append(rep(14, "LEFT", "B"))
        chunks.append(rep(3, ))
    if run == "plain":
        chunks.append(rep(run_f, "RIGHT", "B"))
    elif run == "pump_lr":
        chunks.append(pump_lr(run_f))
    elif run == "pump_angle":
        chunks.append(pump_angle(run_f))
    elif run == "long_left_first":
        # Walk to absolute left edge then max runway dash.
        chunks.append(rep(8, "LEFT", "B"))
        chunks.append(rep(2, ))
        chunks.append(rep(run_f, "RIGHT", "B"))
    else:
        raise ValueError(run)
    chunks.append(rep(spin_f, "RIGHT", "B", "A"))
    return seq(*chunks)


def human_wj(
    la: int = 23,
    ra: int = 16,
    a_mid: int = 6,
) -> list[tuple[str, ...]]:
    return seq(
        rep(4, "B", "A"),
        rep(1, "B"),
        rep(2, ),
        rep(2, "LEFT"),
        rep(la, "LEFT", "A"),
        rep(a_mid, "A"),
        rep(ra, "RIGHT", "A"),
    )


def double_wj(
    la1: int,
    ra1: int,
    la2: int,
    ra2: int = 12,
    la3: int = 0,
    *,
    period_follow: int = 60,
    period: int = 6,
) -> list[tuple[str, ...]]:
    chunks = [
        rep(2, "LEFT"),
        rep(la1, "LEFT", "A"),
        rep(3, "A"),
        rep(ra1, "RIGHT", "A"),
        rep(2, "RIGHT"),
        rep(la2, "LEFT", "A"),
        rep(3, "A"),
        rep(ra2, "RIGHT", "B", "A"),
    ]
    if la3:
        chunks.append(rep(la3, "LEFT", "A"))
        chunks.append(rep(20, "RIGHT", "B", "A"))
    follow: list[tuple[str, ...]] = []
    for i in range(period_follow):
        ph = i % period
        if ph < 1:
            follow.append(("RIGHT", "B"))
        elif ph < 3:
            follow.append(("LEFT", "A"))
        else:
            follow.append(("RIGHT", "B", "A"))
    chunks.append(follow)
    return seq(*chunks)


def peak_lip_scripts() -> list[tuple[str, list[tuple[str, ...]]]]:
    """From human peak pin — try to clear ceiling lip rightward."""
    out: list[tuple[str, list[tuple[str, ...]]]] = []
    # Baseline thrash
    thrash: list[tuple[str, ...]] = []
    for _ in range(20):
        thrash.extend([("RIGHT", "B")] * 2)
        thrash.extend([("LEFT", "A")] * 2)
        thrash.extend([("RIGHT", "B", "A")] * 4)
    out.append(("peak_period8", thrash))
    # Immediate second WJ then push right
    for la, ra, spin in (
        (14, 10, 20),
        (18, 8, 16),
        (22, 12, 24),
        (10, 6, 12),
        (16, 4, 30),
    ):
        out.append(
            (
                f"peak_wj_L{la}_R{ra}_s{spin}",
                seq(
                    rep(la, "LEFT", "A"),
                    rep(2, "A"),
                    rep(ra, "RIGHT", "A"),
                    rep(spin, "RIGHT", "B", "A"),
                    rep(12, "LEFT", "A"),
                    rep(24, "RIGHT", "B", "A"),
                    rep(12, "LEFT", "A"),
                    rep(30, "RIGHT", "B", "A"),
                ),
            )
        )
    # Morph under ceiling crawl
    out.append(
        (
            "peak_morph_right",
            seq(
                rep(3, "DOWN"),
                rep(2, "A"),
                rep(40, "RIGHT", "B"),
                rep(4, "UP"),
                rep(40, "RIGHT", "B", "A"),
            ),
        )
    )
    # Unspin land + walk right (if any ledge)
    out.append(
        (
            "peak_unspin_walk",
            seq(
                rep(8, "UP"),
                rep(4, ),
                rep(30, "RIGHT", "B"),
                rep(8, "A"),
                rep(40, "RIGHT", "B", "A"),
            ),
        )
    )
    # Charge shot bounce (Y beam) while WJ
    out.append(
        (
            "peak_charge_wj",
            seq(
                rep(20, "Y"),
                rep(16, "LEFT", "A"),
                rep(10, "RIGHT", "A", "Y"),
                rep(20, "RIGHT", "B", "A"),
                rep(14, "LEFT", "A"),
                rep(30, "RIGHT", "B", "A"),
            ),
        )
    )
    # Diagonal aim + WJ
    out.append(
        (
            "peak_upright_wj",
            seq(
                rep(18, "LEFT", "A"),
                rep(20, "RIGHT", "UP", "A"),
                rep(14, "LEFT", "A"),
                rep(30, "RIGHT", "B", "A"),
            ),
        )
    )
    return out


def ceiling_scripts() -> list[tuple[str, list[tuple[str, ...]]]]:
    out: list[tuple[str, list[tuple[str, ...]]]] = []
    for name, script in (
        (
            "ceil_wj_double",
            seq(
                rep(16, "LEFT", "A"),
                rep(8, "RIGHT", "A"),
                rep(14, "LEFT", "A"),
                rep(30, "RIGHT", "B", "A"),
            ),
        ),
        (
            "ceil_morph_roll",
            seq(rep(2, "DOWN"), rep(2, "A"), rep(50, "RIGHT"), rep(4, "UP"), rep(40, "RIGHT", "B", "A")),
        ),
        (
            "ceil_drop_right_shelf",
            seq(
                rep(6, "RIGHT"),
                rep(40, "RIGHT", "B"),
                rep(8, "RIGHT", "B", "A"),
                rep(40, "RIGHT", "B"),
            ),
        ),
        (
            "ceil_stick_wj",
            seq(
                rep(8, "LEFT"),
                rep(20, "LEFT", "A"),
                rep(4, "A"),
                rep(6, "RIGHT", "A"),
                rep(18, "LEFT", "A"),
                rep(40, "RIGHT", "B", "A"),
            ),
        ),
    ):
        out.append((name, script))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    runway = SCRATCH / "bubble_human_runway.state"
    peak = SCRATCH / "bubble_human_peak.state"
    ceiling = SCRATCH / "bubble_human_ceiling.state"
    for p in (runway, peak, ceiling):
        if not p.exists():
            print(f"missing pin {p}", file=sys.stderr)
            return 2

    probe = PinProbe()
    results: list[dict[str, Any]] = []

    def add(r: dict[str, Any]) -> None:
        results.append(r)
        tag = "TOP" if r["top"] else "—"
        print(
            f"{r['name']:48} min_y={r['min_y']:4} max_x={r['max_x']:4} "
            f"x@200={r['max_x_y200']:3} x@170={r['max_x_y170']:3} "
            f"x@150={r['max_x_y150']:3} [{tag}] peak={r['peak']} "
            f"end=({r['end']['x']},{r['end']['y']})p{r['end']['p']}"
        )

    try:
        # Pin sanity
        for pin, label in (
            (runway, "runway"),
            (peak, "peak"),
            (ceiling, "ceiling"),
        ):
            st = probe.boot(pin)
            print(f"PIN {label}: {snap(st)}")

        # --- A. Exact human open-loop from runway pin ---
        add(
            probe.run_script(
                "runway_exact_human",
                runway,
                seq(human_launch(21, 83), human_wj(23, 16, 6), rep(40, "RIGHT", "B", "A")),
            )
        )

        # --- B. Max runway / arm pump / clear from runway ---
        for run_f in (18, 21, 24, 28, 32, 36, 40):
            for spin_f in (75, 83, 90, 100):
                for run_kind in ("plain", "pump_lr", "pump_angle", "long_left_first"):
                    if run_kind != "plain" and spin_f not in (83, 90):
                        continue
                    if run_kind == "long_left_first" and run_f not in (28, 36, 40):
                        continue
                    name = f"run_{run_kind}_r{run_f}_s{spin_f}_humWJ"
                    add(
                        probe.run_script(
                            name,
                            runway,
                            seq(
                                human_launch(run_f, spin_f, run=run_kind),
                                human_wj(23, 16, 6),
                                rep(30, "RIGHT", "B", "A"),
                            ),
                        )
                    )

        # clear bug + best run lengths
        for run_f in (24, 28, 32, 36):
            for run_kind in ("plain", "pump_lr"):
                name = f"clear_{run_kind}_r{run_f}_s83"
                add(
                    probe.run_script(
                        name,
                        runway,
                        seq(
                            human_launch(run_f, 83, run=run_kind, clear=True),
                            human_wj(23, 16, 6),
                            rep(30, "RIGHT", "B", "A"),
                        ),
                    )
                )

        # --- C. Double WJ variants from runway (after best launch) ---
        for run_f, spin_f, run_kind in (
            (21, 83, "plain"),
            (28, 83, "plain"),
            (32, 83, "pump_lr"),
            (36, 90, "pump_lr"),
            (28, 83, "long_left_first"),
            (36, 83, "long_left_first"),
        ):
            for la1, ra1, la2, ra2, la3 in (
                (23, 16, 18, 12, 14),
                (20, 10, 20, 10, 16),
                (18, 8, 22, 8, 18),
                (25, 12, 16, 14, 12),
                (22, 6, 24, 6, 20),
                (16, 12, 16, 12, 16),
            ):
                name = (
                    f"dwj_{run_kind}_r{run_f}_s{spin_f}_"
                    f"L{la1}R{ra1}L{la2}R{ra2}L{la3}"
                )
                add(
                    probe.run_script(
                        name,
                        runway,
                        seq(
                            human_launch(run_f, spin_f, run=run_kind),
                            double_wj(la1, ra1, la2, ra2, la3),
                        ),
                    )
                )

        # --- D. Peak pin lip clear ---
        for name, script in peak_lip_scripts():
            add(probe.run_script(name, peak, script))

        # --- E. Ceiling pin ---
        for name, script in ceiling_scripts():
            add(probe.run_script(name, ceiling, script))

        # --- F. Extra second WJ right after human peak sequence mid-air ---
        # Replay human to peak-ish then extra WJ
        add(
            probe.run_script(
                "runway_human_plus_extra_wj",
                runway,
                seq(
                    human_launch(21, 83),
                    human_wj(23, 16, 6),
                    rep(14, "LEFT", "A"),
                    rep(4, "A"),
                    rep(12, "RIGHT", "A"),
                    rep(16, "LEFT", "A"),
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
            r["max_x_y200"] >= 300,
            r["max_x_y170"] >= 280,
            -r["min_y"],
            r["max_x_y200"],
            r["max_x_y170"],
            r["max_x_y150"],
            r["max_x"],
        ),
        reverse=True,
    )
    summary = {
        "n": len(results),
        "best15": [
            {
                k: r[k]
                for k in (
                    "name",
                    "min_y",
                    "max_x",
                    "max_x_y200",
                    "max_x_y170",
                    "max_x_y150",
                    "top",
                    "peak",
                    "end",
                )
            }
            for r in ranked[:15]
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print("\n=== BEST 15 ===")
    for b in summary["best15"]:
        print(b)
    print(f"\nwrote {args.output} n={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
