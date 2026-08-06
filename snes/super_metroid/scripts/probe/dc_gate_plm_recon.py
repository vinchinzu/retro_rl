#!/usr/bin/env python3
"""Scratch recon: Double Chamber blue-gate seat / shot logging (rr-dbu.10).

Does **not** edit product ``k4_wave.py``. Boots the pure Single→Double pin,
optionally places near the left-of-gate Kamer seat, fires controlled shot
patterns, and logs max x / pose / ammo / trusted door-nav fields.

Open proof for the gate is walk past bars (x≳420 solid / x≳480 platform), not
PLM WRAM (blocked in red_diag).

```bash
# Place at human seat, assist off (ammo drain truth), beam peak band
uv run python snes/super_metroid/scripts/probe/dc_gate_plm_recon.py \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --mode place-seat --no-assist --weapon 0

# Natural hop toward seat (no place), assist on (heat survivability)
uv run python snes/super_metroid/scripts/probe/dc_gate_plm_recon.py --mode hop-seat

# Idle log only after place (Kamer cycle watch)
uv run python snes/super_metroid/scripts/probe/dc_gate_plm_recon.py --mode place-seat --shots none
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import (  # noqa: E402
    boot_from_state,
    make_dev_env,
    place_samus,
    select_weapon as dev_select_weapon,
)
from super_metroid.paths import DEBUG_DIR, SCRATCH_STATE_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.scripts.probe.red_diag import (  # noqa: E402
    build_door_plm_snapshot,
    display_path,
)

ROOM_DOUBLE = 0xADAD
DEFAULT_SOURCE = SCRATCH_STATE_DIR / "post_single_to_double_chamber_pure.state"
OUT_DIR = DEBUG_DIR / "wave_recon" / "dc_gate_plm_recon"

# Human seat / open geometry (see SM-K4.10-GATE-HITBOX-recon.md).
SEAT_X = 378
SEAT_Y = 139
GATE_HARD_X = 411
PAST_GATE_X = 420
PAST_PLATFORM_X = 480
PEAK_Y = (104, 111)
STANDING = frozenset({1, 2, 5, 6, 7, 8})


def _snap(st: Any, frame: int, *, reason: str = "") -> dict[str, Any]:
    return {
        "frame": frame,
        "room": f"0x{int(st.room_id):04X}",
        "x": int(st.samus_x),
        "y": int(st.samus_y),
        "pose": int(st.pose),
        "vx": int(st.velocity_x),
        "vy": int(st.velocity_y),
        "selected": int(st.selected_item),
        "missiles": int(st.missiles),
        "max_missiles": int(st.max_missiles),
        "supers": int(st.super_missiles),
        "health": int(st.health),
        "door_transition": int(st.door_transition),
        "game_state": int(st.game_state),
        "beams": f"0x{int(st.collected_beams):04X}",
        "reason": reason,
    }


class Sess:
    """Minimal controller session; assist optional for ammo-drain truth."""

    def __init__(self, env: Any, assist: UnlimitedResourcesAssist | None) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.log: list[dict[str, Any]] = []
        self.max_x_upper = int(self.state.samus_x)
        self.max_x_any = int(self.state.samus_x)

    def step(self, action: Any, reason: str = "", *, record: bool = True) -> Any:
        self.env.step(action)
        self.frame += 1
        if self.assist is not None:
            st0 = parse_env_state(self.env, frame=self.frame, mode="nav")
            try:
                self.assist.apply(self.env.data, st0)
            except Exception:  # noqa: BLE001 — assist surface varies
                try:
                    self.assist.apply(self.env, st0)
                except Exception:  # noqa: BLE001
                    pass
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        st = self.state
        if int(st.room_id) == ROOM_DOUBLE:
            x = int(st.samus_x)
            self.max_x_any = max(self.max_x_any, x)
            if int(st.samus_y) < 220 and x < 2000:
                self.max_x_upper = max(self.max_x_upper, x)
        if record:
            self.log.append(_snap(st, self.frame, reason=reason))
        return st

    def hold(self, n: int, *btns: str, reason: str = "") -> Any:
        act = buttons(*btns) if btns else idle_action()
        for _ in range(n):
            self.step(act, reason=reason)
        return self.state


def boot(source: Path, *, assist: bool) -> tuple[Any, Sess]:
    env = make_dev_env()
    boot_from_state(env, source, settle_frames=8)
    a = UnlimitedResourcesAssist() if assist else None
    sess = Sess(env, a)
    for _ in range(4):
        sess.step(idle_action(), reason="boot", record=False)
    sess.frame = 0
    sess.log.clear()
    sess.state = parse_env_state(env, mode="nav")
    sess.max_x_upper = int(sess.state.samus_x)
    sess.max_x_any = int(sess.state.samus_x)
    return env, sess


def hop_toward_seat(sess: Sess, *, budget: int = 500) -> None:
    """Crude upper-path hop; not product geometry — recon only."""
    for frame in range(budget):
        st = sess.state
        if int(st.room_id) != ROOM_DOUBLE:
            return
        if int(st.samus_y) < 200 and SEAT_X - 15 <= int(st.samus_x) <= SEAT_X + 20:
            if int(st.velocity_y) == 0:
                return
        if int(st.samus_y) > 360 and int(st.velocity_y) == 0:
            return
        if int(st.pose) in (137, 138):
            sess.hold(8, "UP", reason="unmorph")
            continue
        phase = frame % 28
        if phase < 14:
            sess.hold(1, "RIGHT", "B", "A", reason="hop_spin")
        elif phase < 22:
            sess.hold(1, "RIGHT", "B", reason="hop_run")
        else:
            sess.hold(1, "RIGHT", reason="hop_walk")


def wait_kamer_top(sess: Sess, *, budget: int = 400) -> bool:
    """Wait for high Kamer seat y≤145 near seat x (cycle ~200f half)."""
    for _ in range(budget):
        st = sess.state
        if int(st.room_id) != ROOM_DOUBLE:
            return False
        if int(st.samus_x) >= PAST_PLATFORM_X and int(st.samus_y) < 220:
            return True
        if (
            int(st.velocity_y) == 0
            and int(st.samus_y) <= 145
            and SEAT_X - 20 <= int(st.samus_x) <= SEAT_X + 25
        ):
            return True
        if int(st.samus_x) < SEAT_X - 5:
            sess.hold(1, "RIGHT", reason="kamer_r")
        elif int(st.samus_x) > SEAT_X + 15:
            sess.hold(1, "LEFT", reason="kamer_l")
        else:
            sess.hold(1, reason="kamer_wait")
    return False


def stand_settle(sess: Sess, *, frames: int = 20) -> None:
    for _ in range(frames):
        st = sess.hold(1, reason="stand_settle")
        if (
            int(st.velocity_y) == 0
            and int(st.pose) in STANDING
            and int(st.samus_y) <= 150
        ):
            break


def fire_peak_r_angle(sess: Sess, *, weapon: int) -> dict[str, Any]:
    """Controlled peak volley: stand R+X, jump, fire X+R only in y band."""
    dev_select_weapon(sess.env, weapon)
    sess.hold(2, reason="sel_settle")
    sess.hold(3, "RIGHT", reason="face")
    stand_settle(sess)
    pre = _snap(sess.state, sess.frame, reason="pre_volley")

    sess.hold(6, "R", reason="angle_hold")
    for _ in range(2):
        sess.hold(4, "X", "R", reason="stand_shot")
        sess.hold(5, "R", reason="stand_wait")

    sess.hold(2, "A", "R", reason="jump")
    fired_peak = False
    peak_rows: list[dict[str, Any]] = []
    for _ in range(32):
        st = sess.hold(1, "A", "R", reason="rise")
        y = int(st.samus_y)
        if PEAK_Y[0] <= y <= PEAK_Y[1] and not fired_peak:
            for _ in range(6):
                st = sess.hold(1, "X", "R", reason="peak_shot")
                peak_rows.append(_snap(st, sess.frame, reason="peak_shot"))
            fired_peak = True
            break
        if y < PEAK_Y[0] and not fired_peak:
            # Overshot band — still fire once.
            st = sess.hold(4, "X", "R", reason="peak_shot_late")
            peak_rows.append(_snap(st, sess.frame, reason="peak_shot_late"))
            fired_peak = True
            break

    for _ in range(20):
        st = sess.hold(1, "X", reason="fall_x")
        if int(st.velocity_y) == 0 and int(st.samus_y) > 130:
            break

    sess.hold(20, reason="open_fuse")
    return {
        "pre": pre,
        "fired_peak": fired_peak,
        "peak_rows": peak_rows,
        "post": _snap(sess.state, sess.frame, reason="post_volley"),
    }


def walk_probe(sess: Sess, *, frames: int = 60) -> dict[str, Any]:
    """Walk-only probe into bars (no spin) to test open."""
    start = _snap(sess.state, sess.frame, reason="walk_start")
    for _ in range(frames):
        st = sess.state
        if int(st.samus_x) >= PAST_PLATFORM_X and int(st.samus_y) < 220:
            break
        if int(st.samus_y) > 300:
            break
        if int(st.velocity_y) == 0 and int(st.samus_x) >= GATE_HARD_X - 5:
            # Nudge back if stuck on closed face.
            if int(st.samus_x) <= GATE_HARD_X + 2:
                sess.hold(1, "RIGHT", reason="walk_face")
            else:
                sess.hold(1, "RIGHT", reason="walk_past")
        else:
            sess.hold(1, "RIGHT", reason="walk")
    end = _snap(sess.state, sess.frame, reason="walk_end")
    return {
        "start": start,
        "end": end,
        "past_bars": int(sess.state.samus_x) > GATE_HARD_X + 2
        and int(sess.state.samus_y) < 220,
        "past_platform": int(sess.state.samus_x) >= PAST_PLATFORM_X
        and int(sess.state.samus_y) < 220,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument(
        "--mode",
        choices=("place-seat", "hop-seat", "pin-only"),
        default="place-seat",
    )
    ap.add_argument(
        "--shots",
        choices=("peak-band", "none"),
        default="peak-band",
        help="peak-band = human-like R-angle peak in y 104–111",
    )
    ap.add_argument(
        "--weapon",
        type=int,
        default=0,
        help="0 beam, 1 missiles, 2 supers (default beam = human final volley)",
    )
    ap.add_argument("--no-assist", action="store_true")
    ap.add_argument("--place-x", type=int, default=SEAT_X)
    ap.add_argument("--place-y", type=int, default=SEAT_Y)
    ap.add_argument("--volleys", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    source = args.source
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2

    env, sess = boot(source, assist=not args.no_assist)
    report: dict[str, Any] = {
        "kind": "dc_gate_plm_recon",
        "source": display_path(source),
        "mode": args.mode,
        "shots": args.shots,
        "weapon": args.weapon,
        "assist": not args.no_assist,
        "room_expected": f"0x{ROOM_DOUBLE:04X}",
        "geometry": {
            "seat": [SEAT_X, SEAT_Y],
            "hardStopX": GATE_HARD_X,
            "pastBarsX": PAST_GATE_X,
            "pastPlatformX": PAST_PLATFORM_X,
            "peakY": list(PEAK_Y),
            "switchCandidate": {
                "x": [398, 416],
                "y": [80, 100],
                "side": "top",
                "projectile": "R-angle up-right from left seat; or UP from below",
            },
        },
        "boot": _snap(sess.state, 0, reason="boot"),
        "volleys": [],
        "walk": None,
        "nonClaims": [
            "Not product controller / not k4_wave edit",
            "PLM open WRAM blocked (red_diag)",
            "No Wave beam claim",
            "No STATUS promote",
        ],
    }

    if int(sess.state.room_id) != ROOM_DOUBLE:
        report["error"] = (
            f"not in Double Chamber: room=0x{int(sess.state.room_id):04X}"
        )
        print(json.dumps(report, indent=2))
        return 1

    if args.mode == "place-seat":
        place_samus(env, args.place_x, args.place_y)
        for _ in range(12):
            sess.step(idle_action(), reason="place_settle")
        report["after_place"] = _snap(sess.state, sess.frame, reason="after_place")
        wait_kamer_top(sess)
        report["after_kamer"] = _snap(sess.state, sess.frame, reason="after_kamer")
    elif args.mode == "hop-seat":
        hop_toward_seat(sess)
        wait_kamer_top(sess)
        report["after_hop"] = _snap(sess.state, sess.frame, reason="after_hop")
    else:
        report["pin_only"] = True

    if args.shots == "peak-band":
        for i in range(max(1, args.volleys)):
            wait_kamer_top(sess)
            stand_settle(sess)
            vol = fire_peak_r_angle(sess, weapon=args.weapon)
            vol["index"] = i
            report["volleys"].append(vol)
            report["walk"] = walk_probe(sess)
            if report["walk"]["past_platform"]:
                break
    else:
        # Kamer cycle sample without shots.
        for _ in range(120):
            sess.hold(1, reason="idle_cycle")

    report["walk"] = report["walk"] or walk_probe(sess)
    report["max_x_upper"] = sess.max_x_upper
    report["max_x_any"] = sess.max_x_any
    report["final"] = _snap(sess.state, sess.frame, reason="final")
    report["gate_open_heuristic"] = bool(
        report["walk"]["past_platform"]
        or (
            report["walk"]["past_bars"]
            and int(sess.state.samus_x) >= PAST_GATE_X
            and int(sess.state.samus_y) < 200
            and int(sess.state.velocity_y) == 0
        )
    )
    report["ammo_delta"] = {
        "missiles": report["boot"]["missiles"] - report["final"]["missiles"],
        "supers": report["boot"]["supers"] - report["final"]["supers"],
        "assist": not args.no_assist,
    }
    report["door_plm_snapshot"] = build_door_plm_snapshot(
        env,
        sess.state,
        error="" if report["gate_open_heuristic"] else "gate not cleared",
        segment="dc_gate_plm_recon",
        source=display_path(source),
        frames=sess.frame,
        extra={
            "max_x_upper": sess.max_x_upper,
            "gate_open_heuristic": report["gate_open_heuristic"],
        },
    )

    out = args.out or (OUT_DIR / "last_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Compact log sample (every 5th + endpoints) to keep artifact small.
    sample = [sess.log[i] for i in range(0, len(sess.log), 5)]
    if sess.log and sess.log[-1] not in sample:
        sample.append(sess.log[-1])
    report["log_sample"] = sample
    report["log_len"] = len(sess.log)
    report["out"] = display_path(out)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Human-readable one-liner for residual paste.
    fin = report["final"]
    print(
        f"dc_gate_plm_recon open={report['gate_open_heuristic']} "
        f"max_x_upper={sess.max_x_upper} "
        f"final=({fin['x']},{fin['y']}) p={fin['pose']} "
        f"sel={fin['selected']} mis={fin['missiles']} "
        f"frames={sess.frame} assist={not args.no_assist} "
        f"out={display_path(out)}"
    )
    return 0 if report["gate_open_heuristic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
