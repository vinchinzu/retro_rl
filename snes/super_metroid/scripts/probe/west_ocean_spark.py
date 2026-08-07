#!/usr/bin/env python3
"""West Ocean edge-turn-hop shinespark probe (VOD recipe).

Recipe (from screenshots + human callout):
  run to water edge → store → turn back a few steps → jump up → spark right
  a few tiles up into the door band.

```bash
# Headless pure (free-places onto dry spit from Moat handoff)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure

# Without free-place (source must already be on dry spit)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure --no-place

# Live watch
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py watch
```

Default source: ``scratch/post_moat_west_ocean_spark.state`` (post-Moat spark).
Door currently reached: ``0xC98E`` Bowling Alley mid-right (not green WS).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
_SNES = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.env import make_env, read_state_bytes  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import place_samus, save_dev_state  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR  # noqa: E402
from super_metroid.ram import parse_env_state, write_wram_u16  # noqa: E402
from super_metroid.routes.kpdr import west_ocean as wo  # noqa: E402
from super_metroid.routes.skills import shinespark as spark  # noqa: E402

SCRATCH = INTEGRATION_DIR / "scratch"
DEFAULT_SOURCE = SCRATCH / "post_moat_west_ocean_spark.state"
DEFAULT_OUT = SCRATCH / "post_west_ocean_door_spark.state"
DEBUG = Path("snes/super_metroid/debug/west_ocean_spark")


class _Sess:
    def __init__(self, env: Any, assist: UnlimitedResourcesAssist | None):
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        st = parse_env_state(self.env, mode="nav")
        if self.assist is not None:
            try:
                self.assist.apply(self.env.data, st)
            except Exception:  # noqa: BLE001
                try:
                    self.assist.apply(self.env, st)
                except Exception:  # noqa: BLE001
                    pass
        self.state = parse_env_state(self.env, mode="nav")
        return self.state


def boot(
    source: Path,
    *,
    place_spit: bool = True,
    place_xy: tuple[int, int] = wo.SPIT_PLACE_XY,
    assist: bool = True,
) -> tuple[Any, _Sess]:
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    a = UnlimitedResourcesAssist() if assist else None
    env.reset()
    env.em.set_state(read_state_bytes(source))
    sess = _Sess(env, a)
    for _ in range(12):
        sess.step(idle_action())
    if place_spit and sess.state.room_id == wo.ROOM_WEST_OCEAN:
        # Lower-left water handoff cannot charge — bootstrap dry spit.
        if sess.state.samus_y > 900 or sess.state.samus_x < 200:
            place_samus(env, place_xy[0], place_xy[1])
            write_wram_u16(env, 0x18AA, 0)
            write_wram_u16(env, 0x18A8, 0x400)
            for i in range(100):
                sess.step(idle_action())
                if sess.state.velocity_y == 0 and i > 15:
                    break
    return env, sess


def cmd_pure(args: argparse.Namespace) -> int:
    source = Path(args.source or DEFAULT_SOURCE)
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2
    env, sess = boot(
        source,
        place_spit=not args.no_place,
        place_xy=(args.place_x, args.place_y),
        assist=not args.no_assist,
    )
    boot_snap = spark.spark_snapshot(env, 0)
    print(
        f"boot room=0x{sess.state.room_id:04X} xy=({sess.state.samus_x},{sess.state.samus_y}) "
        f"pose={sess.state.pose} place={not args.no_place}"
    )
    try:
        st = wo.play_west_ocean_edge_spark(
            sess,
            back_frames=args.back,
            hop_frames=args.hop,
            aim_buttons=tuple(args.aim.replace("+", ",").split(",")),
        )
        print(
            f"GREEN room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) "
            f"pose={st.pose} frames={sess.frame}"
        )
        out = Path(args.out or DEFAULT_OUT)
        save_dev_state(env, out)
        print(f"saved {out}")
        DEBUG.mkdir(parents=True, exist_ok=True)
        (DEBUG / "pure.json").write_text(
            json.dumps(
                {
                    "boot": boot_snap,
                    "final": spark.spark_snapshot(env, sess.frame),
                    "frames": sess.frame,
                    "params": {
                        "back": args.back,
                        "hop": args.hop,
                        "aim": args.aim,
                        "place": not args.no_place,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"RED {exc}")
        print(f"pin {spark.spark_snapshot(env, sess.frame)}")
        return 1
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pure", help="Headless edge-turn-hop spark once")
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--back", type=int, default=wo.DEFAULT_BACK_FRAMES)
    p.add_argument("--hop", type=int, default=wo.DEFAULT_HOP_FRAMES)
    p.add_argument("--aim", type=str, default="RIGHT", help="e.g. RIGHT or RIGHT+UP")
    p.add_argument("--no-place", action="store_true", help="Do not free-place spit")
    p.add_argument("--place-x", type=int, default=wo.SPIT_PLACE_XY[0])
    p.add_argument("--place-y", type=int, default=wo.SPIT_PLACE_XY[1])
    p.add_argument("--no-assist", action="store_true")
    p.set_defaults(func=cmd_pure)

    # alias
    p2 = sub.add_parser("hop", help="Alias of pure")
    p2.add_argument("--source", type=Path, default=None)
    p2.add_argument("--out", type=Path, default=None)
    p2.add_argument("--back", type=int, default=wo.DEFAULT_BACK_FRAMES)
    p2.add_argument("--hop", type=int, default=wo.DEFAULT_HOP_FRAMES)
    p2.add_argument("--aim", type=str, default="RIGHT")
    p2.add_argument("--no-place", action="store_true")
    p2.add_argument("--place-x", type=int, default=wo.SPIT_PLACE_XY[0])
    p2.add_argument("--place-y", type=int, default=wo.SPIT_PLACE_XY[1])
    p2.add_argument("--no-assist", action="store_true")
    p2.set_defaults(func=cmd_pure)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
