#!/usr/bin/env python3
"""West Ocean shinespark probe — product over-ocean → WS + spit edge bowling.

**Product (green Super WS ``0xCA08``):** ocean-floor stutter charge → spark →
Super open from Moat handoff ``~(49,1163)``.

**Practice (Bowling ``0xC98E``):** spit edge store→hop→spark (VOD recipe).

```bash
# Product pure — natural Moat handoff → 0xCA08 (no free-place)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure-ws
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure-ws --charge-mode short

# Headed product watch (prefer for attempt review)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py watch-ws
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py watch-ws --charge-mode stutter

# Spit-edge bowling pure (free-place spit; wrong door for Phantoon)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure --no-place
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py watch

# Short-charge distance only (spit place)
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py short-charge --mode stutter
```

Default source: ``scratch/post_moat_west_ocean_spark.state``.
Product out pin: ``scratch/post_west_ocean_ws_spark.state``.
See ``docs/tasks/SHINE_PRACTICE.md``.
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
from retro_harness.play_session import PlaySession  # noqa: E402
from retro_harness.runtime import step_env  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import place_samus, save_dev_state  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR  # noqa: E402
from super_metroid.ram import parse_env_state, write_wram_u16  # noqa: E402
from super_metroid.routes.kpdr import west_ocean as wo  # noqa: E402
from super_metroid.routes.skills import shinespark as spark  # noqa: E402

SCRATCH = INTEGRATION_DIR / "scratch"
DEFAULT_SOURCE = SCRATCH / "post_moat_west_ocean_spark.state"
DEFAULT_OUT = SCRATCH / "post_west_ocean_door_spark.state"  # bowling edge pure
DEFAULT_WS_OUT = SCRATCH / "post_west_ocean_ws_spark.state"  # product 0xCA08
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


def cmd_short_charge(args: argparse.Namespace) -> int:
    """Measure short-charge distance on the dry spit (charge only)."""
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
        f"mode={args.mode} store_on_last={args.store_on_last}"
    )
    try:
        charge = spark.charge_until_boost(
            sess,
            "RIGHT",
            budget=400,
            mode=args.mode,
            store_on_last=bool(args.store_on_last),
            label=f"wo_{args.mode}",
        )
        final = spark.spark_snapshot(env, sess.frame)
        ok = bool(charge.get("ok"))
        flag = "GREEN" if ok else "RED"
        print(
            f"{flag} mode={args.mode} ok={ok} frames={charge.get('frames')} "
            f"delta_x={charge.get('delta_x')} start_x={charge.get('start_x')} "
            f"end_x={charge.get('end_x')} echoes={final.get('speed_echoes')} "
            f"timer={final.get('spark_timer')} dash={charge.get('dash_frames')}"
        )
        DEBUG.mkdir(parents=True, exist_ok=True)
        out_path = DEBUG / f"short_charge_{args.mode}.json"
        out_path.write_text(
            json.dumps(
                {
                    "boot": boot_snap,
                    "charge": charge,
                    "final": final,
                    "params": {
                        "mode": args.mode,
                        "store_on_last": args.store_on_last,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {out_path}")
        return 0 if ok else 1
    finally:
        env.close()


def _run_watch(
    args: argparse.Namespace,
    *,
    mode: str,
) -> int:
    """Headed PlaySession for edge (bowling) or over-ocean (WS) spark.

    Keys: ``[`` ``]`` speed · TAB turbo · ESC quit.
    ``mode``: ``edge`` | ``ws``.
    """
    source = Path(args.source or DEFAULT_SOURCE)
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedResourcesAssist() if not args.no_assist else None
    state_bytes = read_state_bytes(source)
    place_spit = mode == "edge" and not getattr(args, "no_place", False)
    place_xy = (
        getattr(args, "place_x", wo.SPIT_PLACE_XY[0]),
        getattr(args, "place_y", wo.SPIT_PLACE_XY[1]),
    )
    charge_mode = args.charge_mode
    back = getattr(args, "back", wo.DEFAULT_BACK_FRAMES)
    hop = args.hop
    aim = tuple(getattr(args, "aim", "RIGHT").replace("+", ",").split(","))
    default_out = DEFAULT_WS_OUT if mode == "ws" else DEFAULT_OUT
    title = (
        "West Ocean over-ocean → WS 0xCA08"
        if mode == "ws"
        else "West Ocean edge spark → bowling 0xC98E"
    )
    target_room = wo.ROOM_WS_ENTRANCE if mode == "ws" else wo.ROOM_BOWLING

    class _Watch:
        def __init__(self) -> None:
            self.env = env
            self.assist = assist
            self.frame = 0
            self.state = parse_env_state(env, mode="nav")
            self.done = False
            self.success = False
            self.error: str | None = None
            self.phase = "boot"
            self.detail = (
                "over-ocean stutter → Super 0xCA08"
                if mode == "ws"
                else "edge spark → bowling 0xC98E"
            )

        def step(self, action, reason: str = ""):
            del reason
            env.step(action)
            self.frame += 1
            st = parse_env_state(env, mode="nav")
            if self.assist is not None:
                try:
                    self.assist.apply(env.data, st)
                except Exception:  # noqa: BLE001
                    try:
                        self.assist.apply(env, st)
                    except Exception:  # noqa: BLE001
                        pass
            self.state = parse_env_state(env, mode="nav")
            return self.state

    watch = _Watch()
    ran = {"started": False}

    def bot(_env, _info):
        import numpy as np

        if watch.done:
            return np.zeros(12, dtype=np.int8)
        if not ran["started"]:
            ran["started"] = True
            watch.phase = "spark"
            if mode == "ws":
                watch.detail = f"over-ocean charge={charge_mode} hop={hop}"
            else:
                watch.detail = f"charge={charge_mode} back={back} hop={hop} aim={aim}"
            try:
                if mode == "ws":
                    st = wo.play_west_ocean_over_ocean_spark(
                        watch,
                        hop_frames=hop,
                        charge_mode=charge_mode,
                    )
                else:
                    st = wo.play_west_ocean_edge_spark(
                        watch,
                        back_frames=back,
                        hop_frames=hop,
                        aim_buttons=aim,
                        charge_mode=charge_mode,
                    )
                watch.success = st.room_id == target_room or (
                    mode == "edge" and st.room_id != wo.ROOM_WEST_OCEAN
                )
                if mode == "ws":
                    watch.success = (
                        st.room_id == wo.ROOM_WS_ENTRANCE
                        and st.game_state == 8
                        and st.door_transition == 0
                    ) or st.room_id == wo.ROOM_WS_ENTRANCE
                watch.phase = "done"
                watch.detail = (
                    f"room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) "
                    f"gs={st.game_state}"
                )
                if watch.success:
                    out = Path(args.out or default_out)
                    save_dev_state(env, out)
                    watch.detail += f" saved {out.name}"
            except Exception as exc:  # noqa: BLE001
                watch.error = str(exc)
                watch.phase = "fail"
                watch.detail = watch.error[:80]
            watch.done = True
        return np.zeros(12, dtype=np.int8)

    def on_hud(_info):
        st = watch.state
        try:
            w = spark.read_spark_wram(env)
            echoes = w["speed_echoes"]
            timer = w["spark_timer"]
        except Exception:  # noqa: BLE001
            echoes = getattr(st, "speed_counter", 0)
            timer = getattr(st, "shinespark_timer", 0)
        flag = (
            "GREEN"
            if watch.success
            else ("RED" if watch.error else "RUN")
        )
        recipe = (
            "ocean-floor charge→store→hop→spark→Super 0xCA08"
            if mode == "ws"
            else "edge store→hop→spark → bowling 0xC98E (not product WS)"
        )
        return [
            f"{flag} phase={watch.phase}  {watch.detail}",
            f"room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) p={st.pose}",
            f"echoes={echoes} $0A68={timer} frame={watch.frame} charge={charge_mode}",
            recipe,
            "[ ] speed · TAB turbo · ESC quit",
        ]

    import retro_harness.play_session as ps_mod

    _orig_reset = ps_mod.reset_env

    def _reset_then_boot(e):
        obs, info = _orig_reset(e)
        e.em.set_state(state_bytes)
        for _ in range(12):
            obs, *_rest, info = step_env(e, idle_action())
            if assist is not None:
                st = parse_env_state(e, mode="nav")
                try:
                    assist.apply(e.data, st)
                except Exception:  # noqa: BLE001
                    try:
                        assist.apply(e, st)
                    except Exception:  # noqa: BLE001
                        pass
        watch.state = parse_env_state(e, mode="nav")
        watch.frame = 0
        if place_spit and watch.state.room_id == wo.ROOM_WEST_OCEAN:
            if watch.state.samus_y > 900 or watch.state.samus_x < 200:
                place_samus(e, place_xy[0], place_xy[1])
                write_wram_u16(e, 0x18AA, 0)
                write_wram_u16(e, 0x18A8, 0x400)
                for i in range(100):
                    obs, *_rest, info = step_env(e, idle_action())
                    watch.state = parse_env_state(e, mode="nav")
                    if watch.state.velocity_y == 0 and i > 15:
                        break
        print(
            f"[BOOT] room=0x{watch.state.room_id:04X} "
            f"xy=({watch.state.samus_x},{watch.state.samus_y}) charge={charge_mode} mode={mode}"
        )
        print(f"[BOT] {title} — HUD echoes / $0A68")
        return obs, info

    ps_mod.reset_env = _reset_then_boot  # type: ignore[assignment]
    try:
        session = PlaySession(
            env,
            game_dir=str(GAME_DIR),
            game=GAME,
            scale=args.scale,
            title=title,
            bot=bot,
            action_size=12,
            base_fps=60,
            initial_speed=args.speed,
            headless=False,
        )
        session.on_hud = on_hud
        session.run()
    finally:
        ps_mod.reset_env = _orig_reset  # type: ignore[assignment]
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    if watch.success:
        print(f"SUCCESS frames≈{watch.frame} {watch.detail}")
        return 0
    if watch.error:
        print(f"RED {watch.error}")
        return 1
    print(f"ended phase={watch.phase} frames≈{watch.frame}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Headed edge spark (bowling practice)."""
    return _run_watch(args, mode="edge")


def cmd_watch_ws(args: argparse.Namespace) -> int:
    """Headed over-ocean spark → green Super WS (product)."""
    return _run_watch(args, mode="ws")


def cmd_pure(args: argparse.Namespace) -> int:
    """Headless spit-edge spark → bowling 0xC98E (practice; not product WS)."""
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
            charge_mode=args.charge_mode,
        )
        print(
            f"GREEN room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) "
            f"pose={st.pose} frames={sess.frame} charge={args.charge_mode} "
            f"(bowling practice — not product WS)"
        )
        out = Path(args.out or DEFAULT_OUT)
        save_dev_state(env, out)
        print(f"saved {out}")
        DEBUG.mkdir(parents=True, exist_ok=True)
        report_name = (
            "pure.json" if args.charge_mode == "full" else f"pure_{args.charge_mode}.json"
        )
        (DEBUG / report_name).write_text(
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
                        "charge_mode": args.charge_mode,
                        "path": "edge_bowling",
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


def cmd_pure_ws(args: argparse.Namespace) -> int:
    """Headless over-ocean spark → green Super WS 0xCA08 (product).

    Natural Moat handoff — no spit free-place. Default charge ``stutter``.
    """
    source = Path(args.source or DEFAULT_SOURCE)
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2
    # Product path: natural lower-left; optional place only if asked
    place = bool(getattr(args, "place", False))
    env, sess = boot(
        source,
        place_spit=place,
        place_xy=(
            getattr(args, "place_x", wo.OCEAN_FLOOR_PLACE_XY[0]),
            getattr(args, "place_y", wo.OCEAN_FLOOR_PLACE_XY[1]),
        ),
        assist=not args.no_assist,
    )
    boot_snap = spark.spark_snapshot(env, 0)
    print(
        f"boot room=0x{sess.state.room_id:04X} xy=({sess.state.samus_x},{sess.state.samus_y}) "
        f"pose={sess.state.pose} place={place} path=over_ocean_ws"
    )
    try:
        st = wo.play_west_ocean_over_ocean_spark(
            sess,
            hop_frames=args.hop,
            charge_mode=args.charge_mode,
        )
        ok = (
            st.room_id == wo.ROOM_WS_ENTRANCE
            and st.game_state == 8
            and st.door_transition == 0
        )
        flag = "GREEN" if ok else "AMBER"
        print(
            f"{flag} room=0x{st.room_id:04X} xy=({st.samus_x},{st.samus_y}) "
            f"pose={st.pose} gs={st.game_state} dt={st.door_transition} "
            f"frames={sess.frame} charge={args.charge_mode}"
        )
        out = Path(args.out or DEFAULT_WS_OUT)
        save_dev_state(env, out)
        print(f"saved {out}")
        DEBUG.mkdir(parents=True, exist_ok=True)
        report_name = (
            "pure_ws.json"
            if args.charge_mode == "stutter"
            else f"pure_ws_{args.charge_mode}.json"
        )
        (DEBUG / report_name).write_text(
            json.dumps(
                {
                    "boot": boot_snap,
                    "final": spark.spark_snapshot(env, sess.frame),
                    "frames": sess.frame,
                    "ok": ok,
                    "params": {
                        "hop": args.hop,
                        "charge_mode": args.charge_mode,
                        "place": place,
                        "path": "over_ocean_ws",
                        "target": "0xCA08",
                    },
                },
                indent=2,
            )
            + "\n"
        )
        return 0 if st.room_id == wo.ROOM_WS_ENTRANCE else 1
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

    def _add_edge_common(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--source", type=Path, default=None)
        pp.add_argument("--out", type=Path, default=None)
        pp.add_argument("--back", type=int, default=wo.DEFAULT_BACK_FRAMES)
        pp.add_argument("--hop", type=int, default=wo.DEFAULT_HOP_FRAMES)
        pp.add_argument("--aim", type=str, default="RIGHT", help="e.g. RIGHT or RIGHT+UP")
        pp.add_argument("--no-place", action="store_true", help="Do not free-place spit")
        pp.add_argument("--place-x", type=int, default=wo.SPIT_PLACE_XY[0])
        pp.add_argument("--place-y", type=int, default=wo.SPIT_PLACE_XY[1])
        pp.add_argument("--no-assist", action="store_true")
        pp.add_argument(
            "--charge-mode",
            choices=("full", "short", "stutter"),
            default="full",
            help="full=continuous dash; short/stutter=magic-frame short charge",
        )

    def _add_ws_common(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--source", type=Path, default=None)
        pp.add_argument("--out", type=Path, default=None)
        pp.add_argument(
            "--hop",
            type=int,
            default=wo.DEFAULT_OCEAN_HOP_FRAMES,
            help="A-hop frames after store before spark activate",
        )
        pp.add_argument(
            "--place",
            action="store_true",
            help="Optional free-place ocean floor (default: natural Moat handoff)",
        )
        pp.add_argument("--place-x", type=int, default=wo.OCEAN_FLOOR_PLACE_XY[0])
        pp.add_argument("--place-y", type=int, default=wo.OCEAN_FLOOR_PLACE_XY[1])
        pp.add_argument("--no-assist", action="store_true")
        pp.add_argument(
            "--charge-mode",
            choices=("full", "short", "stutter"),
            default="stutter",
            help="Product default stutter; short also greens; full usually RED",
        )

    p = sub.add_parser("pure", help="Headless edge spark → bowling 0xC98E (practice)")
    _add_edge_common(p)
    p.set_defaults(func=cmd_pure)

    p2 = sub.add_parser("hop", help="Alias of pure (edge bowling)")
    _add_edge_common(p2)
    p2.set_defaults(func=cmd_pure)

    p_ws = sub.add_parser(
        "pure-ws",
        help="Headless over-ocean spark → green Super WS 0xCA08 (product)",
    )
    _add_ws_common(p_ws)
    p_ws.set_defaults(func=cmd_pure_ws)

    p3 = sub.add_parser(
        "short-charge",
        help="Measure short/stutter charge delta_x on spit (no spark)",
    )
    p3.add_argument("--source", type=Path, default=None)
    p3.add_argument("--no-place", action="store_true")
    p3.add_argument("--place-x", type=int, default=wo.SPIT_PLACE_XY[0])
    p3.add_argument("--place-y", type=int, default=wo.SPIT_PLACE_XY[1])
    p3.add_argument("--no-assist", action="store_true")
    p3.add_argument(
        "--mode",
        choices=("short", "stutter", "full"),
        default="stutter",
    )
    p3.add_argument(
        "--store-on-last",
        action="store_true",
        help="Press DOWN on final magic frame",
    )
    p3.set_defaults(func=cmd_short_charge)

    p4 = sub.add_parser(
        "watch",
        help="Headed edge spark → bowling (practice)",
    )
    _add_edge_common(p4)
    p4.add_argument("--scale", type=int, default=2)
    p4.add_argument("--speed", type=float, default=1.0)
    p4.set_defaults(func=cmd_watch)

    p5 = sub.add_parser(
        "watch-ws",
        help="Headed over-ocean → WS 0xCA08 (product; prefer for review)",
    )
    _add_ws_common(p5)
    p5.add_argument("--scale", type=int, default=2)
    p5.add_argument("--speed", type=float, default=1.0)
    p5.set_defaults(func=cmd_watch_ws)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
