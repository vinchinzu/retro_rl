#!/usr/bin/env python3
"""Development probe: natural Super room → farming → Big Pink → crest / 3b.

```bash
uv run python snes/super_metroid/scripts/probe/post_spore_pb.py
uv run python snes/super_metroid/scripts/probe/post_spore_pb.py --to farming
uv run python snes/super_metroid/scripts/probe/post_spore_pb.py --to supers
uv run python snes/super_metroid/scripts/probe/post_spore_pb.py --to crest
uv run python snes/super_metroid/scripts/probe/post_spore_pb.py --to super-block
uv run python snes/super_metroid/scripts/probe/post_spore_pb.py --to tunnel-west \\
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_big_pink_open.state
```

KPDR K0/K1 board: ``docs/routes/ROUTE_KPDR.md`` (controllers in
``routes/kpdr/``). Starts from ``natural_post_spore_spawn.state`` unless
``--source`` overrides. Not continuous acceptance evidence.

``super-block``: crest + crouch-Super clear of tile (69, 87).
``tunnel-west``: morph-roll + X bombs from raised tunnel floor (use open/tunnel
source; Super block must already be clear for full west progress).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get('_SNES_IMPORT_ROOT', ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.env import make_env, read_state_bytes, write_state_bytes  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR  # noqa: E402
from super_metroid.dev.common import place_samus  # noqa: E402
from super_metroid.routes.kpdr import (  # noqa: E402
    play_big_pink_bomb_to_walkway_edge,
    play_big_pink_clear_super_block,
    play_big_pink_crest_pocket,
    play_big_pink_drop_to_pocket,
    play_big_pink_enter_pb_door_from_sill,
    play_big_pink_enter_pb_door_from_top_ledge,
    play_big_pink_into_main_shaft,
    play_big_pink_morph_to_tunnel,
    play_big_pink_tunnel_west,
    play_farming_to_big_pink,
    play_pink_pb_break_maze_wall,
    play_pink_pb_from_left_zone,
    play_pink_pb_mid_maze_to_collect,
    play_pink_pb_morph_bomb_collect,
    play_super_room_collect,
    play_super_room_to_farming,
)
from super_metroid.ram import parse_state, write_wram_u16  # noqa: E402


@dataclass
class _Session:
    env: object
    assist: UnlimitedResourcesAssist
    frame: int = 0

    def __post_init__(self) -> None:
        self.action_reasons: Counter[str] = Counter()
        self.state = parse_state(self.env.get_ram(), frame=0)

    def step(self, action, reason: str):
        self.env.step(action)
        self.frame += 1
        self.state = parse_state(self.env.get_ram(), frame=self.frame)
        self.assist.apply(self.env.data, self.state)
        self.action_reasons[reason] += 1
        return self.state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--to",
        choices=(
            "supers",
            "farming",
            "big-pink",
            "crest",
            "super-block",
            "walkway-edge",
            "tunnel-floor",
            "tunnel-west",
            "main",
            "main-bridged",
            "pb-door",
            "pb-top-door",
            "pb-maze-wall",
            "pb-mid-maze",
            "pb-collect",
        ),
        default="crest",
        help=(
            "How far to run (default: crest). "
            "tunnel-floor = crest+S-clear+double-tap morph. "
            "main = full into_main_shaft (controller). "
            "main-bridged = place hop (legacy dev bridge). "
            "pb-door = enter 0x9E11 from bottom sill (expects on sill; place for dev). "
            "pb-top-door = enter via solid top ledge y≈907 (preferred; place for dev). "
            "pb-maze-wall = pure morph-bomb open wall@437 from bottom spawn. "
            "pb-mid-maze = pure wall + mid-maze 405→225 + collect (no place). "
            "pb-collect = morph-bomb collect; pure mid-maze first, place only "
            "with --allow-place."
        ),
    )
    parser.add_argument(
        "--allow-place",
        action="store_true",
        help="Allow place_samus bridges on pb-door / pb-collect (debug only).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=INTEGRATION_DIR / "natural_post_spore_spawn.state",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional dev state output path",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="If >0, log x,y,pose,vel every N frames inside bomb-roll loops",
    )
    parser.add_argument(
        "--save-fail",
        type=Path,
        default=None,
        help=(
            "On TimeoutError/RuntimeError, write emulator state here "
            "(default: <game>/debug/post_spore/fail_<to>.state)"
        ),
    )
    args = parser.parse_args()

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedResourcesAssist()
    session: _Session | None = None
    try:
        env.reset()
        env.em.set_state(read_state_bytes(args.source))
        for _ in range(5):
            env.step(idle_action())
        session = _Session(env, assist)
        result: dict[str, object] = {"developmentOnly": True}

        if args.to == "tunnel-west":
            # Suffix-only: expects raised tunnel floor with Super block clear.
            play_big_pink_tunnel_west(session)
            result["tunnelWest"] = {
                "roomIdHex": f"0x{session.state.room_id:04X}",
                "samusX": session.state.samus_x,
                "samusY": session.state.samus_y,
            }
            result["success"] = (
                session.state.room_id == 0x9D19 and session.state.samus_x <= 750
            )
        elif args.to == "pb-door":
            # Expects Big Pink on bottom-door sill (~580,1136).
            if session.state.room_id == 0x9D19 and not (
                560 <= session.state.samus_x <= 620
                and 1100 <= session.state.samus_y <= 1160
            ):
                if not args.allow_place:
                    raise SystemExit(
                        "pb-door: not on sill; use a sill state or --allow-place"
                    )
                place_samus(env, 580, 1136)
                write_wram_u16(env, 0x0A1C, 1)
                for _ in range(12):
                    env.step(idle_action())
                    session.state = parse_state(env.get_ram(), frame=session.frame)
                    session.assist.apply(env.data, session.state)
                result["sillBridge"] = {
                    "note": "place_samus(580,1136) — approach still open",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
            play_big_pink_enter_pb_door_from_sill(session)
            result["pbDoor"] = {
                "roomIdHex": f"0x{session.state.room_id:04X}",
                "samusX": session.state.samus_x,
                "samusY": session.state.samus_y,
            }
            result["success"] = session.state.room_id == 0x9E11
        elif args.to == "pb-top-door":
            # Prefer solid top ledge (~532,907); place bridge for dev.
            on_top = (
                session.state.room_id == 0x9D19
                and 510 <= session.state.samus_x <= 570
                and 890 <= session.state.samus_y <= 930
            )
            if session.state.room_id == 0x9D19 and not on_top:
                if not args.allow_place:
                    raise SystemExit(
                        "pb-top-door: not on top ledge y≈907; "
                        "use a top-ledge state or --allow-place"
                    )
                place_samus(env, 532, 907)
                write_wram_u16(env, 0x0A1C, 1)
                for _ in range(12):
                    env.step(idle_action())
                    session.state = parse_state(env.get_ram(), frame=session.frame)
                    session.assist.apply(env.data, session.state)
                result["topLedgeBridge"] = {
                    "note": "place_samus(532,907) — pure climb still open",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
            # Already transitioned during place settle?
            if session.state.room_id == 0x9E11:
                result["pbTopDoor"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                    "note": "entered during place settle",
                }
            else:
                play_big_pink_enter_pb_door_from_top_ledge(session)
                result["pbTopDoor"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
            result["success"] = session.state.room_id == 0x9E11
        elif args.to == "pb-maze-wall":
            # Pure controller: open wall@437 from bottom-door spawn in 0x9E11.
            if session.state.room_id != 0x9E11:
                if session.state.room_id == 0x9D19 and args.allow_place:
                    place_samus(env, 580, 1136)
                    write_wram_u16(env, 0x0A1C, 1)
                    for _ in range(12):
                        env.step(idle_action())
                        session.state = parse_state(env.get_ram(), frame=session.frame)
                        session.assist.apply(env.data, session.state)
                    play_big_pink_enter_pb_door_from_sill(session)
                    result["sillBridge"] = {
                        "note": "entered 0x9E11 via sill place for maze-wall probe",
                    }
                else:
                    raise SystemExit(
                        "pb-maze-wall: need 0x9E11 source "
                        "(e.g. dev_b1_pb_door_entered) or --allow-place from Big Pink"
                    )
            play_pink_pb_break_maze_wall(session)
            result["pbMazeWall"] = {
                "roomIdHex": f"0x{session.state.room_id:04X}",
                "samusX": session.state.samus_x,
                "samusY": session.state.samus_y,
                "pose": session.state.pose,
            }
            result["success"] = (
                session.state.room_id == 0x9E11 and session.state.samus_x <= 410
            )
        elif args.to == "pb-mid-maze":
            # Pure wall + mid-maze → collect (no place).
            if session.state.room_id != 0x9E11:
                raise SystemExit(
                    "pb-mid-maze: need 0x9E11 source "
                    "(e.g. dev_b1_pb_door_entered or dev_b1_pb_x405)"
                )
            if session.state.samus_x > 410:
                play_pink_pb_break_maze_wall(session)
                result["pbMazeWall"] = {
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                    "pose": session.state.pose,
                }
            play_pink_pb_mid_maze_to_collect(
                session, log_every=args.log_every or 0
            )
            result["pbMidMaze"] = {
                "roomIdHex": f"0x{session.state.room_id:04X}",
                "samusX": session.state.samus_x,
                "samusY": session.state.samus_y,
                "maxPowerBombs": session.state.max_power_bombs,
                "pose": session.state.pose,
            }
            result["success"] = session.state.max_power_bombs > 0
        elif args.to == "pb-collect":
            # Prefer pure mid-maze; place bridge only with --allow-place.
            if session.state.room_id != 0x9E11:
                if session.state.room_id == 0x9D19 and args.allow_place:
                    place_samus(env, 580, 1136)
                    write_wram_u16(env, 0x0A1C, 1)
                    for _ in range(12):
                        env.step(idle_action())
                        session.state = parse_state(env.get_ram(), frame=session.frame)
                        session.assist.apply(env.data, session.state)
                    play_big_pink_enter_pb_door_from_sill(session)
                    result["sillBridge"] = {
                        "note": "entered 0x9E11 via sill place",
                    }
                else:
                    raise SystemExit(
                        "pb-collect: need 0x9E11 source or --allow-place from Big Pink"
                    )
            if session.state.room_id == 0x9E11 and session.state.samus_x > 410:
                play_pink_pb_break_maze_wall(session)
                result["pbMazeWall"] = {
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
            if session.state.room_id == 0x9E11 and session.state.samus_x > 225:
                # Pure mid-maze first.
                try:
                    play_pink_pb_mid_maze_to_collect(
                        session, log_every=args.log_every or 0
                    )
                    result["pbMidMaze"] = {
                        "samusX": session.state.samus_x,
                        "samusY": session.state.samus_y,
                        "maxPowerBombs": session.state.max_power_bombs,
                        "note": "pure mid-maze",
                    }
                except (TimeoutError, RuntimeError) as exc:
                    if not args.allow_place:
                        raise
                    # Prefer left free volume (map-informed) then pure drop+collect.
                    place_samus(env, 180, 360)
                    write_wram_u16(env, 0x0A1C, 1)
                    for _ in range(8):
                        env.step(idle_action())
                        session.state = parse_state(env.get_ram(), frame=session.frame)
                        session.assist.apply(env.data, session.state)
                    result["mazeBridge"] = {
                        "note": (
                            f"place_samus(180,360) left-zone after pure mid-maze fail: {exc}"
                        ),
                        "samusX": session.state.samus_x,
                        "samusY": session.state.samus_y,
                    }
                    play_pink_pb_from_left_zone(session)
            elif session.state.max_power_bombs <= 0:
                play_pink_pb_morph_bomb_collect(session)
            result["pbCollect"] = {
                "roomIdHex": f"0x{session.state.room_id:04X}",
                "samusX": session.state.samus_x,
                "samusY": session.state.samus_y,
                "maxPowerBombs": session.state.max_power_bombs,
            }
            result["success"] = session.state.max_power_bombs > 0
        else:
            evidence = play_super_room_collect(session)
            result["superCollect"] = evidence.to_dict()
            suffix = (
                "farming",
                "big-pink",
                "crest",
                "super-block",
                "walkway-edge",
                "tunnel-floor",
                "main",
                "main-bridged",
            )
            if args.to in suffix:
                play_super_room_to_farming(session)
                result["farming"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                    "maxSuperMissiles": session.state.max_super_missiles,
                }
            if args.to in suffix[1:]:
                play_farming_to_big_pink(session)
                result["bigPink"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                    "maxSuperMissiles": session.state.max_super_missiles,
                }
            if args.to in (
                "crest",
                "super-block",
                "walkway-edge",
                "tunnel-floor",
                "main-bridged",
            ):
                play_big_pink_crest_pocket(session)
                result["crest"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                    "maxSuperMissiles": session.state.max_super_missiles,
                }
            if args.to in (
                "super-block",
                "walkway-edge",
                "tunnel-floor",
                "main-bridged",
            ):
                play_big_pink_clear_super_block(session)
                result["superBlock"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
            if args.to == "walkway-edge":
                play_big_pink_drop_to_pocket(session)
                play_big_pink_bomb_to_walkway_edge(session)
                result["walkwayEdge"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
            if args.to == "tunnel-floor":
                play_big_pink_morph_to_tunnel(session)
                result["tunnelFloor"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
            if args.to == "main":
                play_big_pink_into_main_shaft(session)
                result["mainShaft"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
            if args.to == "main-bridged":
                # Legacy place bridge (prefer --to main).
                place_samus(env, 1140, 1401)
                write_wram_u16(env, 0x0A1C, 0x41)  # morph ground left
                write_wram_u16(env, 0x0B2E, 0)
                for _ in range(10):
                    session.step(idle_action(), "bridge_settle")
                result["bridge"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                    "developmentOnly": True,
                    "note": "legacy place_samus; prefer --to main",
                }
                play_big_pink_tunnel_west(session)
                result["mainShaft"] = {
                    "roomIdHex": f"0x{session.state.room_id:04X}",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
            crested = (
                args.to
                not in (
                    "crest",
                    "super-block",
                    "walkway-edge",
                    "tunnel-floor",
                    "main",
                    "main-bridged",
                )
                or (
                    session.state.room_id == 0x9D19
                    and session.state.samus_x <= 1135
                )
                or args.to
                in ("walkway-edge", "tunnel-floor", "main", "main-bridged")
            )
            main_ok = args.to not in ("main", "main-bridged") or (
                session.state.room_id == 0x9D19 and session.state.samus_x <= 750
            )
            result["success"] = (
                session.state.max_super_missiles >= 5 and crested and main_ok
            )

        result["final"] = {
            "roomIdHex": f"0x{session.state.room_id:04X}",
            "samusX": session.state.samus_x,
            "samusY": session.state.samus_y,
            "pose": session.state.pose,
            "velocityY": session.state.velocity_y,
            "velocityX": session.state.velocity_x,
            "maxSuperMissiles": session.state.max_super_missiles,
            "maxPowerBombs": session.state.max_power_bombs,
            "frame": session.frame,
        }
        if args.save is not None:
            write_state_bytes(args.save, env.em.get_state())
            result["savedState"] = str(args.save)
        print(json.dumps(result, indent=2))
    except (TimeoutError, RuntimeError) as exc:
        # Always dump failure coordinates for threshold tuning.
        fail_path = args.save_fail
        if fail_path is None:
            fail_path = (
                GAME_DIR
                / "debug"
                / "post_spore"
                / f"fail_{args.to.replace('-', '_')}.state"
            )
        fail_path = Path(fail_path)
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_state_bytes(fail_path, env.em.get_state())
        except Exception:
            fail_path = None
        if session is not None:
            s = session.state
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": str(exc),
                        "failState": str(fail_path) if fail_path else None,
                        "final": {
                            "roomIdHex": f"0x{s.room_id:04X}",
                            "samusX": s.samus_x,
                            "samusY": s.samus_y,
                            "pose": s.pose,
                            "velocityY": s.velocity_y,
                            "velocityX": s.velocity_x,
                            "maxPowerBombs": s.max_power_bombs,
                            "frame": session.frame,
                        },
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
