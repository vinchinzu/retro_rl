#!/usr/bin/env python3
"""Development probe: natural Super room → farming → Big Pink → crest / 3b.

```bash
uv run python super_metroid/scripts/probe_post_spore_pb.py
uv run python super_metroid/scripts/probe_post_spore_pb.py --to farming
uv run python super_metroid/scripts/probe_post_spore_pb.py --to supers
uv run python super_metroid/scripts/probe_post_spore_pb.py --to crest
uv run python super_metroid/scripts/probe_post_spore_pb.py --to super-block
uv run python super_metroid/scripts/probe_post_spore_pb.py --to tunnel-west \\
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_big_pink_open.state
```

Route board: ``docs/ROUTE_SUPERS_TO_PHANTOON.md``. Starts from
``natural_post_spore_spawn.state`` unless ``--source`` overrides. Not continuous
acceptance evidence.

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.env import make_env, read_state_bytes, write_state_bytes  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR  # noqa: E402
from super_metroid.dev_common import place_samus  # noqa: E402
from super_metroid.post_spore_controller import (  # noqa: E402
    play_big_pink_bomb_to_walkway_edge,
    play_big_pink_clear_super_block,
    play_big_pink_crest_pocket,
    play_big_pink_drop_to_pocket,
    play_big_pink_enter_pb_door_from_sill,
    play_big_pink_into_main_shaft,
    play_big_pink_morph_to_tunnel,
    play_big_pink_tunnel_west,
    play_farming_to_big_pink,
    play_pink_pb_break_maze_wall,
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
            "pb-maze-wall",
            "pb-collect",
        ),
        default="crest",
        help=(
            "How far to run (default: crest). "
            "tunnel-floor = crest+S-clear+double-tap morph. "
            "main = full into_main_shaft (controller). "
            "main-bridged = place hop (legacy dev bridge). "
            "pb-door = enter 0x9E11 from sill (expects on sill; place for dev). "
            "pb-maze-wall = pure morph-bomb open wall@437 from bottom spawn. "
            "pb-collect = morph-bomb collect (expects pocket x≤225; place bridge)."
        ),
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
    args = parser.parse_args()

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedResourcesAssist()
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
            # Expects Big Pink on bottom-door sill (~580,1136). Place for dev.
            if session.state.room_id == 0x9D19 and not (
                560 <= session.state.samus_x <= 620
                and 1100 <= session.state.samus_y <= 1160
            ):
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
        elif args.to == "pb-maze-wall":
            # Pure controller: open wall@437 from bottom-door spawn in 0x9E11.
            if session.state.room_id != 0x9E11:
                if session.state.room_id == 0x9D19:
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
            play_pink_pb_break_maze_wall(session)
            result["pbMazeWall"] = {
                "roomIdHex": f"0x{session.state.room_id:04X}",
                "samusX": session.state.samus_x,
                "samusY": session.state.samus_y,
            }
            result["success"] = (
                session.state.room_id == 0x9E11 and session.state.samus_x <= 410
            )
        elif args.to == "pb-collect":
            # Expects 0x9E11 collect pocket (x≤225, y≈395). Place bridge if needed.
            if session.state.room_id != 0x9E11:
                # Enter first via sill bridge if still in Big Pink.
                if session.state.room_id == 0x9D19:
                    place_samus(env, 580, 1136)
                    write_wram_u16(env, 0x0A1C, 1)
                    for _ in range(12):
                        env.step(idle_action())
                        session.state = parse_state(env.get_ram(), frame=session.frame)
                        session.assist.apply(env.data, session.state)
                    play_big_pink_enter_pb_door_from_sill(session)
            if session.state.room_id == 0x9E11 and session.state.samus_x > 410:
                # Pure open wall@437 first when still east of it.
                play_pink_pb_break_maze_wall(session)
                result["pbMazeWall"] = {
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
            if session.state.room_id == 0x9E11 and (
                session.state.samus_x > 225 or abs(session.state.samus_y - 395) > 40
            ):
                place_samus(env, 220, 395)
                write_wram_u16(env, 0x0A1C, 1)
                for _ in range(8):
                    env.step(idle_action())
                    session.state = parse_state(env.get_ram(), frame=session.frame)
                    session.assist.apply(env.data, session.state)
                result["mazeBridge"] = {
                    "note": "place_samus(220,395) — mid-maze 405→225 still open",
                    "samusX": session.state.samus_x,
                    "samusY": session.state.samus_y,
                }
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
            "maxSuperMissiles": session.state.max_super_missiles,
            "maxPowerBombs": session.state.max_power_bombs,
            "frame": session.frame,
        }
        if args.save is not None:
            write_state_bytes(args.save, env.em.get_state())
            result["savedState"] = str(args.save)
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
