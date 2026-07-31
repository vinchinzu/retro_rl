#!/usr/bin/env python3
"""Capture a correctly-roomed HJ shaft fixture from the existing HJ anchor.

The input anchor is development-only. The controller leg itself uses inputs
only; the resulting fixture remains development-only until natural-entry
provenance is established by a planner/reviewer.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import (  # noqa: E402
    boot_from_state,
    make_dev_env,
    save_dev_state,
)
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.kpdr.hijump_return import (  # noqa: E402
    play_hj_room_to_shaft,
)
from super_metroid.routes.kpdr.rooms import ROOM_HJ_SHAFT  # noqa: E402


SOURCE = Path(
    "custom_integrations/SuperMetroid-Snes/dev_hijump_collected_dev.state"
)
OUTPUT = Path(
    "custom_integrations/SuperMetroid-Snes/scratch/"
    "hj_shaft_to_business_source.state"
)


class ProbeSession:
    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env)

    def step(self, action, reason: str = ""):
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame)
        self.assist.apply(self.env.data, self.state)
        return self.state


def main() -> None:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot = boot_from_state(env, SOURCE)
        for _ in range(5):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env))
        session = ProbeSession(env, assist)
        play_hj_room_to_shaft(session)
        state = session.state
        if state.room_id != ROOM_HJ_SHAFT:
            raise RuntimeError(
                f"expected room 0x{ROOM_HJ_SHAFT:04X}, "
                f"got 0x{state.room_id:04X}"
            )
        save_dev_state(env, OUTPUT)
        print(
            json.dumps(
                {
                    "success": True,
                    "source": str(SOURCE),
                    "statePath": str(OUTPUT.resolve()),
                    "roomIdHex": f"0x{state.room_id:04X}",
                    "samusX": state.samus_x,
                    "samusY": state.samus_y,
                    "pose": state.pose,
                    "frame": session.frame,
                    "bootRoomIdHex": f"0x{boot.room_id:04X}",
                    "controllerOnlyLeg": True,
                    "developmentOnly": True,
                },
                indent=2,
            )
        )
    except Exception as exc:
        state = parse_env_state(env)
        print(
            json.dumps(
                {
                    "success": False,
                    "error": str(exc),
                    "roomIdHex": f"0x{state.room_id:04X}",
                    "samusX": state.samus_x,
                    "samusY": state.samus_y,
                    "pose": state.pose,
                    "developmentOnly": True,
                },
                indent=2,
            )
        )
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
