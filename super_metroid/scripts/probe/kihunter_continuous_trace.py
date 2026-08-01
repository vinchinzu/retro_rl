#!/usr/bin/env python3
"""Trace a post-Varia reverse controller from a named natural source.

This is a read-only geometry diagnostic.  It uses the same lightweight
navigation-state probe session as ``kpdr.py pure`` and records each controller
step so a failed launch can be redesigned from the exact x/y trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.ram import parse_env_state, probe_pin  # noqa: E402
from super_metroid.routes.kpdr.kraid_return import (  # noqa: E402
    play_kihunter_to_zeela_return,
    play_zeela_to_warehouse_return,
)
from super_metroid.routes.kpdr.warehouse import play_warehouse_to_business  # noqa: E402
from super_metroid.scripts.probe.kpdr import _ProbeSession  # noqa: E402


class _TraceSession(_ProbeSession):
    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        super().__init__(env, assist)
        self.trace: list[dict[str, object]] = []

    def step(self, action, reason: str = ""):
        state = super().step(action, reason)
        self.trace.append({"reason": reason, **probe_pin(state)})
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment",
        choices=(
            "kihunter-to-zeela-return",
            "zeela-to-warehouse-return",
            "warehouse-to-business",
        ),
        default="kihunter-to-zeela-return",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Optional RGB screenshot at the controller's final state.",
    )
    args = parser.parse_args()

    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, args.source)
        for _ in range(5):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))
        session = _TraceSession(env, assist)
        error = None
        try:
            {
                "kihunter-to-zeela-return": play_kihunter_to_zeela_return,
                "zeela-to-warehouse-return": play_zeela_to_warehouse_return,
                "warehouse-to-business": play_warehouse_to_business,
            }[args.segment](session)
        except Exception as exc:  # diagnostic: preserve the controller error
            error = f"{type(exc).__name__}: {exc}"
        payload = {
            "success": error is None,
            "error": error,
            "source": str(args.source),
            "segment": args.segment,
            "final": probe_pin(session.state),
            "trace": session.trace,
        }
        if args.screenshot is not None:
            import cv2

            frame = env.get_screen()
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(args.screenshot), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            payload["screenshot"] = str(args.screenshot)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in payload.items() if k != "trace"}, indent=2))
        sys.exit(0 if error is None else 1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
