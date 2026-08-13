#!/usr/bin/env python3
"""Probe full-knowledge Phantoon Super-spray strategy (policy only, no RL).

Starts from a Phantoon-room entry state (human ship tape preferred) and runs
the deterministic controller until body HP 0 + Wrecked Ship boss bit 0.

```bash
# From human ship free-record end (in room 0xCD13)
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py strategy \\
  --state ws_ship_human_end --save-state

# Explicit path + custom out pin
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py strategy \\
  --state snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/ws_ship_human_end.state \\
  --save-state snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_phantoon_defeated.state
```
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get("_SNES_IMPORT_ROOT", ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from retro_harness.env import make_env, read_state_bytes  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.combat.features import phantoon_catalog  # noqa: E402
from super_metroid.combat.phantoon import (  # noqa: E402
    ROOM_PHANTOON,
    WEAPON_SUPERS,
    PhantoonStrategy,
    play_phantoon_fight,
)
from super_metroid.dev.common import save_dev_state  # noqa: E402
from super_metroid.dev.phantoon_dev import (  # noqa: E402
    PHANTOON_DEFEATED_STATE,
    PHANTOON_ENTRY_STATE,
    phantoon_defeated,
    wrecked_ship_boss_bits,
)
from super_metroid.paths import (  # noqa: E402
    GAME,
    GAME_DIR,
    INTEGRATION_DIR,
    SCRATCH_STATE_DIR,
)
from super_metroid.ram import parse_state  # noqa: E402

DEFAULT_ENTRY = SCRATCH_STATE_DIR / "ws_ship_human_end.state"
DEFAULT_OUT = SCRATCH_STATE_DIR / "post_phantoon_defeated.state"

_NAMED_STATES: dict[str, Path] = {
    "ws_ship_human_end": DEFAULT_ENTRY,
    "human": DEFAULT_ENTRY,
    "human-end": DEFAULT_ENTRY,
    "entry": PHANTOON_ENTRY_STATE,
    "dev_entry": PHANTOON_ENTRY_STATE,
    "dev_phantoon_entry": PHANTOON_ENTRY_STATE,
}


class _Session:
    """Minimal ControllerSession for combat probes."""

    def __init__(self, env: object, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.action_reasons: Counter[str] = Counter()
        self.state = parse_state(env.get_ram(), frame=0)  # type: ignore[attr-defined]

    def step(self, action, reason: str):
        self.env.step(action)  # type: ignore[attr-defined]
        self.frame += 1
        self.state = parse_state(self.env.get_ram(), frame=self.frame)  # type: ignore[attr-defined]
        self.assist.apply(self.env.data, self.state)  # type: ignore[attr-defined]
        self.action_reasons[reason] += 1
        return self.state


def _resolve_state(name: str) -> Path:
    key = name.strip()
    if key in _NAMED_STATES:
        return _NAMED_STATES[key]
    path = Path(key)
    if path.suffix == ".state" or "/" in key or path.exists():
        if not path.is_absolute():
            for candidate in (
                path,
                GAME_DIR / path,
                INTEGRATION_DIR / path.name,
                SCRATCH_STATE_DIR / path.name,
                GAME_DIR / "tasks" / path.name,
            ):
                if candidate.exists():
                    return candidate
        return path
    for candidate in (
        SCRATCH_STATE_DIR / f"{key}.state",
        INTEGRATION_DIR / f"{key}.state",
        GAME_DIR / "tasks" / f"{key}.state",
    ):
        if candidate.exists():
            return candidate
    return path


def _open_env(state_path: Path):
    if not state_path.exists():
        raise FileNotFoundError(
            f"Phantoon entry state not found: {state_path}\n"
            "Record ship path: guided_human.py --from ws-entrance --name ws_ship_human\n"
            "Or use --state entry for dev_phantoon_entry."
        )
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.em.set_state(read_state_bytes(state_path))
    for _ in range(4):
        env.step([0] * 12)
    return env, str(state_path)


def cmd_strategy(args: argparse.Namespace) -> int:
    catalog = phantoon_catalog()
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist()
    try:
        session = _Session(env, assist)
        if session.state.room_id != ROOM_PHANTOON:
            report = {
                "command": "strategy",
                "state": loaded,
                "success": False,
                "outcome": "wrong_room",
                "room_id_hex": f"0x{session.state.room_id:04X}",
                "notes": "Load a Phantoon-room entry state (0xCD13).",
            }
            print(json.dumps(report, indent=2))
            return 1

        entry = {
            "room_id_hex": f"0x{session.state.room_id:04X}",
            "samus_x": session.state.samus_x,
            "samus_y": session.state.samus_y,
            "pose": session.state.pose,
            "health": session.state.health,
            "missiles": session.state.missiles,
            "super_missiles": session.state.super_missiles,
            "selected_item": session.state.selected_item,
            "body_hp": session.state.enemy0_hp,
            "enemy0_x": session.state.enemy0_x,
            "enemy0_y": session.state.enemy0_y,
            "boss_bits_wrecked_ship": wrecked_ship_boss_bits(env),
            "items_hex": f"0x{session.state.collected_items:04X}",
            "boss_name": catalog.name,
            "max_body_hp": catalog.max_hp,
        }
        strategy = PhantoonStrategy(
            max_fight_frames=args.max_frames,
            weapon=WEAPON_SUPERS if args.weapon == "supers" else 1,
            fire_period=args.fire_period,
        )
        evidence = play_phantoon_fight(
            session,
            strategy=strategy,
            require_boss_bit=not args.body_only,
        )
        success = evidence.outcome == "phantoon_defeated" or (
            args.body_only and evidence.body_zero_frame is not None
        )
        out_path: Path | None = None
        if success and args.save_state is not False:
            out = (
                DEFAULT_OUT
                if args.save_state is True or args.save_state is None
                else Path(args.save_state)
            )
            # settle a few frames in ordinary gameplay
            for _ in range(30):
                session.step([0] * 12, "settle")
            save_dev_state(env, out)
            # Mirror to integration dev pin + task companion when using default.
            if out.resolve() == DEFAULT_OUT.resolve():
                save_dev_state(env, PHANTOON_DEFEATED_STATE)
                task_out = GAME_DIR / "tasks" / "ws_ship_human_post_phantoon.state"
                save_dev_state(env, task_out)
            out_path = out

        tel = assist.telemetry
        report = {
            "command": "strategy",
            "state": loaded,
            "success": success,
            "entry": entry,
            "fight": evidence.to_dict(),
            "assist": {
                "energy_restored": tel.energy.restored,
                "energy_writes": tel.energy.writes,
                "maximum_single_frame_damage": tel.maximum_single_frame_damage,
                "deaths": tel.deaths,
            },
            "final": {
                "room_id_hex": f"0x{session.state.room_id:04X}",
                "body_hp": session.state.enemy0_hp,
                "health": session.state.health,
                "samus_x": session.state.samus_x,
                "samus_y": session.state.samus_y,
                "boss_bits_wrecked_ship": wrecked_ship_boss_bits(env),
                "phantoon_defeated": phantoon_defeated(env),
                "items_hex": f"0x{session.state.collected_items:04X}",
            },
            "saved_state": str(out_path) if out_path is not None else None,
            "method": "full_knowledge_strategy",
            "developmentOnly": True,
            "notes": (
                "Policy-only Super spray from room entry. Death anim waits for "
                "WS boss bit 0. Not continuous evidence until natural ship compose."
            ),
        }
        text = json.dumps(report, indent=2)
        print(text)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        return 0 if success else 1
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("strategy", help="Run Super-spray from Phantoon-room entry")
    p.add_argument(
        "--state",
        default="ws_ship_human_end",
        help=(
            "ws_ship_human_end|entry|path "
            f"(default: {DEFAULT_ENTRY.name})"
        ),
    )
    p.add_argument("--max-frames", type=int, default=20_000)
    p.add_argument(
        "--weapon",
        choices=("supers", "missiles"),
        default="supers",
        help="Primary spray weapon (default: supers)",
    )
    p.add_argument("--fire-period", type=int, default=2)
    p.add_argument(
        "--body-only",
        action="store_true",
        help="Succeed on body HP 0 without waiting for boss bit",
    )
    p.add_argument(
        "--save-state",
        nargs="?",
        const=True,
        default=True,
        help=(
            f"Write defeated pin (default: {DEFAULT_OUT}). "
            "Pass a path for a custom out; --no-save-state to skip."
        ),
    )
    p.add_argument(
        "--no-save-state",
        action="store_true",
        help="Do not write a defeated save pin",
    )
    p.add_argument("--report", type=Path, default=None)
    p.set_defaults(func=cmd_strategy)

    argv = list(sys.argv[1:])
    known = {"strategy", "-h", "--help"}
    if not argv or argv[0] not in known:
        argv = ["strategy", *argv]

    args = parser.parse_args(argv)
    if getattr(args, "no_save_state", False):
        args.save_state = False
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
