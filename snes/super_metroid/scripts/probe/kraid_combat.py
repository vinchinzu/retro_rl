#!/usr/bin/env python3
"""Probe full-knowledge Kraid Super-spray strategy (policy only, no RL).

Starts from a Kraid-room entry state (doorway-natural preferred) and runs
the deterministic controller until body HP 0 + Brinstar boss bit 0, and
optionally rear-door exit + real Varia PLM collect.

```bash
# Doorway-natural entry (Warehouse→Hi-Jump→Kraid composed) — fight only
uv run python snes/super_metroid/scripts/probe/kraid_combat.py strategy \\
  --state entry

# Boss-only closeout: fight → rear door → Varia
uv run python snes/super_metroid/scripts/probe/kraid_combat.py varia --state entry

# Named KPDR entry anchor
uv run python snes/super_metroid/scripts/probe/kraid_combat.py strategy \\
  --state dev_kpdr_kraid_entry

# Mid-arena placed save (older dev probe)
uv run python snes/super_metroid/scripts/probe/kraid_combat.py strategy \\
  --state dev_kraid_room_natural
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
for _p in (ROOT, globals().get('_SNES_IMPORT_ROOT', ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from retro_harness.env import make_env, read_state_bytes  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.combat.features import kraid_catalog  # noqa: E402
from super_metroid.combat.kraid import (  # noqa: E402
    ROOM_KRAID,
    VARIA_MASK,
    KraidStrategy,
    body_hp,
    brinstar_boss_bits,
    play_kraid_fight,
    play_kraid_fight_to_varia,
)
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR, SCRATCH_STATE_DIR  # noqa: E402
from super_metroid.ram import parse_state  # noqa: E402

# Preferred doorway entry for KPDR K3 iteration (controller-composed).
DEFAULT_ENTRY_STATE = SCRATCH_STATE_DIR / "eye_hj_kraid_entry.state"
# Fallbacks checked in order when --state is a bare name.
_NAMED_STATES: dict[str, Path] = {
    "entry": DEFAULT_ENTRY_STATE,
    "eye": DEFAULT_ENTRY_STATE,
    "eye_hj": DEFAULT_ENTRY_STATE,
    "natural": DEFAULT_ENTRY_STATE,
    "composed": SCRATCH_STATE_DIR / "warehouse_hijump_kraid_composed.state",
    "dev_kpdr_kraid_entry": INTEGRATION_DIR / "dev_kpdr_kraid_entry.state",
    "dev_kraid_room_natural": INTEGRATION_DIR / "dev_kraid_room_natural.state",
    "dev_kraid_eye_at_eye": INTEGRATION_DIR / "dev_kraid_eye_at_eye.state",
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
            ):
                if candidate.exists():
                    return candidate
        return path
    # Named integration state without .state suffix.
    candidate = INTEGRATION_DIR / f"{key}.state"
    if candidate.exists():
        return candidate
    scratch = SCRATCH_STATE_DIR / f"{key}.state"
    if scratch.exists():
        return scratch
    return path


def _open_env(state_path: Path):
    if not state_path.exists():
        raise FileNotFoundError(
            f"Kraid entry state not found: {state_path}\n"
            "Compose with: kpdr.py pure warehouse-hijump-kraid "
            "(writes scratch/eye_hj_kraid_entry.state)"
        )
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.em.set_state(read_state_bytes(state_path))
    # Settle two frames so RAM is coherent after load.
    for _ in range(2):
        env.step([0] * 12)
    return env, str(state_path)


def cmd_strategy(args: argparse.Namespace) -> int:
    catalog = kraid_catalog()
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist()
    try:
        session = _Session(env, assist)
        if session.state.room_id != ROOM_KRAID:
            report = {
                "command": "strategy",
                "state": loaded,
                "success": False,
                "outcome": "wrong_room",
                "room_id_hex": f"0x{session.state.room_id:04X}",
                "notes": "Load a Kraid-room entry state (0xA59F), not Eye Door.",
            }
            print(json.dumps(report, indent=2))
            return 1

        entry = {
            "room_id_hex": f"0x{session.state.room_id:04X}",
            "samus_x": session.state.samus_x,
            "samus_y": session.state.samus_y,
            "pose": session.state.pose,
            "health": session.state.health,
            "max_health": session.state.max_health,
            "missiles": session.state.missiles,
            "super_missiles": session.state.super_missiles,
            "max_super_missiles": session.state.max_super_missiles,
            "selected_item": session.state.selected_item,
            "body_hp": body_hp(session.state),
            "enemy0_x": session.state.enemy0_x,
            "enemy0_y": session.state.enemy0_y,
            "num_enemies": session.state.num_enemies,
            "boss_bits_brinstar": brinstar_boss_bits(env),
            "items_hex": f"0x{session.state.collected_items:04X}",
            "boss_name": catalog.name,
            "max_body_hp": catalog.max_hp,
        }
        evidence = play_kraid_fight(
            session,
            strategy=KraidStrategy(max_fight_frames=args.max_frames),
            require_boss_bit=not args.body_only,
        )
        tel = assist.telemetry
        report = {
            "command": "strategy",
            "state": loaded,
            "success": evidence.outcome == "kraid_defeated"
            or (args.body_only and evidence.body_zero_frame is not None),
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
                "body_hp": body_hp(session.state),
                "health": session.state.health,
                "samus_x": session.state.samus_x,
                "samus_y": session.state.samus_y,
                "boss_bits_brinstar": brinstar_boss_bits(env),
            },
            "method": "full_knowledge_strategy",
            "developmentOnly": True,
            "notes": (
                "Policy-only Super spray from room entry. Not continuous "
                "evidence until composed on the power-on KPDR prefix."
            ),
        }
        text = json.dumps(report, indent=2)
        print(text)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        return 0 if report["success"] else 1
    finally:
        env.close()


def cmd_varia(args: argparse.Namespace) -> int:
    """Fight Kraid from doorway entry, rear-door exit, collect Varia PLM."""
    catalog = kraid_catalog()
    state_path = _resolve_state(args.state)
    env, loaded = _open_env(state_path)
    assist = UnlimitedResourcesAssist()
    try:
        session = _Session(env, assist)
        if session.state.room_id != ROOM_KRAID:
            report = {
                "command": "varia",
                "state": loaded,
                "success": False,
                "outcome": "wrong_room",
                "room_id_hex": f"0x{session.state.room_id:04X}",
                "notes": "Load a Kraid-room entry state (0xA59F).",
            }
            print(json.dumps(report, indent=2))
            return 1

        entry = {
            "room_id_hex": f"0x{session.state.room_id:04X}",
            "samus_x": session.state.samus_x,
            "samus_y": session.state.samus_y,
            "pose": session.state.pose,
            "health": session.state.health,
            "max_health": session.state.max_health,
            "super_missiles": session.state.super_missiles,
            "max_super_missiles": session.state.max_super_missiles,
            "selected_item": session.state.selected_item,
            "body_hp": body_hp(session.state),
            "boss_bits_brinstar": brinstar_boss_bits(env),
            "items_hex": f"0x{session.state.collected_items:04X}",
            "boss_name": catalog.name,
            "max_body_hp": catalog.max_hp,
        }
        evidence = play_kraid_fight_to_varia(
            session,
            strategy=KraidStrategy(max_fight_frames=args.max_frames),
        )
        tel = assist.telemetry
        success = bool(evidence.to_dict()["success"])
        report = {
            "command": "varia",
            "state": loaded,
            "success": success,
            "entry": entry,
            "fight": evidence.fight.to_dict(),
            "varia": evidence.varia.to_dict(),
            "assist": {
                "energy_restored": tel.energy.restored,
                "energy_writes": tel.energy.writes,
                "maximum_single_frame_damage": tel.maximum_single_frame_damage,
                "deaths": tel.deaths,
            },
            "final": {
                "room_id_hex": f"0x{session.state.room_id:04X}",
                "body_hp": body_hp(session.state),
                "health": session.state.health,
                "samus_x": session.state.samus_x,
                "samus_y": session.state.samus_y,
                "items_hex": f"0x{session.state.collected_items:04X}",
                "varia_collected": bool(session.state.collected_items & VARIA_MASK),
                "boss_bits_brinstar": brinstar_boss_bits(env),
            },
            "method": "full_knowledge_strategy_to_varia",
            "developmentOnly": True,
            "notes": (
                "Boss-only closeout from doorway entry: Super spray → rear "
                "door → real Varia PLM. Not continuous evidence until composed "
                "on the power-on KPDR prefix after play_eye_to_kraid."
            ),
        }
        text = json.dumps(report, indent=2)
        print(text)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        if args.save_state is not None and success:
            args.save_state.parent.mkdir(parents=True, exist_ok=True)
            args.save_state.write_bytes(env.em.get_state())
        return 0 if success else 1
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_strategy = sub.add_parser(
        "strategy", help="Run Super-spray strategy from Kraid-room entry"
    )
    p_strategy.add_argument(
        "--state",
        default="entry",
        help=(
            "entry|composed|dev_kpdr_kraid_entry|dev_kraid_room_natural|path "
            f"(default: {DEFAULT_ENTRY_STATE.name})"
        ),
    )
    p_strategy.add_argument("--max-frames", type=int, default=15_000)
    p_strategy.add_argument(
        "--body-only",
        action="store_true",
        help="Succeed on body HP 0 without waiting for boss bit",
    )
    p_strategy.add_argument("--report", type=Path, default=None)
    p_strategy.set_defaults(func=cmd_strategy)

    p_varia = sub.add_parser(
        "varia",
        help="Fight Kraid, rear-door exit, collect Varia PLM (boss-only closeout)",
    )
    p_varia.add_argument(
        "--state",
        default="entry",
        help=f"entry state name or path (default: {DEFAULT_ENTRY_STATE.name})",
    )
    p_varia.add_argument("--max-frames", type=int, default=15_000)
    p_varia.add_argument("--report", type=Path, default=None)
    p_varia.add_argument(
        "--save-state",
        type=Path,
        default=None,
        help="Write post-Varia emulator state on success",
    )
    p_varia.set_defaults(func=cmd_varia)

    argv = list(sys.argv[1:])
    known = {"strategy", "varia", "-h", "--help"}
    if not argv or argv[0] not in known:
        argv = ["strategy", *argv]

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
