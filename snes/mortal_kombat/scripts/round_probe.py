#!/usr/bin/env python3
"""Round-1 / between-round RAM probe with roster model swap.

Default: Fight_LiuKang + timeout-KO through Match 1, then watch Match 2
fight-ready and the Fight→Match2 specialist swap. Pixel CNNs are fallbacks.

    uv run --extra ml python snes/mortal_kombat/scripts/round_probe.py
    uv run --extra ml python snes/mortal_kombat/scripts/round_probe.py --play
    uv run python snes/mortal_kombat/scripts/round_probe.py --boot --force-rounds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT, _ROOT / "snes"):
    _t = str(_p)
    if _t not in sys.path:
        sys.path.insert(0, _t)

from retro_harness.env import make_env, reset_obs  # noqa: E402
from retro_harness.segment_runner import save_rgb_png  # noqa: E402
from mortal_kombat.boot import boot_to_fight  # noqa: E402
from mortal_kombat.paths import GAME_DIR, GAME_ID, RECORDINGS_DIR  # noqa: E402
from mortal_kombat.ram import (  # noqa: E402
    MAX_HEALTH,
    Screen,
    char_name,
    is_match_won,
    parse_ram,
)
from mortal_kombat.compat import install_fighters_common_alias  # noqa: E402
from mortal_kombat.tournament import TournamentRunner  # noqa: E402


def _poke_timeout_ko(env) -> None:
    """Timeout-win a round. Direct HP=0 does not trip SNES MK1 KO logic."""
    data = getattr(env.unwrapped, "data", None)
    if data is None or not hasattr(data, "set_value"):
        return
    data.set_value("health", MAX_HEALTH)
    data.set_value("enemy_health", 1)
    data.set_value("timer", 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="Fight_LiuKang")
    parser.add_argument("--boot", action="store_true", help="Power-on instead of --state")
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument(
        "--force-rounds",
        action="store_true",
        default=True,
        help="Timeout-KO until --force-wins (default on)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Do not poke RAM; let the roster policy play",
    )
    parser.add_argument("--force-wins", type=int, default=1)
    parser.add_argument(
        "--until-match",
        type=int,
        default=1,
        help="Stop once match_counter reaches this in FIGHT (1 = Match 2)",
    )
    args = parser.parse_args()
    if args.play:
        args.force_rounds = False
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    install_fighters_common_alias()

    force_wins = 0
    last_won = False

    def on_frame(env, frame, snap, prev) -> bool | None:
        nonlocal force_wins, last_won
        won = is_match_won(snap)
        if won and not last_won:
            force_wins += 1
        last_won = won
        if (
            args.force_rounds
            and force_wins < args.force_wins
            and snap.screen is Screen.FIGHT
            and snap.timer > 8
        ):
            # Poke while the round is live, then let timer 1 expire (do not
            # hold it at 1 every frame or the round never ends).
            _poke_timeout_ko(env)
        if (
            snap.match_counter >= args.until_match
            and snap.screen is Screen.FIGHT
            and snap.timer > 50
            and snap.p1_health == MAX_HEALTH
        ):
            return True
        return None

    state = "NONE" if args.boot else args.state
    env = make_env(GAME_ID, state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        reset_obs(env)
        if args.boot:
            boot_to_fight(env)
        runner = TournamentRunner(on_frame=on_frame, deterministic=False)
        print(f"slots={[f'{s.prefix}:{s.kind}:{s.model}' for s in runner.slots]}")
        result = runner.run_on(env, max_frames=args.max_frames)
        events = result.events
        print(
            f"cleared={result.cleared} furthest={result.furthest} "
            f"wins={result.wins} losses={result.losses} frames={result.frames} "
            f"force_wins={force_wins}"
        )
        print("events:")
        for event in events:
            extra = f" swap={event.swap}" if event.swap else ""
            print(
                f"  f={event.frame:5d} {event.screen:<14} "
                f"match={event.match} vs={char_name(event.p2):<12} "
                f"rounds={event.p1_rounds}-{event.p2_rounds} "
                f"hp={event.hp[0]}/{event.hp[1]} t={event.timer}{extra}"
            )
        print("swaps:")
        for item in result.swaps:
            print(f"  {item}")
        payload = {
            "cleared": result.cleared,
            "furthest": result.furthest,
            "wins": result.wins,
            "losses": result.losses,
            "frames": result.frames,
            "force_wins": force_wins,
            "swaps": result.swaps,
            "events": [event.__dict__ for event in events],
        }
        out = RECORDINGS_DIR / "round_probe.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        frame = env.render()
        png = save_rgb_png(frame, RECORDINGS_DIR / "round_probe.png")
        print(f"wrote {out} png={png}")
        saw_between = any(e.screen == "BETWEEN_ROUNDS" for e in events)
        saw_swap = any(item.startswith("fight:") for item in result.swaps)
        saw_next = any(
            e.match >= args.until_match and e.screen == "FIGHT" for e in events
        )
        print(
            f"CHECK between_rounds={saw_between} model_swap={saw_swap} "
            f"next_match_fight={saw_next}"
        )
        return 0 if saw_between and saw_swap else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
