"""Stub: 0x0d post-boss TF collect graduated into ``level2_boss_path``.

Historical recon (rr-n5i): after Dodongo kill doors LEFT-only → 0x0d west of
boss. Collect is south-band maze (not north green sprite). LIVE waypoints in
``level2_puzzles.POST_BOSS_TF_POLICY`` / ``Level2PostBossTfController``.

Prefer the assisted runner::

    uv run python nes/zelda_i/scripts/run_level2_dodongo.py --infinite-life --from-state Level2_0E
    uv run python nes/zelda_i/scripts/run_level2_dodongo.py --infinite-life --from-state Level2_0D_PostBoss
    uv run python nes/zelda_i/scripts/run_level2_complete.py --infinite-life --trials 2

Library: ``zelda_i.level2_boss_path`` (collect_and_tf, make_post_boss_tf_controller).
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = _REPO_ROOT / "nes"
for _p in (_REPO_ROOT, _NES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from zelda_i.level2_boss_path import (
    LEVEL2_TF_BIT,
    ROOM_0E,
    ROOM_TF,
    default_tf_waypoints,
    load_tf_policy,
    make_post_boss_tf_controller,
)
from zelda_i.level2_puzzles import POST_BOSS_TF_POLICY
from zelda_i.scripts.run_level2_dodongo import run_once


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level2_0E")
    p.add_argument("--infinite-life", action="store_true", default=True)
    p.add_argument("--no-infinite-life", action="store_true")
    p.add_argument("--tag", default="l2_0d_tf")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--save-state", action="store_true")
    p.add_argument(
        "--policy-only",
        action="store_true",
        help="Print LIVE TF policy / controller defaults without emulator",
    )
    args = p.parse_args()

    if args.policy_only:
        pol = load_tf_policy()
        ctrl = make_post_boss_tf_controller(policy=pol)
        print(
            f"boss=0x{ROOM_0E:02x} tf_room=0x{ROOM_TF:02x} bit=0x{LEVEL2_TF_BIT:02x}"
        )
        print(f"policy_source={pol.get('source')} live={pol.get('live')}")
        print(f"waypoints={default_tf_waypoints()}")
        print(f"catalog_live={POST_BOSS_TF_POLICY.live}")
        print(f"ctrl_phase={ctrl.phase.name} n_wp={len(ctrl.waypoints)}")
        print("delegate: run_level2_dodongo / level2_boss_path")
        return

    inf = not args.no_infinite_life
    for t in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{t}"
        r = run_once(
            start_state=args.from_state,
            infinite_life=inf,
            tag=tag,
            save_checkpoint=args.save_state and t == 0,
        )
        print(
            f"trial{t}: result={r.get('result')} ok={r.get('ok')} "
            f"tf={r.get('triforce')} sc={(r.get('final') or {}).get('sc')}"
        )


if __name__ == "__main__":
    main()
