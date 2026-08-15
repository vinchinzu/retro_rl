"""Assisted Level 2 complete compose → ``ADDR_TRIFORCE & 0x02`` (rr-5dk).

Primary green tip (2026-08-07): continuous post-boom chain from
``Level2Boom`` through Dodongo + TF south-band maze. Delegates to
``run_level2_dodongo.run_once`` (same env session, no mid-run state reload).

Not Clean STATUS. Prefer ``--infinite-life``. Full Entrance→TF multi-stage
compose is PARTIAL / deferred (isolated pure segments already green).

Examples::

    uv run python nes/zelda_i/scripts/run_level2_complete.py --infinite-life --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level2_complete.py --from-state Level2Boom --tag l2_complete_assisted
    uv run python nes/zelda_i/scripts/run_level2_complete.py --from-state Level2_0E --trials 1
"""

from __future__ import annotations

import argparse

from retro_harness.segment_runner import write_json_report
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.runner import add_video_args, resolve_video
from zelda_i.scripts.run_level2_dodongo import run_once

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level2Boom")
    p.add_argument("--infinite-life", action="store_true", default=True)
    p.add_argument("--no-infinite-life", action="store_true")
    p.add_argument("--tag", default="l2_complete_assisted")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--save-state", action="store_true")
    p.add_argument(
        "--poke-bombs",
        action="store_true",
        help="RECON bomb inventory poke (default off)",
    )
    add_video_args(p)
    args = p.parse_args()
    inf = not args.no_infinite_life
    video_path, video_config, intro_frames = resolve_video(
        args,
        default_path=RECORDINGS_DIR / f"{args.tag}.mp4",
    )
    results = []
    for t in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{t}"
        trial_video = video_path
        if video_path is not None and args.trials > 1:
            trial_video = video_path.with_name(
                f"{video_path.stem}_t{t}{video_path.suffix}"
            )
        r = run_once(
            start_state=args.from_state,
            infinite_life=inf,
            tag=tag,
            save_checkpoint=args.save_state and t == 0,
            poke=args.poke_bombs,
            video_path=trial_video,
            video_config=video_config,
            intro_frames=intro_frames,
        )
        results.append(r)
        print(
            f"trial{t}: result={r.get('result')} ok={r.get('ok')} "
            f"tf={r.get('triforce')} final_sc={(r.get('final') or {}).get('sc')}"
        )
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"summary: {n_ok}/{len(results)} TF 0x02 (assisted L2 complete)")
    payload = {
        "bead": "rr-5dk",
        "result": "TF_02_LIVE_ASSISTED" if n_ok == len(results) and n_ok else "PARTIAL",
        "ok": n_ok == len(results) and n_ok > 0,
        "ok_count": n_ok,
        "trials": len(results),
        "track": "assisted" if inf else "clean",
        "intervention_class": "survival" if inf else "clean",
        "status_promote": False,
        "natural_entry": False,
        "start_state": args.from_state,
        "compose_scope": f"{args.from_state} → TF 0x02 continuous",
        "compose_partial_note": (
            "Primary green: Level2Boom → TF. Entrance/Compass continuous "
            "compose deferred; use isolated pure runners for earlier rooms."
        ),
        "triforce_bit_0x02": n_ok > 0,
        "poke": args.poke_bombs,
        "checkpoint": next((r.get("checkpoint") for r in results if r.get("checkpoint")), None),
        "runner": "nes/zelda_i/scripts/run_level2_complete.py",
        "delegate": "run_level2_dodongo.run_once",
        "assist": next((r.get("assist") for r in results if r.get("assist")), None),
        "video": next((r.get("video") for r in results if r.get("video")), None),
        "trial_results": [
            {
                "trial": i,
                "ok": r.get("ok"),
                "result": r.get("result"),
                "triforce": r.get("triforce"),
                "tf_policy_live": r.get("tf_policy_live"),
                "final": r.get("final"),
                "checkpoint": r.get("checkpoint"),
                "assist": r.get("assist"),
                "video": r.get("video"),
            }
            for i, r in enumerate(results)
        ],
        "evidence": [
            f"recordings/{args.tag}.json",
            f"recordings/{args.tag}_summary.json",
        ],
    }
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_report(RECORDINGS_DIR / f"{args.tag}_summary.json", payload)

if __name__ == "__main__":
    main()
