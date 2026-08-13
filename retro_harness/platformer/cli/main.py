"""CLI entry point: argparse setup and command dispatch."""

from __future__ import annotations

import argparse
import sys

# Trigger level registration on import
import retro_harness.platformer.levels  # noqa: F401

from retro_harness.platformer.cli.helpers import _parse_room_id_arg
from retro_harness.platformer.cli.optimize import (
    cmd_analyze_seed,
    cmd_hillclimb,
    cmd_hillclimb_raw,
    cmd_neuro,
    cmd_optimize,
    cmd_segment_hillclimb,
    cmd_trim_seed,
)
from retro_harness.platformer.cli.play import cmd_play
from retro_harness.platformer.cli.practice import cmd_practice
from retro_harness.platformer.cli.selftest import cmd_selftest
from retro_harness.platformer.cli.watch import (
    cmd_auto_state,
    cmd_chain,
    cmd_chain_live,
    cmd_chain_optimize,
    cmd_chain_video,
    cmd_extract,
    cmd_extract_all,
    cmd_list_levels,
    cmd_list_routes,
    cmd_prepare_seeds,
    cmd_trace_map,
    cmd_verify,
    cmd_watch,
    cmd_watch_bk2,
)



def _add_frame_save_args(parser: argparse.ArgumentParser, *, raw_only: bool = False) -> None:
    """Shared frame-save / window flags for hillclimb and hillclimb-raw."""
    if raw_only:
        window_help = "Raw path only: mutate START:END (segment engine)"
        prefer_help = "Raw path: force frame-deletion bias"
        no_prefer_help = "Raw path: disable auto frame-deletion bias"
        require_help = "Raw path: force completion gating"
        allow_help = "Raw path: allow non-completing acceptances"
    else:
        window_help = (
            "Only mutate START:END (enables checkpoint-accelerated segment engine)"
        )
        prefer_help = (
            "Force frame-deletion / hold shortening bias "
            "(default: auto if seed clears)"
        )
        no_prefer_help = "Disable auto frame-deletion bias"
        require_help = "Force completion gating (default: auto if seed clears)"
        allow_help = "Allow accepting non-completing candidates"

    parser.add_argument("--window", help=window_help)
    parser.add_argument("--prefer-trim", action="store_true", help=prefer_help)
    parser.add_argument("--no-prefer-trim", action="store_true", help=no_prefer_help)
    parser.add_argument(
        "--require-completion", action="store_true", help=require_help
    )
    parser.add_argument("--allow-incomplete", action="store_true", help=allow_help)


def main(default_level: str | None = None) -> None:
    """Build and run the CLI parser.

    Args:
        default_level: If set, use this level when --level is omitted.
            Used by game-specific wrappers (e.g., DKC optimizer).
    """
    parser = argparse.ArgumentParser(
        description="Platformer Speedrun Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global --level flag
    level_required = default_level is None
    parser.add_argument(
        "--level", "-l",
        default=default_level,
        required=False,
        help=f"Level ID or alias (default: {default_level or 'required'})",
    )

    sub = parser.add_subparsers(dest="command")

    # list-levels
    sub.add_parser("list-levels", help="List all registered levels")

    # extract
    p_extract = sub.add_parser("extract", help="Extract actions from a bk2 recording")
    p_extract.add_argument("--bk2", required=True, help="Path to bk2 file")
    p_extract.add_argument("--output", "-o", help="Output JSON path")
    p_extract.add_argument("--raw-preview", action="store_true")

    # extract-all
    p_extract_all = sub.add_parser("extract-all", help="Extract and evaluate all bk2 recordings")
    p_extract_all.add_argument("--recordings-dir", help="Recordings directory")

    # verify
    p_verify = sub.add_parser("verify", help="Verify action sequence via headless replay")
    p_verify.add_argument("--actions", required=True, help="Path to actions JSON")
    p_verify.add_argument("--trace", action="store_true", help="Log all level_id changes")
    p_verify.add_argument("--state", help="Override start state")

    # optimize
    p_optimize = sub.add_parser("optimize", help="Run GA optimization")
    p_optimize.add_argument("--seed", help="Path to seed actions JSON")
    p_optimize.add_argument("--seeds-dir", help="Directory of recordings to use as multi-seed (mutually exclusive with --seed)")
    p_optimize.add_argument("--min-frames", type=int, default=60, help="Skip seeds shorter than N frames (default: 60)")
    p_optimize.add_argument("--raw", action="store_true", help="Use raw-button GA (no lossy action-index conversion)")
    p_optimize.add_argument("--generations", type=int, default=None)
    p_optimize.add_argument("--population", type=int, default=None)
    p_optimize.add_argument("--output-dir", help="Output directory")
    p_optimize.add_argument("--workers", type=int, default=1, help="Parallel workers")
    p_optimize.add_argument("--resume", help="Resume from checkpoint JSON")
    p_optimize.add_argument("--render", type=int, nargs="?", const=1, default=0,
                            metavar="N", help="Render best every N gens (default: every gen)")
    p_optimize.add_argument("--state", help="Override start state")

    # hillclimb — routes to raw-button path when seed has raw_buttons
    p_hill = sub.add_parser(
        "hillclimb",
        help="Hill climb a seed (raw-button path when raw_buttons present)",
    )
    p_hill.add_argument("--seed", required=True, help="Path to seed actions JSON")
    p_hill.add_argument("--iterations", type=int, default=2000)
    p_hill.add_argument("--output-dir", help="Output directory")
    p_hill.add_argument("--render", type=int, nargs="?", const=100, default=0,
                        metavar="N", help="Render best every N iterations (default: every 100)")
    p_hill.add_argument("--scale", type=int, default=3, help="Render scale")
    p_hill.add_argument("--state", help="Override start state")
    p_hill.add_argument(
        "--force-index",
        action="store_true",
        help="Mutate action-table indices even when raw_buttons exist (lossy)",
    )
    _add_frame_save_args(p_hill, raw_only=True)

    # hillclimb-raw (raw button mutation, no lossy action-index conversion)
    p_hraw = sub.add_parser("hillclimb-raw", help="Hill climb with raw button mutation")
    p_hraw.add_argument("--seed", required=True, help="Path to seed JSON with raw_buttons")
    p_hraw.add_argument("--iterations", type=int, default=2000)
    p_hraw.add_argument("--output-dir", help="Output directory")
    p_hraw.add_argument("--state", help="Override start state")
    _add_frame_save_args(p_hraw)

    # analyze-seed (static + optional live eval)
    p_an = sub.add_parser(
        "analyze-seed",
        help="Report leading idle, hold stalls, optional clear-frame eval",
    )
    p_an.add_argument("--seed", required=True, help="Path to raw_buttons / nes9_rle seed")
    p_an.add_argument("--state", help="Override start state")
    p_an.add_argument(
        "--static-only",
        action="store_true",
        help="Skip emulator eval (idle/hold stats only)",
    )
    p_an.add_argument("--output", "-o", help="Write JSON report path")

    # trim-seed (deterministic frame-saving transforms)
    p_trim = sub.add_parser(
        "trim-seed",
        help="Trim leading idle, compress holds, drop post-clear pad",
    )
    p_trim.add_argument("--seed", required=True, help="Path to raw_buttons / nes9_rle seed")
    p_trim.add_argument("--state", help="Override start state")
    p_trim.add_argument("--output", "-o", help="Output JSON path")
    p_trim.add_argument("--output-dir", help="Output directory (if --output omitted)")
    p_trim.add_argument(
        "--parity",
        choices=("any", "even", "odd"),
        default="any",
        help="Leading-idle trim parity (SMB 1-1 needs even)",
    )
    p_trim.add_argument("--step", type=int, default=1, help="Leading-trim search step")
    p_trim.add_argument("--max-leading", type=int, default=None, help="Cap leading idle trim")
    p_trim.add_argument("--pad", type=int, default=30, help="Idle frames kept after clear")
    p_trim.add_argument("--no-leading", action="store_true", help="Skip leading-idle search")
    p_trim.add_argument("--no-trailing", action="store_true", help="Skip post-clear trim")
    p_trim.add_argument(
        "--holds",
        action="store_true",
        help="Also binary-search compress long identical-button holds",
    )
    p_trim.add_argument("--min-hold", type=int, default=30, help="Min hold length to probe")
    p_trim.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Accept non-completing candidates (default: require completion)",
    )

    # segment-hillclimb (windowed + checkpoint)
    p_seg = sub.add_parser(
        "segment-hillclimb",
        help="Checkpoint-accelerated hillclimb inside a frame window",
    )
    p_seg.add_argument("--seed", required=True, help="Path to raw_buttons / nes9_rle seed")
    p_seg.add_argument(
        "--window",
        required=True,
        help="Mutable frame range START:END (prefix is checkpointed)",
    )
    p_seg.add_argument("--iterations", type=int, default=1000)
    p_seg.add_argument("--output-dir", help="Output directory")
    p_seg.add_argument("--state", help="Override start state")
    p_seg.add_argument(
        "--no-prefer-trim",
        action="store_true",
        help="Disable delete/hold-trim mutation bias",
    )
    p_seg.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow non-completing fitness improvements",
    )

    # watch
    p_watch = sub.add_parser("watch", help="Watch action sequence visually")
    p_watch.add_argument("--actions", required=True, help="Path to actions JSON")
    p_watch.add_argument("--scale", type=int, default=3)
    p_watch.add_argument("--state", help="Override start state")

    # watch-bk2
    p_watch_bk2 = sub.add_parser("watch-bk2", help="Replay a bk2 recording visually")
    p_watch_bk2.add_argument("--bk2", required=True, help="Path to bk2 file")
    p_watch_bk2.add_argument("--scale", type=int, default=3)

    # prepare-seeds
    p_seeds = sub.add_parser("prepare-seeds", help="Extract and rank recordings, save top N as seeds")
    p_seeds.add_argument("--recordings-dir", help="Recordings directory")
    p_seeds.add_argument("--top", type=int, default=5, help="Number of top seeds to save")

    # auto-state
    p_auto = sub.add_parser("auto-state", help="Create save state via scripted navigation")
    p_auto.add_argument("--from-state", required=True, help="Starting state name")
    p_auto.add_argument("--nav", required=True, help="Navigation steps: 'BUTTON:hold:wait ...'")
    p_auto.add_argument("--settle", type=int, default=30, help="Extra NOOP frames after nav (default: 30)")
    p_auto.add_argument("--screenshot", action="store_true", help="Save screenshot for verification")

    # practice (auto-reset on death, saves all attempts)
    p_practice = sub.add_parser("practice", help="Practice with auto-reset on death, saving all attempts")
    p_practice.add_argument("--scale", type=int, default=3)
    p_practice.add_argument("--state", help="Override start state")
    p_practice.add_argument("--save-name", help="Name for F5 state save (default: Chained_{level}_practice)")
    p_practice.add_argument("--output-dir", help="Directory for attempt JSON and summary files")
    p_practice.add_argument("--session-label", help="Session label stored in metadata and shown in the HUD")
    p_practice.add_argument(
        "--continue",
        "--keep-playing",
        dest="keep_playing",
        action="store_true",
        help="Continue recording across configured segment-completion rooms",
    )
    p_practice.add_argument(
        "--until-room",
        type=_parse_room_id_arg,
        help="End the continuous attempt at this decimal or hexadecimal room ID",
    )
    p_practice.add_argument(
        "--room-debounce",
        type=int,
        default=3,
        help="Stable frames required to confirm a room split (default: 3)",
    )
    p_practice.add_argument(
        "--until-playable",
        action="store_true",
        help="At --until-room, wait for game_state 8 and no door transition",
    )
    p_practice.add_argument(
        "--until-label",
        help="Optional display label stored with the target-room recording",
    )

    # play (record)
    p_play = sub.add_parser("play", help="Play a level manually and record inputs")
    p_play.add_argument("--scale", type=int, default=3)
    p_play.add_argument("--state", help="Override start state (e.g. ResumeRun)")

    # selftest
    sub.add_parser("selftest", help="Run self-tests")

    # list-routes
    sub.add_parser("list-routes", help="List all registered speedrun routes")

    # chain (evaluate a full route)
    p_chain = sub.add_parser("chain", help="Evaluate a full speedrun route")
    p_chain.add_argument("--route", "-r", required=True, help="Route ID or alias")

    # chain-live (true end-to-end on single emulator)
    p_clive = sub.add_parser("chain-live", help="True end-to-end chain on single emulator (no state reloads)")
    p_clive.add_argument("--route", "-r", required=True, help="Route ID or alias")
    p_clive.add_argument("--save-states", action="store_true", help="Save chained states at each segment boundary")
    p_clive.add_argument("--video", help="Output MP4 path (optional)")
    p_clive.add_argument("--scale", type=int, default=3, help="Video pixel scale (default 3)")

    # chain-optimize (iterative hill climb from chained states)
    p_copt = sub.add_parser("chain-optimize", help="Iteratively hill-climb segments from chained states")
    p_copt.add_argument("--route", "-r", required=True, help="Route ID or alias")
    p_copt.add_argument("--iterations", type=int, default=2000, help="Hill climb iterations per segment (default 2000)")

    # chain-video (render full route to MP4)
    p_cvid = sub.add_parser("chain-video", help="Render a full speedrun route to MP4")
    p_cvid.add_argument("--route", "-r", required=True, help="Route ID or alias")
    p_cvid.add_argument("--output", "-o", help="Output MP4 path")
    p_cvid.add_argument("--scale", type=int, default=3, help="Pixel scale (default 3)")

    # trace-map
    p_trace = sub.add_parser("trace-map", help="Render position trace on area map")
    p_trace.add_argument("--trace", help="Path to trace JSON (auto-detected if omitted)")
    p_trace.add_argument("--actions", help="Actions file (to find trace alongside it)")
    p_trace.add_argument("--area", help="Area name: crateria, brinstar, norfair, etc.")
    p_trace.add_argument("-o", "--output", help="Output PNG path")
    p_trace.add_argument("--map-dir", help="Override map PNG directory")

    # neuro (neuroevolution)
    p_neuro = sub.add_parser("neuro", help="Neuroevolution optimizer (evolve neural networks)")
    p_neuro.add_argument("--state", help="Override start state")
    p_neuro.add_argument("--population", type=int, default=100, help="Population size (default 100)")
    p_neuro.add_argument("--generations", type=int, default=300, help="Number of generations (default 300)")
    p_neuro.add_argument("--hidden", type=int, default=20, help="Hidden layer size (default 20)")
    p_neuro.add_argument("--max-frames", type=int, default=6000, help="Max frames per evaluation (default 6000)")
    p_neuro.add_argument("--output-dir", help="Output directory (default: runs_dir/neuro)")
    p_neuro.add_argument("--render", action="store_true", help="Render best network live each generation")
    p_neuro.add_argument("--scale", type=int, default=3, help="Render pixel scale (default 3)")
    p_neuro.add_argument(
        "--entry-corpus",
        help="EntryStateCorpus manifest; training consumes its train split only",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Validate --level is provided for commands that need it
    needs_level = args.command not in (
        "list-levels",
        "list-routes",
        "chain",
        "chain-live",
        "chain-optimize",
        "chain-video",
    )
    # analyze-seed can run static-only without a level
    if args.command == "analyze-seed" and getattr(args, "static_only", False):
        needs_level = False
    if needs_level and not args.level:
        parser.error(f"--level is required for '{args.command}'. Use 'list-levels' to see available levels.")

    commands = {
        "list-levels": cmd_list_levels,
        "extract": cmd_extract,
        "extract-all": cmd_extract_all,
        "verify": cmd_verify,
        "optimize": cmd_optimize,
        "hillclimb": cmd_hillclimb,
        "hillclimb-raw": cmd_hillclimb_raw,
        "analyze-seed": cmd_analyze_seed,
        "trim-seed": cmd_trim_seed,
        "segment-hillclimb": cmd_segment_hillclimb,
        "watch": cmd_watch,
        "watch-bk2": cmd_watch_bk2,
        "prepare-seeds": cmd_prepare_seeds,
        "auto-state": cmd_auto_state,
        "practice": cmd_practice,
        "play": cmd_play,
        "selftest": cmd_selftest,
        "trace-map": cmd_trace_map,
        "list-routes": cmd_list_routes,
        "chain": cmd_chain,
        "chain-live": cmd_chain_live,
        "chain-optimize": cmd_chain_optimize,
        "chain-video": cmd_chain_video,
        "neuro": cmd_neuro,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
