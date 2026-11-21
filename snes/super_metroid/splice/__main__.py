"""Splice planning and the explicit ``chain`` ROM search/replay command."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_SNES = _ROOT / "snes"
for _p in (_ROOT, _SNES):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from retro_harness.repo import ensure_import_paths  # noqa: E402

ensure_import_paths(root=_ROOT)

from super_metroid.human_tape.product_chain import DEFAULT_BOARD, DEFAULT_TASK  # noqa: E402
from super_metroid.paths import CLEAN_BASELINE_STEM, CLEAN_MAIN_STEM  # noqa: E402
from super_metroid.splice.cards import format_cards, generate_cards  # noqa: E402
from super_metroid.splice.errors import (  # noqa: E402
    AssembleError,
    GradeError,
    PrepareError,
    PreflightError,
    SchemaError,
    SpliceError,
)
from super_metroid.splice.manifest import load_manifest, manifest_from_product_chain  # noqa: E402
from super_metroid.splice.preflight import (  # noqa: E402
    format_preflight_summary,
    run_preflight,
)
from super_metroid.splice.prepare import prepare  # noqa: E402
from super_metroid.splice.credits import (  # noqa: E402
    assemble_credits,
    credits_chain,
    format_credits,
)
from super_metroid.splice.ranges import (  # noqa: E402
    assemble_attic_to_gravity,
    attic_to_gravity_range,
    format_range,
)
from super_metroid.splice.schema import INTERVENTION_PROFILES  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m super_metroid.splice",
        description=(
            "Planning/verification and Clean chain search over tips.play_hops. "
            "Artifact digest preflight, read-only task cards, fail-closed "
            "prepare, grade, assemble, the Attic→Gravity Scaffold range, "
            "and the development-only Scaffold credits chain "
            "(refuses to boot without a session factory)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    chain = sub.add_parser("chain", help="Search room tapes from their real predecessor, then replay from power-on")
    chain.add_argument("--board", type=Path, default=DEFAULT_BOARD.parents[1] / "recordings" / "chain_board_clean.json")
    chain.add_argument("--stem", type=Path, help="Tape stem (replay defaults to Grapple Skip; search defaults to the original chain)")
    chain.add_argument("--through-row", type=int, help="Require the replay to reach this board row")
    chain.add_argument("--require-credits", action="store_true", help="Require a verified end-of-credits replay")
    chain.add_argument("--max-delay", type=int, default=12)
    chain.add_argument("--fresh", action="store_true", help="Start a new master tape instead of replaying the existing prefix")
    chain.add_argument("--rebuild-board", action="store_true", help="Rebuild candidate rows from the actual archived and Tourian traces")
    chain.add_argument("--replay", action="store_true", help="Only prove the saved master tape from power-on")
    chain.add_argument("--proof-json", action="store_true", help="Include the full replay proof and its ordered room transitions")
    chain.add_argument("--override", action="append", default=[], metavar="ROW=NPZ", help="Try a searched actions array for one board row")
    chain.add_argument("--from-row", type=int, help="Modify an accepted row: replay --source-stem rows before this one from power-on, then re-solve into a new --stem")
    chain.add_argument("--source-stem", type=Path, default=CLEAN_BASELINE_STEM, help="Master tape that --from-row cuts")
    pre = sub.add_parser(
        "preflight",
        help="Snapshot artifact digests and report gaps (default: report-only)",
    )
    pre.add_argument("--task", type=Path, default=DEFAULT_TASK)
    pre.add_argument("--out", type=Path, default=DEFAULT_BOARD)
    pre.add_argument(
        "--write",
        action="store_true",
        help="Write the hop board with rewritten paths (never host-absolute)",
    )
    pre.add_argument("--json", action="store_true", help="Print JSON report only")
    pre.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if a selected product-chain artifact is missing",
    )
    pre.add_argument("--no-live", action="store_true", help="Skip the live task tape")
    cards = sub.add_parser(
        "cards",
        help="Print immutable task cards from a route manifest (read-only)",
    )
    cards.add_argument("--task", help="Print one TASK_ID")
    cards.add_argument("--json", action="store_true", help="Print JSON cards only")
    cards.add_argument("--manifest", type=Path, help="Route-manifest JSON (skip board adapter)")
    cards.add_argument(
        "--profile",
        default="scaffold",
        choices=INTERVENTION_PROFILES,
        help="Planner selection profile (default: scaffold)",
    )
    cards.add_argument(
        "--chain",
        type=Path,
        default=DEFAULT_TASK,
        help="Product-chain task used when --manifest is omitted",
    )
    cards.add_argument("--no-live", action="store_true", help="Skip the live task tape")
    prep = sub.add_parser(
        "prepare",
        help="Validate a task card before boot (report-only JSON; no emulator)",
    )
    prep.add_argument("task_id")
    prep.add_argument("--manifest", type=Path, help="Route-manifest JSON (skip board adapter)")
    prep.add_argument(
        "--profile",
        default="scaffold",
        choices=INTERVENTION_PROFILES,
        help="Planner selection profile (default: scaffold)",
    )
    prep.add_argument(
        "--chain",
        type=Path,
        default=DEFAULT_TASK,
        help="Product-chain task used when --manifest is omitted",
    )
    prep.add_argument("--no-live", action="store_true", help="Skip the live task tape")
    prep.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if the task cannot be prepared",
    )
    grd = sub.add_parser(
        "grade",
        help="Replay+Join grade (refuses to boot without a runner hook)",
    )
    grd.add_argument("task_id")
    grd.add_argument("candidate")
    grd.add_argument("--manifest", type=Path, help="Route-manifest JSON (skip board adapter)")
    grd.add_argument(
        "--profile",
        default="scaffold",
        choices=INTERVENTION_PROFILES,
        help="Planner selection profile (default: scaffold)",
    )
    asm = sub.add_parser(
        "assemble",
        help="Assemble a route (refuses to boot without a session factory)",
    )
    asm.add_argument("route_id")
    asm.add_argument("--manifest", type=Path, help="Route-manifest JSON (skip board adapter)")
    asm.add_argument(
        "--profile",
        default="scaffold",
        choices=INTERVENTION_PROFILES,
        help="Planner selection profile (default: scaffold)",
    )
    rng = sub.add_parser(
        "range",
        help="Print the Attic→Gravity Scaffold range (no emulator; development-only)",
    )
    rng.add_argument("--segment", type=Path, help="s23 directory (fail closed if missing)")
    rng.add_argument("--json", action="store_true", help="Print JSON plan only")
    asr = sub.add_parser(
        "assemble-range",
        help="Assemble Attic→Gravity Scaffold (refuses to boot without a session factory)",
    )
    asr.add_argument("--segment", type=Path, help="s23 directory (fail closed if missing)")
    crd = sub.add_parser("credits", help="Plan Scaffold credits or benchmark the live finish from a pin")
    crd.add_argument("--segment", type=Path, help="s23 directory (fail closed if missing)")
    crd.add_argument("--json", action="store_true", help="Print JSON plan/report only")
    crd.add_argument(
        "--dry",
        action="store_true",
        help="Print the chain plan; refuse to boot (default)",
    )
    crd.add_argument(
        "--boot",
        action="store_true",
        help="Assemble live (requires an explicit session factory; tests never pass this)",
    )
    crd.add_argument("--splice-state", type=Path, help="Boot a finish pin once and emit a timed JSON report")
    crd.add_argument("--entry-delay", type=int, default=0, help="Idle frames before the first live controller")
    crd.add_argument("--tape-out", type=Path, help="Save the verified splice buttons to a new NPZ")
    crd.add_argument("--compare", type=Path, help="Compare with a splice JSON report from the same pin and delay")

    bench = sub.add_parser(
        "benchmark",
        help="Compare a Clean master tape against the standard 0:56:28.88 KPDR splits and <55:00 goal",
    )
    bench.add_argument("--stem", type=Path, default=CLEAN_MAIN_STEM, help="Tape stem (default: Grapple Skip)")
    bench.add_argument("--board", type=Path, default=DEFAULT_BOARD.parents[1] / "recordings" / "chain_board_clean.json")
    bench.add_argument("--scan", action="store_true", help="Replay the master tape from power-on to extract exact RAM splits")
    bench.add_argument("--save-checkpoints", action="store_true", help="With --scan, save missing rowNNN_leave.state checkpoints")
    bench.add_argument("--overwrite-checkpoints", action="store_true", help="With --save-checkpoints, overwrite existing checkpoints")
    bench.add_argument("--record-history", action="store_true", help="Append result to recordings/speed_benchmarks/history.json")
    bench.add_argument("--label", default="snapshot", help="Label used when recording to benchmark history")
    bench.add_argument("--notes", default="", help="Optional notes for benchmark history")
    bench.add_argument("--top-hotspots", type=int, default=15)
    bench.add_argument("--json", action="store_true", help="Print JSON report only")
    bench.add_argument("--markdown-out", type=Path, help="Also write the Markdown benchmark report to this path")

    pol = sub.add_parser(
        "policy",
        help="List, A/B test, toggle, and build swappable route policies and live Skills",
    )
    pol_sub = pol.add_subparsers(dest="policy_cmd", required=True)
    pol_sub.add_parser("list", help="List available route profiles and registered Skills")
    pol_show = pol_sub.add_parser("show", help="Show a route policy and its rules")
    pol_show.add_argument("policy", help="Policy ID or JSON path")
    pol_show.add_argument("--enable", action="append", default=[], metavar="RULE_ID")
    pol_show.add_argument("--disable", action="append", default=[], metavar="RULE_ID")
    pol_show.add_argument("--use-tape", action="append", default=[], metavar="RULE_ID")
    pol_show.add_argument("--json", action="store_true")

    pol_ab = pol_sub.add_parser("ab", help="Run a local checkpoint A/B comparing one rule vs baseline tape")
    pol_ab.add_argument("policy", help="Policy ID or JSON path")
    pol_ab.add_argument("rule_id", help="Rule ID inside the policy")
    pol_ab.add_argument("--source-stem", type=Path, default=CLEAN_MAIN_STEM, help="Tape supplying the natural entry and comparison rows (default: Grapple Skip)")
    pol_ab.add_argument("--follow-rows", type=int, help="Override downstream rows verified after the rule")
    pol_ab.add_argument("--max-delay", type=int, default=24)
    pol_ab.add_argument("--save-npz", type=Path, help="Optional path to save candidate actions NPZ")
    pol_ab.add_argument("--json", action="store_true")

    pol_build = pol_sub.add_parser("build", help="Materialize and prove a full power-on run from a route policy")
    pol_build.add_argument("policy", help="Policy ID or JSON path")
    pol_build.add_argument("--source-stem", type=Path, help="Tape supplying untouched rows (defaults to Grapple Skip with --only, otherwise the original chain)")
    pol_build.add_argument("--prefix-stem", type=Path, help="Alternate tape through the predecessor of the first active rule")
    pol_build.add_argument("--only", action="append", default=[], metavar="RULE_ID", help="Apply only the named rules to the source tape (repeatable)")
    pol_build.add_argument("--stem", type=Path, required=True, help="New output _clean stem (never overwrites baseline)")
    pol_build.add_argument("--enable", action="append", default=[], metavar="RULE_ID")
    pol_build.add_argument("--disable", action="append", default=[], metavar="RULE_ID")
    pol_build.add_argument("--use-tape", action="append", default=[], metavar="RULE_ID")
    pol_build.add_argument("--max-delay", type=int, default=24)
    pol_build.add_argument("--through-row", type=int, help="Build and replay a prefix ending at this row")
    pol_build.add_argument("--no-checkpoint-seed", action="store_true", help="Replay untouched prefix from power-on during build instead of seeding from checkpoint")
    pol_build.add_argument("--skip-replay", action="store_true", help="Development only: skip the final loadless power-on replay proof")
    return parser


def _print_error(exc: SpliceError, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(exc.to_dict(), indent=2))
        return
    print(str(exc), file=sys.stderr)
    missing = exc.details.get("missing") or []
    for label in missing[:20]:
        print(f"  - {label}", file=sys.stderr)


def policy_coverage(rows: list[dict], tape: Any = None, bodies: dict[int, Any] | None = None) -> dict[str, Any]:
    """Master-tape frames by what produced their buttons.

    A milestone quotes this beside its proof so tape share is visible. With the
    committed ``tape`` and each row's human source slice in ``bodies`` a row
    counts as human only if its buttons equal that slice (a delayed or trimmed
    copy is ``human_retimed``). Ledger labels cannot be trusted for this: an
    override replaces the slice and the row still reads ``tape+idle:0``. Without
    them the labels decide and ``basis`` says so. Only ``controller`` rows are
    RAM-driven; every other kind is stored buttons.
    """
    import numpy as np

    frames = {"human_verbatim": 0, "human_retimed": 0, "searched_buttons": 0, "controller": 0}
    offset, measured = 0, tape is not None and bodies is not None
    for row in rows:
        n, method = int(row["frames"]), str(row.get("method", ""))
        got, body = (tape[offset : offset + n], bodies.get(row["row"])) if measured else (None, None)
        offset += n
        if method == "room-controller" or row.get("controller_id"):
            kind = "controller"
        elif got is not None and body is not None:
            trim = int(method.split(":")[1]) if method.startswith("tape-trim:") else 0
            delay = int(method.split(":")[1]) if method.startswith("tape+idle:") else 0
            keep = body[trim:]
            same = len(got) >= delay + len(keep) and not got[:delay].any() and np.array_equal(got[delay : delay + len(keep)], keep)
            kind = ("human_verbatim" if not (delay or trim) else "human_retimed") if same else "searched_buttons"
        elif method == "tape+idle:0":
            kind = "human_verbatim"
        elif method.startswith(("tape+idle", "tape-trim")):
            kind = "human_retimed"
        else:
            kind = "searched_buttons"
        frames[kind] += n
    total = sum(frames.values()) or 1
    out: dict[str, Any] = {kind: {"frames": n, "share": round(n / total, 3)} for kind, n in frames.items()}
    out["basis"] = "button_diff" if measured and bodies else "ledger_labels"
    return out


def _measured_coverage(stem: Path, board: Path, rows: list[dict]) -> dict[str, Any]:
    """Diff the committed tape against each row's human source slice."""
    import numpy as np
    from super_metroid.splice.assemble import _chain_body, _chain_rows

    try:
        tape, cache, bodies = np.load(stem.with_suffix(".npz"))["actions"], {}, {}
        for hop in _chain_rows(board):
            if hop.get("tape") and not hop.get("controller_id"):
                bodies[hop["row"]] = _chain_body(hop, cache)[0]
    except (OSError, ValueError, KeyError):
        return policy_coverage(rows)
    return policy_coverage(rows, tape, bodies)


def _run_preflight(args: argparse.Namespace) -> int:
    try:
        report = run_preflight(
            args.task,
            include_live=not args.no_live,
            write=bool(args.write),
            out=args.out,
            strict=bool(args.strict),
        )
    except PreflightError as exc:
        _print_error(exc, as_json=bool(args.json))
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_preflight_summary(report))
    return 0


def _run_cards(args: argparse.Namespace) -> int:
    try:
        if args.manifest is not None:
            manifest = load_manifest(args.manifest)
        else:
            manifest = manifest_from_product_chain(
                args.chain, include_live=not args.no_live
            )
        cards = generate_cards(manifest, profile=args.profile)
    except (PreflightError, SchemaError, OSError) as exc:
        if isinstance(exc, (PreflightError, SchemaError)):
            _print_error(exc, as_json=bool(args.json))
        else:
            print(str(exc), file=sys.stderr)
        return 1
    if args.task:
        cards = tuple(c for c in cards if c.task_id == args.task)
        if not cards:
            msg = f"task {args.task!r} not in manifest"
            if args.json:
                print(json.dumps({"error": "task_not_found", "message": msg}, indent=2))
            else:
                print(msg, file=sys.stderr)
            return 1
    if args.json:
        print(json.dumps([c.to_dict() for c in cards], indent=2))
    else:
        print(format_cards(cards))
    return 0


def _run_prepare(args: argparse.Namespace) -> int:
    try:
        prepared = prepare(
            args.task_id,
            manifest=args.manifest,
            profile=args.profile,
            chain=args.chain,
            include_live=not args.no_live,
        )
    except (PrepareError, SchemaError, OSError) as exc:
        if isinstance(exc, (PrepareError, SchemaError)):
            _print_error(exc, as_json=True)
        else:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2))
        if isinstance(exc, PrepareError):
            return 1 if args.strict else 0
        return 1
    print(json.dumps(prepared.to_dict(), indent=2))
    return 0


def _run_grade(args: argparse.Namespace) -> int:
    # Dry planner: never boots. A runner hook is required to replay.
    err = GradeError(
        "grade refuses to boot without a runner hook",
        code="grade.runner",
        details={"task_id": args.task_id, "candidate": args.candidate},
    )
    _print_error(err, as_json=True)
    return 1


def _run_assemble(args: argparse.Namespace) -> int:
    # Dry planner: never boots. A session factory is required to play_hops.
    err = AssembleError(
        "assemble refuses to boot without a session factory",
        code="assemble.session",
        details={"route_id": args.route_id, "profile": args.profile},
    )
    _print_error(err, as_json=True)
    return 1


def _run_range(args: argparse.Namespace) -> int:
    try:
        plan = attic_to_gravity_range(args.segment)
    except (PreflightError, AssembleError, SchemaError) as exc:
        _print_error(exc, as_json=bool(args.json))
        return 1
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(format_range(plan))
    return 0


def _run_assemble_range(args: argparse.Namespace) -> int:
    # Dry planner: load the range (fail closed on missing s23), never boot.
    try:
        assemble_attic_to_gravity(args.segment)
    except (PreflightError, AssembleError, SchemaError) as exc:
        _print_error(exc, as_json=True)
        return 1
    return 0


def _run_credits(args: argparse.Namespace) -> int:
    if args.splice_state is not None:
        from super_metroid.splice.credits import benchmark_finish_splice

        if args.segment or args.dry or args.boot:
            print("--splice-state is a separate mode from --segment, --dry and --boot", file=sys.stderr)
            return 2
        try:
            report = benchmark_finish_splice(
                args.splice_state, delay=args.entry_delay, tape_out=args.tape_out, compare=args.compare,
            )
        except (AssembleError, RuntimeError, ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2))
        return 0
    if args.entry_delay or args.tape_out or args.compare:
        print("--entry-delay, --tape-out and --compare require --splice-state", file=sys.stderr)
        return 2

    # Default dry: print the plan and refuse to boot. --boot still needs a
    # session factory (planning module, not a second runner).
    want_boot = bool(args.boot) and not bool(args.dry)
    try:
        if want_boot:
            # Still refuses without a session factory; CLI does not invent a ROM boot.
            report = assemble_credits(args.segment)
            if args.json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print(format_credits(report.plan))
            return 0
        else:
            plan = credits_chain(args.segment)
            if args.json:
                print(json.dumps(plan.to_dict(), indent=2))
            else:
                print(format_credits(plan))
            return 0
    except (PreflightError, AssembleError, SchemaError) as exc:
        _print_error(exc, as_json=True)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "chain":
        from super_metroid.splice.assemble import replay_tape_chain, solve_tape_chain
        stem = args.stem or (CLEAN_MAIN_STEM if args.replay else CLEAN_BASELINE_STEM)
        if args.replay:
            report = replay_tape_chain(stem=stem)
        else:
            if args.rebuild_board or not args.board.is_file():
                from super_metroid.human_tape.product_chain import write_clean_chain_board
                write_clean_chain_board(out=args.board)
            overrides = {int(value.split("=", 1)[0]): Path(value.split("=", 1)[1]) for value in args.override}
            report = solve_tape_chain(
                board=args.board, stem=stem,
                through_row=9999 if args.through_row is None else args.through_row,
                max_delay=args.max_delay, resume=not args.fresh, overrides=overrides,
                from_row=args.from_row, source=args.source_stem if args.from_row else None,
            )
        proof = report.get("proof") or report
        last_row = proof["last_row"]
        reached = args.through_row is None or (last_row is not None and last_row >= args.through_row)
        accepted = bool(proof["success"] and reached and (not args.require_credits or proof["credits_complete"]))
        print(json.dumps({
            "total_frames": report["total_frames"], "rooms_cleared": report["rooms_cleared"],
            "last_row": last_row, "requested_row": args.through_row, "accepted": accepted,
            "stopped": report.get("stopped"),
            "policy_coverage": _measured_coverage(
                stem, args.board,
                report.get("rows") or json.loads(stem.with_suffix(".json").read_text())["rows"],
            ),
            "proof": proof if args.proof_json else {key: proof.get(key) for key in (
                "success", "rom_vanilla", "state_loads", "ram_writes", "deaths", "endpoint_matches",
                "all_planned_crossings_match", "transition_timeline_matches",
                "observed_transition_count", "detour_rows",
                "credits_complete", "credits_tail_frames",
            )},
        }, indent=2))
        return 0 if accepted else 1
    if args.cmd == "preflight":
        return _run_preflight(args)
    if args.cmd == "cards":
        return _run_cards(args)
    if args.cmd == "prepare":
        return _run_prepare(args)
    if args.cmd == "grade":
        return _run_grade(args)
    if args.cmd == "assemble":
        return _run_assemble(args)
    if args.cmd == "range":
        return _run_range(args)
    if args.cmd == "assemble-range":
        return _run_assemble_range(args)
    if args.cmd == "credits":
        return _run_credits(args)
    if args.cmd == "benchmark":
        return _run_benchmark(args)
    if args.cmd == "policy":
        return _run_policy(args)
    parser.print_help()
    return 2


def _run_benchmark(args: argparse.Namespace) -> int:
    from super_metroid.splice.benchmark import (
        HISTORY_PATH,
        build_benchmark_report,
        format_benchmark_report,
        record_benchmark_history,
        scan_master_splits,
    )

    splits_data = None
    if args.scan:
        splits_data = scan_master_splits(
            args.stem,
            save_checkpoints=bool(args.save_checkpoints),
            overwrite_checkpoints=bool(args.overwrite_checkpoints),
        )
    bench = build_benchmark_report(
        args.stem,
        board=args.board,
        splits_data=splits_data,
        top_hotspots=args.top_hotspots,
    )
    history: list[dict[str, Any]] = []
    if args.record_history:
        history = record_benchmark_history(bench, label=args.label, notes=args.notes)
    elif HISTORY_PATH.is_file():
        history = json.loads(HISTORY_PATH.read_text())

    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(format_benchmark_report(bench, history=history))

    if args.json:
        print(json.dumps(bench, indent=2))
    else:
        print(format_benchmark_report(bench, history=history), end="")
    return 0


def _run_policy(args: argparse.Namespace) -> int:
    from super_metroid.splice.route_policy import (
        ab_test_policy_rule,
        build_policy_chain,
        format_ab_report,
        format_policy_list,
        load_policy,
    )

    if args.policy_cmd == "list":
        print(format_policy_list(), end="")
        return 0
    if args.policy_cmd == "show":
        pol = load_policy(args.policy).with_toggles(
            enable=args.enable, disable=args.disable, use_tape=args.use_tape,
        )
        print(json.dumps(pol.to_dict(), indent=2))
        return 0
    if args.policy_cmd == "ab":
        pol = load_policy(args.policy)
        ab = ab_test_policy_rule(
            pol,
            args.rule_id,
            source_stem=args.source_stem,
            follow_rows=args.follow_rows,
            max_delay=args.max_delay,
            save_npz=args.save_npz,
        )
        if args.json:
            print(json.dumps(ab, indent=2))
        else:
            print(format_ab_report(ab), end="")
        return 0 if ab["verdict"] == "PASS" else 1
    if args.policy_cmd == "build":
        pol = load_policy(args.policy).with_toggles(
            enable=args.enable, disable=args.disable, use_tape=args.use_tape,
        )
        if args.only:
            pol = replace(pol, rules=tuple(pol.rule_by_id(rule_id) for rule_id in dict.fromkeys(args.only)))
        source_stem = args.source_stem or (CLEAN_MAIN_STEM if args.only else CLEAN_BASELINE_STEM)
        report = build_policy_chain(
            pol,
            source_stem=source_stem,
            prefix_stem=args.prefix_stem,
            out_stem=args.stem,
            max_delay=args.max_delay,
            use_checkpoint_seed=not args.no_checkpoint_seed,
            verify_power_on=not args.skip_replay,
            through_row=args.through_row,
        )
        proof = report.get("proof") or {}
        accepted = bool(proof.get("success") and (args.through_row is not None or proof.get("credits_complete")))
        print(json.dumps({
            "policy_id": pol.policy_id,
            "stem": str(args.stem),
            "total_frames": report["total_frames"],
            "rooms_cleared": report["rooms_cleared"],
            "accepted": accepted,
            "stopped": report.get("stopped"),
            "rules_applied": report.get("policy_rules_applied"),
            "proof": {k: proof.get(k) for k in (
                "success", "rom_vanilla", "state_loads", "ram_writes", "deaths",
                "endpoint_matches", "credits_complete", "credits_tail_frames",
            )},
        }, indent=2))
        return 0 if accepted else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
