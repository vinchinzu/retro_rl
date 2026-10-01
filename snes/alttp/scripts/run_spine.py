"""Opening spine CLI. ``--through <id>`` stops on that hop.

``room_50`` plays the verified prefix. ``room_01`` and ``room_72`` continue
power-on through the east door and the north-wall well. ``room_82`` and
``room_81`` continue from that predecessor through the B1 south door and
the west bridge. ``zelda`` earns the keys and rescues Zelda; ``sanctuary``
escorts through the mantle and sewers. Controller results require audited
forward replay; their successful execution alone is not a Clean claim.
Does not overwrite ``recordings/verified_tip_run.json``. ``--no-video`` is
the default.

    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_spine.py --through room_50 --no-video
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/run_spine.py --through zelda --no-video
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable

from alttp.opening_route.escape_graph import N_ROOM_50
from alttp.opening_route.full_tip import (
    FullTipResult,
    _exit_acceptance_ok,
    run_to_verified_tip,
)
from alttp.paths import RECORDINGS_DIR
from alttp.ram import AlttpSnapshot
from alttp.spine.hops import (
    DEFAULT_THROUGH,
    FAIL_CLOSED_BLOCKER,
    FAIL_CLOSED_NOTES,
    FAIL_STAGE,
    SPINE_THROUGH,
    SpineHop,
    hop_for,
    hops_through,
)
from alttp.startup import build_boot_env

VERIFIED_TIP_RUN = RECORDINGS_DIR / "verified_tip_run.json"

_TipFn = Callable[..., FullTipResult]
_BootFn = Callable[[], Any]
_Writer = Callable[[Path, dict[str, Any]], Path]


def leftover_path(through: str) -> Path:
    """Leftover JSON for one ``--through`` stop. Never the verified-tip artifact."""
    return RECORDINGS_DIR / f"opening_spine_{through}.json"


def leftover_glance(snapshot: AlttpSnapshot) -> dict[str, Any]:
    """RAM glance used as leave proof (no MP4)."""
    return {
        "room_hex": f"0x{snapshot.room_base_id:02X}",
        "module": snapshot.game_mode,
        "submodule": snapshot.submodule,
        "xy": [snapshot.link_x, snapshot.link_y],
        "sword": snapshot.sword_level,
        "follower": snapshot.follower,
        "follower_addr": "$F3CC",
        "keys": snapshot.num_keys,
        "indoors": snapshot.indoors,
        "has_control": snapshot.has_control,
        "has_zelda_follower": snapshot.has_zelda_follower,
    }


def fail_closed_payload(
    through: str, blocker: str, *, notes: list[str]
) -> dict[str, Any]:
    return {
        "kind": "alttp_opening_spine",
        "through": through,
        "ok": False,
        "phase": "fail_closed",
        "frames": 0,
        "blocker": blocker,
        "tip_node": N_ROOM_50,
        "verified_tip": N_ROOM_50,
        "source": "fail_closed",
        "clean_chain": False,
        "continuous": False,
        "leftover": None,
        "video": None,
        "notes": list(notes),
    }


def spine_payload_from_tip(
    result: FullTipResult, *, through: str = DEFAULT_THROUGH
) -> dict[str, Any]:
    return {
        "kind": "alttp_opening_spine",
        "through": through,
        "ok": bool(result.ok),
        "phase": result.phase,
        "frames": result.frames,
        "blocker": result.blocker,
        "tip_node": result.tip_node,
        "verified_tip": N_ROOM_50,
        "source": result.source,
        "clean_chain": result.source == "natural_boot" and bool(result.ok),
        "continuous": result.source == "natural_boot" and bool(result.ok),
        "leftover": leftover_glance(result.snapshot),
        "video": None,
        "notes": list(result.notes),
        "report": result.to_report(),
    }


class _FrameEnv:
    """Count ``step`` calls. Other attributes fall through to the real env."""

    def __init__(self, env: object) -> None:
        self._env = env
        self.frames = 0

    def step(self, action: object, *args: object, **kwargs: object) -> object:
        self.frames += 1
        return self._env.step(action, *args, **kwargs)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self._env, name)


class _PinSegment:
    """Save a pin at stage start, then play the real segment."""

    def __init__(self, segment: object, on_start: Callable[[object], None]) -> None:
        self._segment = segment
        self._on_start = on_start

    def play_checked(self, env: object, **kwargs: object) -> object:
        self._on_start(env)
        return self._segment.play_checked(env, **kwargs)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self._segment, name)


def write_save_point(
    env: object,
    prefix: str,
    stage: str,
    *,
    kind: str,
    source_run: str,
) -> Path:
    """Write ``<prefix>_<stage>.state`` and its provenance sidecar."""
    from retro_harness.env import save_state

    from alttp.paths import GAME, GAME_DIR
    from alttp.spine.hops import save_point_name
    from alttp.spine.tape import write_pin_provenance

    path = save_state(env, GAME_DIR, GAME, save_point_name(prefix, stage))  # type: ignore[arg-type]
    write_pin_provenance(
        path,
        source_run=source_run,
        frame=int(getattr(env, "frames", 0)),
        kind=kind,
    )
    return path


def load_save_point(env: object, prefix: str, stage: str) -> Path:
    """Load ``<prefix>_<stage>``. Reset, then ``set_state``. Do not step after.

    stable-retro refuses ``step`` until ``reset``, and ``reset`` advances one
    boot frame. ``set_state`` replaces that pose; a following step would not.
    """
    from retro_harness.env import read_state_bytes, state_path

    from alttp.paths import GAME, GAME_DIR
    from alttp.spine.hops import save_point_name

    path = state_path(GAME_DIR, GAME, save_point_name(prefix, stage))
    if not path.is_file():
        raise FileNotFoundError(f"no save point for stage {stage!r}: {path}")
    env.reset()  # type: ignore[attr-defined]
    env.em.set_state(read_state_bytes(path))  # type: ignore[attr-defined]
    data = getattr(env, "data", None)
    refresh = getattr(data, "update_ram", None)
    if refresh is not None:
        refresh()
    return path


def _saving_lookup(
    prefix: str,
    *,
    kind: str,
    source_run: str,
    written: list[str],
) -> Callable[[str], object]:
    from alttp.opening_route.segment import get_segment

    def lookup(segment_id: str) -> object:
        segment = get_segment(segment_id)

        def on_start(active: object, sid: str = segment_id) -> None:
            path = write_save_point(
                active, prefix, sid, kind=kind, source_run=source_run
            )
            written.append(path.name)

        return _PinSegment(segment, on_start)

    return lookup


def _call_tip(
    fn: _TipFn,
    env: object,
    *,
    close: bool,
    hop: SpineHop,
    extra: dict[str, Any] | None = None,
) -> FullTipResult:
    """Pass hop stages when the tip callable accepts them (test fakes may not)."""
    params = inspect.signature(fn).parameters
    var_kw = any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in params.values()
    )
    kwargs: dict[str, Any] = {"close": close}
    if "segment_ids" in params or var_kw:
        kwargs["segment_ids"] = hop.stages
    for key, value in (extra or {}).items():
        if value is None:
            continue
        if key in params or var_kw:
            kwargs[key] = value
    return fn(env, **kwargs)


def _merged_hop(through: str) -> SpineHop:
    played = hops_through(through)
    if not played:
        raise ValueError(f"hop table has no stop {through!r}")
    hop = played[-1]
    if hop.through != through or hop_for(through) is None:
        raise ValueError(f"hop table has no stop {through!r}")
    return SpineHop(
        through=hop.through,
        stop=hop.stop,
        stages=tuple(stage for item in played for stage in item.stages),
        success=hop.success,
    )


def _apply_stop(payload: dict[str, Any], result: FullTipResult, hop: SpineHop) -> None:
    if result.tip_node != hop.success:
        payload["ok"] = False
        payload["continuous"] = False
        payload["clean_chain"] = False
        if not payload["blocker"]:
            payload["blocker"] = (
                f"expected stop {hop.success!r}, resolved {result.tip_node!r}"
            )


def _apply_room_72_glance(payload: dict[str, Any], snapshot: object) -> None:
    """Fail a room_72 stop that misses the stair-column leave band.

    Does not widen the band. A miss replaces a green stop; it does not hide
    an earlier blocker.
    """
    from alttp.screen_glance import ROOM_72, grade_leftover, leftover_from_snapshot

    graded = grade_leftover(leftover_from_snapshot(snapshot), ROOM_72)
    payload["glance"] = {
        "hop": ROOM_72.hop,
        "ok": graded.ok,
        "misses": list(graded.misses),
    }
    if payload.get("ok") and not graded.ok:
        payload["ok"] = False
        payload["continuous"] = False
        payload["clean_chain"] = False
        payload["blocker"] = "ROOM_72 glance: " + "; ".join(graded.misses)


def _mark_resumed(payload: dict[str, Any], stage: str) -> None:
    """A loaded pin is not continuous, even when the tail reaches the stop."""
    payload["resumed_from"] = stage
    payload["continuous"] = False
    payload["clean_chain"] = False
    payload["source"] = "resumed"


def play_from_stages(
    env: object,
    stages: tuple[str, ...],
    *,
    resume_stage: str,
    success: str,
) -> FullTipResult:
    """Play ``stages`` from a pin already loaded. No boot and no idle frame."""
    from alttp import primitives
    from alttp.opening_route.anchors import resolve_continuous_tip_node
    from alttp.opening_route.segment import get_segment
    from alttp.route_report import RoutePhaseResult
    from alttp.startup import snapshot_env

    pre = snapshot_env(env)  # type: ignore[arg-type]
    first = get_segment(resume_stage)
    boot = RoutePhaseResult(
        phase="resumed",
        ok=first.entry.matches(pre),
        frames=0,
        snapshot=pre,
        detail=f"loaded {resume_stage} with no idle frame",
    )
    notes = [
        f"Resumed from {resume_stage}; one disclosed state load, no idle frame.",
        "Not a continuous power-on.",
    ]
    if not boot.ok:
        return FullTipResult(
            ok=False,
            phase="entry_rejected",
            frames=0,
            snapshot=pre,
            tip_node=resolve_continuous_tip_node(pre),
            boot=boot,
            blocker=(
                f"entry requirement not met for {resume_stage}: "
                f"{first.entry.description}"
            ),
            notes=notes + ["play_fn not called (entry gate)"],
            source="resumed",
        )
    total = 0
    rows = []
    for segment_id in stages:
        segment = get_segment(segment_id)
        evidence = segment.play_checked(env, source="resumed")  # type: ignore[arg-type]
        rows.append(evidence)
        total += evidence.frames
        if not _exit_acceptance_ok(segment, evidence):
            missing = [
                key
                for key in segment.exit.acceptance_keys
                if not evidence.acceptance.get(key, False)
            ]
            detail = evidence.blocker or "segment did not satisfy registered exit"
            if missing:
                detail = f"{detail}; missing acceptance: {', '.join(missing)}"
            return FullTipResult(
                ok=False,
                phase=segment_id,
                frames=total,
                snapshot=evidence.snapshot,
                tip_node=resolve_continuous_tip_node(evidence.snapshot),
                boot=boot,
                segments=rows,
                blocker=detail,
                notes=notes,
                source="resumed",
            )
    settle = primitives.settle_control(env)
    total += settle.frames
    final = settle.snapshot
    if rows:
        rows[-1].phases.append(
            RoutePhaseResult(
                phase="tip_settle_control",
                ok=settle.ok,
                frames=settle.frames,
                snapshot=final,
                detail=settle.reason,
            )
        )
        rows[-1].snapshot = final
    tip_node = resolve_continuous_tip_node(final)
    if not settle.ok:
        return FullTipResult(
            ok=False,
            phase="tip_settle_control",
            frames=total,
            snapshot=final,
            tip_node=tip_node,
            boot=boot,
            segments=rows,
            blocker=f"tip transition did not settle: {settle.reason}",
            notes=notes,
            source="resumed",
        )
    if tip_node != success:
        return FullTipResult(
            ok=False,
            phase="tip_mismatch",
            frames=total,
            snapshot=final,
            tip_node=tip_node,
            boot=boot,
            segments=rows,
            blocker=f"expected stop {success!r}, resolved {tip_node!r}",
            notes=notes,
            source="resumed",
        )
    return FullTipResult(
        ok=True,
        phase="verified_tip_reached",
        frames=total,
        snapshot=final,
        tip_node=tip_node,
        boot=boot,
        segments=rows,
        notes=notes,
        source="resumed",
    )


def dispatch(
    through: str,
    *,
    env: object | None = None,
    close: bool = True,
    run_tip_fn: _TipFn | None = None,
    save_points: str | None = None,
    resume: str | None = None,
) -> dict[str, Any]:
    """Play hops through ``through``. Fail-closed ids do not call the tip."""
    tip_fn = run_to_verified_tip if run_tip_fn is None else run_tip_fn
    played = hops_through(through)
    if resume and not played:
        raise ValueError(f"cannot resume fail-closed stop {through!r}")
    if played:
        hop = _merged_hop(through)
        source_run = f"spine:{through}"
        if resume is not None:
            if not save_points:
                raise ValueError("resume needs a save-point prefix")
            if resume not in hop.stages:
                known = ", ".join(hop.stages)
                raise ValueError(
                    f"unknown resume stage {resume!r}; stages: {known}"
                )
            if env is None:
                raise RuntimeError("resume needs an env")
            load_save_point(env, save_points, resume)
            start = hop.stages.index(resume)
            result = play_from_stages(
                env,
                hop.stages[start:],
                resume_stage=resume,
                success=hop.success,
            )
            payload = spine_payload_from_tip(result, through=through)
            _apply_stop(payload, result, hop)
            if through == "room_72":
                _apply_room_72_glance(payload, result.snapshot)
            _mark_resumed(payload, resume)
            payload["save_points"] = []
            return payload
        written: list[str] = []
        extra: dict[str, Any] = {"success": hop.success}
        if save_points:
            extra["get_segment_fn"] = _saving_lookup(
                save_points,
                kind="natural",
                source_run=source_run,
                written=written,
            )
        result = _call_tip(tip_fn, env, close=close, hop=hop, extra=extra)
        payload = spine_payload_from_tip(result, through=through)
        _apply_stop(payload, result, hop)
        if through == "room_72":
            _apply_room_72_glance(payload, result.snapshot)
        payload["resumed_from"] = None
        payload["save_points"] = list(written)
        return payload
    blocker = FAIL_CLOSED_BLOCKER.get(through)
    if blocker is not None:
        return fail_closed_payload(
            through,
            blocker,
            notes=list(FAIL_CLOSED_NOTES[through]),
        )
    known = ", ".join(SPINE_THROUGH)
    raise ValueError(f"unknown --through {through!r}; known: {known}")


def write_leftover(
    path: Path, payload: dict[str, Any], *, banned: Path | None = None
) -> Path:
    resolved = path.expanduser().resolve()
    banned_path = VERIFIED_TIP_RUN if banned is None else banned
    if resolved == banned_path.expanduser().resolve():
        raise RuntimeError(
            "refusing to overwrite recordings/verified_tip_run.json; "
            "use recordings/opening_spine_<through>.json"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--through",
        choices=SPINE_THROUGH,
        default=DEFAULT_THROUGH,
        help=(
            "Spine stop. room_50 is the verified prefix. "
            "room_01 and room_72 continue power-on through the east door and the well. "
            "room_82 and room_81 continue through the B1 south door and the west bridge. "
            "zelda rescues the follower; sanctuary completes the escort. "
            "Use audit_power_on for HP-assisted controller and forward replay evidence."
        ),
    )
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record MP4 (default off). Encode is not wired; leftover JSON only.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Leftover JSON path (default: recordings/opening_spine_<through>.json)",
    )
    parser.add_argument(
        "--save-points",
        default=None,
        help=(
            "Write <prefix>_<stage>.state at every stage start, "
            "and <prefix>_fail on a red run."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Load <prefix>_<stage> with no idle frame and play forward.",
    )
    return parser


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def main(
    argv: list[str] | None = None,
    *,
    boot_fn: _BootFn | None = None,
    tip_fn: _TipFn | None = None,
    leftover_writer: _Writer | None = None,
    banned_report: Path | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and not args.save_points:
        parser.error("--resume needs --save-points PREFIX")
    _configure_headless()
    report_path = args.report if args.report is not None else leftover_path(args.through)
    banned = VERIFIED_TIP_RUN if banned_report is None else banned_report
    if report_path.expanduser().resolve() == banned.expanduser().resolve():
        raise RuntimeError("refusing to overwrite recordings/verified_tip_run.json")
    make_env = build_boot_env if boot_fn is None else boot_fn
    writer = write_leftover if leftover_writer is None else leftover_writer
    active: object | None = None
    live = None
    try:
        if hop_for(args.through) is not None:
            raw = make_env()
            if hasattr(raw, "em") and hasattr(raw, "data"):
                from alttp.spine.tape import LiveRun

                live = LiveRun(raw, provider="alttp.spine_controller")
                raw = live.env
            active = _FrameEnv(raw) if args.save_points else raw
        payload = dispatch(
            args.through,
            env=active,
            close=False,
            run_tip_fn=tip_fn,
            save_points=args.save_points,
            resume=args.resume,
        )
        if live is not None:
            from dataclasses import asdict

            audit = asdict(live.report())
            payload["audit"] = audit
            # This CLI runs a controller, including search restores and boot
            # SRAM preparation. A separate forward replay proves Clean.
            payload.update(clean_chain=False, continuous=False, development_only=True,
                           source="controller_search", intervention="controller search")
            payload["notes"] = list(payload.get("notes") or []) + [
                "Controller execution; verify the committed input tape before a Clean claim."
            ]
            report = payload.get("report")
            if isinstance(report, dict):
                report.update(audit=audit, clean_chain=False, development_only=True,
                              source="controller_search", intervention="controller search")
                report["notes"] = list(payload["notes"])
                for row in report.get("segments", []):
                    row.update(source="controller_search", cleanChain=False, developmentOnly=True)
        payload["video"] = None
        if args.video:
            notes = list(payload.get("notes") or [])
            notes.append(
                "--video requested; MP4 encode is not wired on this CLI (leftover JSON only)."
            )
            payload["notes"] = notes
        if args.save_points and active is not None and payload["ok"] and not args.resume:
            # Arrival pin for the stop. A later resume from it is not continuous.
            path = write_save_point(
                active,
                args.save_points,
                args.through,
                kind="natural",
                source_run=f"spine:{args.through}",
            )
            pins = list(payload.get("save_points") or [])
            pins.append(path.name)
            payload["save_points"] = pins
        elif args.save_points and active is not None and not payload["ok"]:
            kind = "resumed" if args.resume else "natural"
            path = write_save_point(
                active,
                args.save_points,
                FAIL_STAGE,
                kind=kind,
                source_run=f"spine:{args.through}",
            )
            pins = list(payload.get("save_points") or [])
            pins.append(path.name)
            payload["save_points"] = pins
    finally:
        if active is not None:
            active.close()  # type: ignore[attr-defined]

    writer(report_path, payload)
    print(f"Wrote {report_path}")
    print(
        f"ok={payload['ok']} through={payload['through']} "
        f"phase={payload.get('phase')} frames={payload.get('frames', 0)} "
        f"blocker={payload.get('blocker')!r} "
        f"resumed_from={payload.get('resumed_from')!r} "
        f"pins={payload.get('save_points')} "
        f"glance={payload.get('glance')}"
    )
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
