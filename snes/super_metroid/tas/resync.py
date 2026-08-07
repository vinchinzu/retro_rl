"""Phase-aligned Super Metroid TAS resync under stable-retro.

Raw power-on Sniq any% desyncs after first Ceres control (lsnes vs harness).
This module re-anchors with the **product pure morph spine** through Ceres →
Landing Site, then splices the movie body at a searched movie index so Zebes
rooms (Parlor / Climb / …) can be annotated.

```bash
# Product → Landing, search movie align, annotate through a few rooms
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m super_metroid.tas.resync --to landing --search \\
  --body 8000 --out snes/super_metroid/recordings/tas_import/resync_landing

# Use a known movie index (skip search)
uv run python -m super_metroid.tas.resync --to landing --movie-start 21000 --body 6000

# Product through Climb → Pit entry (prefer product first-jump for room tech)
uv run python -m super_metroid.tas.resync --to pit --movie-start 17000 --body 2000
```

lsnes is not required: button order already matches env SNES-12. True lsnes
bit-exact playback is unavailable without that core; product re-anchor is the
harness-equivalent "wiring".
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from retro_harness.actions import buttons, idle_action
from retro_harness.env import make_env, write_state_bytes
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.paths import GAME, GAME_DIR
from super_metroid.progression import MORPH_GRAPH
from super_metroid.ram import SuperMetroidState, parse_env_state, probe_pin
from super_metroid.routes.kpdr.early_spine import (
    MORPH_SPINE,
    play_boot_to_ceres,
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BLUE_BRINSTAR_ELEVATOR,
    ROOM_CLIMB,
    ROOM_LANDING_SITE,
    ROOM_MORPH,
    ROOM_PARLOR,
    ROOM_PIT,
)
from super_metroid.routes.runtime import RouteSession
from super_metroid.tas.annotate import Annotator, is_settled_control
from super_metroid.tas.lsmv import parse_lsmv
from super_metroid.tas.trace import (
    DEFAULT_OUT_ROOT,
    FrameSample,
    MovieTrace,
    action_array,
    frame_button_names,
    write_trace_artifacts,
)

# Sniq any% first B+RIGHT (Ceres movement) — movie-relative.
SNIQ_ANY_CERES_OPEN = 8639
# Product morph dual: landing ~21.5k; Ceres open relative ≈ 12–14k movie frames
# after 8639 → landing movie index often near 20–24k. Search widens this.
DEFAULT_LANDING_SEARCH = range(18_000, 28_000, 250)

ZEBES_MILESTONES = (
    ROOM_LANDING_SITE,
    ROOM_PARLOR,
    ROOM_CLIMB,
    ROOM_PIT,
    ROOM_MORPH,
)


def _act(*names: str) -> np.ndarray:
    return np.asarray(buttons(*names) if names else idle_action(), dtype=np.int8)


def _combo_assist() -> UnlimitedResourcesAssist:
    """Unlimited energy+ammo assist used by continuous product path."""
    return UnlimitedResourcesAssist()


def play_product_to_landing(session: RouteSession) -> None:
    """Product pure: power-on → Ceres → Landing Site (first three spine hops)."""
    play_boot_to_ceres(session)
    play_ceres_outbound_to_ridley(session)
    play_ceres_escape_to_landing(session)
    if int(session.state.room_id) != ROOM_LANDING_SITE:
        raise RuntimeError(
            f"product path missed Landing Site: room=0x{session.state.room_id:04X} "
            f"frame={session.frame}"
        )


def play_product_spine_until(
    session: RouteSession,
    *,
    stop_room: int,
) -> list[str]:
    """Play MORPH_SPINE hops until ``stop_room`` is the hop destination."""
    played: list[str] = []
    for hop in MORPH_SPINE:
        hop.play(session)
        played.append(hop.hop_id)
        if int(hop.to_room) == stop_room and int(session.state.room_id) == stop_room:
            break
        if stop_room == ROOM_LANDING_SITE and hop.hop_id == "zebes_landing":
            break
    return played


@dataclass
class AlignTrial:
    movie_start: int
    pad: int
    score: float
    unique_rooms: int
    room_order: list[str]
    hit_parlor: int | None = None
    hit_climb: int | None = None
    hit_pit: int | None = None
    hit_elev: int | None = None
    hit_morph: int | None = None
    deaths: int = 0
    frames_played: int = 0
    # Best rightward progress while still in Pit (x pin for jump tuning).
    pit_max_x: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_zebes_progress(
    rooms: Sequence[int],
    *,
    hit_parlor: int | None,
    hit_climb: int | None,
    hit_pit: int | None = None,
    hit_elev: int | None = None,
    hit_morph: int | None,
    deaths: int,
    pit_max_x: int = 0,
) -> float:
    score = float(len(set(rooms)))
    if hit_parlor is not None:
        score += 20
    if hit_climb is not None:
        score += 30
    if hit_pit is not None:
        score += 35
    if hit_elev is not None:
        score += 45
    if hit_morph is not None:
        score += 50
    # Ordered milestone depth.
    for rid, bonus in (
        (ROOM_LANDING_SITE, 2),
        (ROOM_PARLOR, 5),
        (ROOM_CLIMB, 8),
        (ROOM_PIT, 10),
        (ROOM_BLUE_BRINSTAR_ELEVATOR, 11),
        (ROOM_MORPH, 12),
    ):
        if rid in rooms:
            score += bonus
        else:
            break
    # Prefer trials that actually traverse Pit (horizontal platform jumps).
    if pit_max_x > 0:
        score += min(pit_max_x, 800) / 80.0
    score -= deaths * 8
    return score


def trial_movie_from_state(
    state_bytes: bytes,
    frames: list[list[int]],
    *,
    movie_start: int,
    pad: int = 0,
    body: int = 6000,
) -> AlignTrial:
    """Load a harness state, optional idle pad, play movie[movie_start:…]."""
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        env.em.set_state(state_bytes)
        frame = 0
        for _ in range(pad):
            env.step(_act())
            frame += 1
        seen: list[int] = []
        seen_set: set[int] = set()
        hit_parlor = hit_climb = hit_pit = hit_elev = hit_morph = None
        deaths = 0
        pit_max_x = 0
        mi = movie_start
        limit = min(body, max(0, len(frames) - movie_start))
        for _ in range(limit):
            env.step(action_array(frames[mi]))
            frame += 1
            mi += 1
            st = parse_env_state(env, frame=frame, mode="nav")
            if st.phase.name == "DEATH_OR_GAME_OVER":
                deaths += 1
                break
            if is_settled_control(st) and st.room_id and st.room_id not in seen_set:
                seen_set.add(int(st.room_id))
                seen.append(int(st.room_id))
            rid = int(st.room_id)
            if rid == ROOM_PIT:
                pit_max_x = max(pit_max_x, int(st.samus_x))
            if (
                hit_parlor is None
                and rid == ROOM_PARLOR
                and is_settled_control(st)
            ):
                hit_parlor = frame
            if (
                hit_climb is None
                and rid == ROOM_CLIMB
                and is_settled_control(st)
            ):
                hit_climb = frame
            if (
                hit_pit is None
                and rid == ROOM_PIT
                and is_settled_control(st)
            ):
                hit_pit = frame
            if (
                hit_elev is None
                and rid == ROOM_BLUE_BRINSTAR_ELEVATOR
                and is_settled_control(st)
            ):
                hit_elev = frame
            if (
                hit_morph is None
                and rid == ROOM_MORPH
                and is_settled_control(st)
            ):
                hit_morph = frame
                break
        score = score_zebes_progress(
            seen,
            hit_parlor=hit_parlor,
            hit_climb=hit_climb,
            hit_pit=hit_pit,
            hit_elev=hit_elev,
            hit_morph=hit_morph,
            deaths=deaths,
            pit_max_x=pit_max_x,
        )
        return AlignTrial(
            movie_start=movie_start,
            pad=pad,
            score=score,
            unique_rooms=len(seen_set),
            room_order=[f"0x{r:04X}" for r in seen],
            hit_parlor=hit_parlor,
            hit_climb=hit_climb,
            hit_pit=hit_pit,
            hit_elev=hit_elev,
            hit_morph=hit_morph,
            deaths=deaths,
            frames_played=frame,
            pit_max_x=pit_max_x,
        )
    finally:
        env.close()


def search_movie_align(
    state_bytes: bytes,
    frames: list[list[int]],
    *,
    starts: Sequence[int],
    pads: Sequence[int] = (0,),
    body: int = 6000,
    progress: Callable[[AlignTrial], None] | None = None,
) -> list[AlignTrial]:
    """Grid-search movie_start × pad; return trials sorted by score desc."""
    trials: list[AlignTrial] = []
    for ms in starts:
        if ms < 0 or ms >= len(frames):
            continue
        for pad in pads:
            tr = trial_movie_from_state(
                state_bytes, frames, movie_start=ms, pad=pad, body=body
            )
            trials.append(tr)
            if progress is not None:
                progress(tr)
    trials.sort(key=lambda t: (-t.score, t.deaths, t.movie_start))
    return trials


@dataclass
class ResyncRun:
    """Product prefix + movie body with full annotation."""

    prefix: str
    movie_start: int
    pad: int
    product_frames: int
    movie_frames: int
    total_frames: int
    rooms: list[dict[str, Any]] = field(default_factory=list)
    events_summary: dict[str, Any] = field(default_factory=dict)
    final: dict[str, Any] = field(default_factory=dict)
    align_best: dict[str, Any] | None = None
    state_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Product pure prefix targets → stop room id (None = special-cased).
PREFIX_STOP_ROOM: dict[str, int | None] = {
    "landing": ROOM_LANDING_SITE,
    "parlor": ROOM_PARLOR,
    "climb": ROOM_CLIMB,
    "pit": ROOM_PIT,
    "elev": ROOM_BLUE_BRINSTAR_ELEVATOR,
    "morph": ROOM_MORPH,
}


def run_product_prefix(
    *,
    to: str = "landing",
) -> tuple[Any, RouteSession, bytes, int, list[str]]:
    """Open env, play product prefix, return (env, session, state_bytes, frame, hops)."""
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    assist = _combo_assist()
    session = RouteSession(env, writer=None, assist=assist, graph=MORPH_GRAPH)

    hops: list[str] = []
    if to == "landing":
        # Fast path: first three spine hops only (no parlor/climb seeds).
        play_product_to_landing(session)
        hops = ["first_ceres_control", "ridley_countdown", "zebes_landing"]
    elif to in PREFIX_STOP_ROOM:
        stop = PREFIX_STOP_ROOM[to]
        assert stop is not None
        hops = play_product_spine_until(session, stop_room=stop)
        if int(session.state.room_id) != stop:
            raise RuntimeError(
                f"product prefix --to {to} missed 0x{stop:04X}: "
                f"room=0x{session.state.room_id:04X} frame={session.frame}"
            )
    else:
        raise ValueError(
            f"unknown prefix target {to!r}; choose from {sorted(PREFIX_STOP_ROOM)}"
        )

    state_bytes = env.em.get_state()
    return env, session, state_bytes, session.frame, hops


def resync_and_annotate(
    *,
    frames: list[list[int]],
    movie_start: int,
    pad: int = 0,
    body: int = 8000,
    to: str = "landing",
    series_stride: int = 4,
    dump_states_on: Sequence[str] = ("room_enter", "control", "item_gain"),
    out_dir: Path | None = None,
    state_bytes: bytes | None = None,
    product_frames: int | None = None,
    prefix_hops: list[str] | None = None,
) -> tuple[ResyncRun, MovieTrace]:
    """Product prefix (or given state) + movie body with annotation artifacts."""
    own_env = False
    if state_bytes is None:
        env, session, state_bytes, product_frames, prefix_hops = run_product_prefix(to=to)
        own_env = True
        # Continue on same env after product prefix (already at landing).
        # Re-get state after optional pad on this env.
        frame0 = product_frames or session.frame
    else:
        env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
        env.reset()
        env.em.set_state(state_bytes)
        frame0 = product_frames or 0
        session = None
        own_env = True
        prefix_hops = prefix_hops or []

    out = out_dir or (DEFAULT_OUT_ROOT / f"resync_{to}_{movie_start}")
    out.mkdir(parents=True, exist_ok=True)
    states_dir = out / "states"
    states_dir.mkdir(exist_ok=True)

    # Save product anchor state.
    anchor_path = out / "product_anchor.state"
    write_state_bytes(anchor_path, state_bytes)

    annotator = Annotator()
    series: list[FrameSample] = []
    state_dumps: list[dict[str, Any]] = []
    rooms_log: list[dict[str, Any]] = []
    last_room: int | None = None
    dump_kinds = frozenset(dump_states_on)

    frame = frame0
    st = parse_env_state(env, frame=frame, mode="nav")
    if is_settled_control(st):
        annotator.observe(st, buttons=())
        last_room = int(st.room_id)
        rooms_log.append(
            {
                "frame": frame,
                "room_id_hex": f"0x{st.room_id:04X}",
                "pose": st.pose,
                "x": st.samus_x,
                "y": st.samus_y,
                "source": "product_anchor",
            }
        )

    for _ in range(pad):
        env.step(_act())
        frame += 1
        st = parse_env_state(env, frame=frame, mode="nav")
        annotator.observe(st, buttons=())

    mi = movie_start
    movie_played = 0
    limit = min(body, max(0, len(frames) - movie_start))
    try:
        for _ in range(limit):
            fr = frames[mi]
            names = frame_button_names(fr)
            env.step(action_array(fr))
            frame += 1
            mi += 1
            movie_played += 1
            st = parse_env_state(env, frame=frame, mode="nav")
            new_ev = annotator.observe(st, buttons=names)

            if series_stride > 0 and frame % series_stride == 0:
                series.append(FrameSample.from_state(st, names))

            if is_settled_control(st) and int(st.room_id) != last_room and st.room_id:
                last_room = int(st.room_id)
                rooms_log.append(
                    {
                        "frame": frame,
                        "movie_index": mi - 1,
                        "room_id_hex": f"0x{st.room_id:04X}",
                        "pose": st.pose,
                        "x": st.samus_x,
                        "y": st.samus_y,
                        "items": f"0x{st.collected_items:04X}",
                        "source": "movie",
                    }
                )

            should_dump = False
            reason = ""
            for ev in new_ev:
                if ev.kind in dump_kinds:
                    should_dump = True
                    reason = ev.kind
                    break
            if should_dump:
                stem = f"f{frame:06d}_{reason}_r{st.room_id:04X}"
                path = states_dir / f"{stem}.state"
                write_state_bytes(path, env.em.get_state())
                state_dumps.append(
                    {
                        "frame": frame,
                        "reason": reason,
                        "path": str(path),
                        "pin": probe_pin(st),
                    }
                )

        final = {
            **probe_pin(st),
            "game_state": st.game_state,
            "phase": st.phase.value,
            "items": f"0x{st.collected_items:04X}",
            "beams": f"0x{st.collected_beams:04X}",
            "health": st.health,
        }
        trace = MovieTrace(
            source=f"resync product→movie@{movie_start}",
            start_mode=f"product_{to}+movie",
            num_frames=movie_played,
            frames_played=frame,
            events=list(annotator.events),
            rooms=[],
            room_timer_report={},
            series=series,
            final=final,
            state_dumps=state_dumps,
            annotate_summary=annotator.summary(),
            meta={
                "product_frames": product_frames,
                "movie_start": movie_start,
                "pad": pad,
                "movie_played": movie_played,
                "prefix_hops": prefix_hops or [],
                "series_stride": series_stride,
            },
        )
        run = ResyncRun(
            prefix=to,
            movie_start=movie_start,
            pad=pad,
            product_frames=int(product_frames or 0),
            movie_frames=movie_played,
            total_frames=frame,
            rooms=rooms_log,
            events_summary=annotator.summary(),
            final=final,
            state_path=str(anchor_path),
            meta=trace.meta,
        )
        return run, trace
    finally:
        if own_env:
            env.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--to",
        choices=tuple(PREFIX_STOP_ROOM),
        default="landing",
        help=(
            "Product pure prefix target before movie splice "
            "(landing|parlor|climb|pit|elev|morph; default: landing). "
            "Prefer pit to skip Climb under TAS — product owns Climb descent."
        ),
    )
    p.add_argument(
        "--movie",
        type=Path,
        default=GAME_DIR / "tas" / "ref" / "sniq_any_3653M.lsmv",
        help="LSMV / path (default Sniq any%)",
    )
    p.add_argument(
        "--movie-start",
        type=int,
        default=None,
        help="Movie index to splice after product prefix",
    )
    p.add_argument("--pad", type=int, default=0, help="Idle frames after product pin")
    p.add_argument("--body", type=int, default=8000, help="Movie frames to play")
    p.add_argument(
        "--search",
        action="store_true",
        help="Grid-search movie_start for Zebes room progress",
    )
    p.add_argument("--search-lo", type=int, default=18_000)
    p.add_argument("--search-hi", type=int, default=28_000)
    p.add_argument("--search-step", type=int, default=250)
    p.add_argument(
        "--search-pads",
        default="0,2,4,8",
        help="Comma pads for search",
    )
    p.add_argument("--series-stride", type=int, default=4)
    p.add_argument(
        "--states-on",
        default="room_enter,control,item_gain,capacity_gain",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    movie_path = Path(args.movie)
    if not movie_path.exists():
        print(f"error: missing movie {movie_path}", flush=True)
        return 1

    print(f"parse {movie_path.name}…", flush=True)
    if movie_path.suffix.lower() == ".lsmv":
        frames = parse_lsmv(movie_path).frames
    else:
        from super_metroid.tas.bk2 import parse_bk2

        frames = parse_bk2(movie_path).frames
    print(f"movie frames={len(frames)}", flush=True)

    print(f"product prefix → {args.to}…", flush=True)
    env, session, state_bytes, product_frames, hops = run_product_prefix(to=args.to)
    pin = probe_pin(session.state)
    print(
        f"  product f={product_frames} room={pin.get('room')} "
        f"pose={pin.get('pose')} xy=({pin.get('x')},{pin.get('y')}) "
        f"items={pin.get('collected_items')} hops={hops}",
        flush=True,
    )
    # Close product env; trials / annotate reopen from state bytes.
    env.close()

    out = args.out or (
        DEFAULT_OUT_ROOT / f"resync_{args.to}_{int(time.time())}"
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_state_bytes(out / "product_anchor.state", state_bytes)

    movie_start = args.movie_start
    pad = args.pad
    align_best: AlignTrial | None = None
    search_report: list[dict[str, Any]] = []

    if args.search or movie_start is None:
        pads = [int(x) for x in args.search_pads.split(",") if x.strip()]
        starts = list(range(args.search_lo, args.search_hi + 1, args.search_step))
        print(
            f"search movie_start {args.search_lo}..{args.search_hi} "
            f"step {args.search_step} pads={pads} body={min(args.body, 5000)}…",
            flush=True,
        )
        best_so_far: AlignTrial | None = None

        def _prog(tr: AlignTrial) -> None:
            nonlocal best_so_far
            if best_so_far is None or tr.score > best_so_far.score:
                best_so_far = tr
                print(
                    f"  BEST ms={tr.movie_start} pad={tr.pad} score={tr.score:.0f} "
                    f"rooms={tr.room_order} parlor={tr.hit_parlor} "
                    f"climb={tr.hit_climb} pit={tr.hit_pit} elev={tr.hit_elev} "
                    f"morph={tr.hit_morph} pit_max_x={tr.pit_max_x}",
                    flush=True,
                )

        trials = search_movie_align(
            state_bytes,
            frames,
            starts=starts,
            pads=pads,
            body=min(args.body, 5000),
            progress=_prog,
        )
        search_report = [t.to_dict() for t in trials[:40]]
        (out / "align_search.json").write_text(
            json.dumps(
                {"trials_top": search_report, "n_trials": len(trials)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if not trials:
            print("error: no alignment trials", flush=True)
            return 1
        align_best = trials[0]
        movie_start = align_best.movie_start
        pad = align_best.pad
        print(
            f"align pick ms={movie_start} pad={pad} score={align_best.score} "
            f"rooms={align_best.room_order}",
            flush=True,
        )
    assert movie_start is not None

    print(
        f"annotate splice movie_start={movie_start} pad={pad} body={args.body}…",
        flush=True,
    )
    dump_on = [s.strip() for s in args.states_on.split(",") if s.strip()]
    run, trace = resync_and_annotate(
        frames=frames,
        movie_start=movie_start,
        pad=pad,
        body=args.body,
        to=args.to,
        series_stride=args.series_stride,
        dump_states_on=dump_on,
        out_dir=out,
        state_bytes=state_bytes,
        product_frames=product_frames,
        prefix_hops=hops,
    )
    if align_best is not None:
        run.align_best = align_best.to_dict()

    written = write_trace_artifacts(trace, out, write_series=args.series_stride > 0)
    (out / "resync.json").write_text(
        json.dumps(run.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    elapsed = time.perf_counter() - t0
    ann = run.events_summary
    print(f"wrote {out}", flush=True)
    for k, path in written.items():
        print(f"  {k}: {path}", flush=True)
    print(
        f"done product_f={run.product_frames} movie_f={run.movie_frames} "
        f"total_f={run.total_frames} events={ann.get('event_count')} "
        f"rooms={len(run.rooms)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"  by_kind: {ann.get('by_kind')}", flush=True)
    print("  room timeline:", flush=True)
    for r in run.rooms[:30]:
        print(
            f"    f{r['frame']} {r['room_id_hex']} pose={r.get('pose')} "
            f"xy=({r.get('x')},{r.get('y')}) src={r.get('source')}",
            flush=True,
        )
    if run.final:
        fin = run.final
        print(
            f"  final: room={fin.get('room')} pose={fin.get('pose')} "
            f"xy=({fin.get('x')},{fin.get('y')}) items={fin.get('items')}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out),
                "movie_start": movie_start,
                "pad": pad,
                "rooms": [r["room_id_hex"] for r in run.rooms],
                "by_kind": ann.get("by_kind"),
                "elapsed_s": round(elapsed, 2),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
