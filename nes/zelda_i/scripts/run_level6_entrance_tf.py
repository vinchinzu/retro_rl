"""Fixture-live Clean L6: Level6Entrance → TF 0x20.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/run_level6_entrance_tf.py \
        --no-infinite-life --no-video --trials 1

Pin is ``Level6Entrance`` (play 0x79, (120,205), mode 5). Spine CLI has no
``--from-state``. ``route_eligible=false``. Not a Clean STATUS claim.
No health refill. No wooden-arrow poke.
"""

from __future__ import annotations

from bisect import bisect_right

from retro_harness.env import make_env, reset_obs, resync_custom_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.anchors import TF_BIT_L6
from zelda_i.level6.spine import L6_THROUGH, run_level6_from_entrance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_HEART_PARTIAL,
    ADDR_MODE,
    PLAY_MODE,
    health_byte_is_coherent,
    read_snapshot,
)
from zelda_i.runner import (
    add_common_args,
    add_video_args,
    make_assist,
    resolve_video,
    VideoTap,
)

START = "Level6Entrance"
# ``$0012`` while the death spiral plays. The spine has no named constant.
DEATH_MODE = 17


class HealthLedger:
    """Health at every stage boundary, kept as a change log.

    The per-stage damage report names *what* took a heart; it cannot say what
    Link had when the stage started. ``rr-d6v`` is blocked on arrival health
    at 0x78, not on the 0x78 fight, so the in/out column is the measurement
    that decides where to spend the next sitting. Every number this class
    reports is therefore load-bearing and must stay exact.

    What is *not* load-bearing is how much RAM it touches to get there.
    ``route.chain.run_controller_stage`` already builds one full
    ``read_snapshot`` per frame for the controller; this runner used to build
    two more (here and in the death-mode step wrapper), which is 51us of
    pure duplication on every one of tens of thousands of frames.

    Two reductions, neither of which changes a reported number:

    * **Two bytes, not a snapshot.** ``$066F``/``$0670`` are all the ledger
      ever reads, and indexing them is ~140x cheaper than ``read_snapshot``
      (25.6us -> 0.18us measured). ``observe_ram`` also lets the death-mode
      wrapper hand over the array it already fetched, so the whole runner
      does one ``get_ram`` per frame instead of three.
    * **Store on change, not on every frame.** Health is a step function, so
      a change log answers ``at()`` for *any* frame with the same value a
      dense per-frame table would -- stage boundaries included -- while
      holding a few dozen rows instead of one per frame.

    Sampling only on the stage boundaries themselves is not reachable from
    here: ``on_frame`` is the only hook ``run_level6_from_entrance`` exposes
    and it carries no stage identity, so the boundary frames are not known
    until ``run.stages`` exists (see the report for ``route/chain.py``).
    """

    def __init__(self) -> None:
        self._frames: list[int] = []
        self._values: list[tuple[int, int]] = []
        self._ram = None
        self.reads = 0
        self.deaths = 0
        self._saw_death = False

    def observe_ram(self, ram) -> None:
        """Take the array the caller already fetched for this frame.

        Also owns the death count: the step wrapper sees every ``env.step``,
        including the hop ``before``/``after`` steps that never reach
        ``on_frame``, so mode has to be graded here to keep ``deaths``
        identical to the per-frame ``read_snapshot`` it replaces.
        """
        self._ram = ram
        if int(ram[ADDR_MODE]) == DEATH_MODE:
            if not self._saw_death:
                self.deaths += 1
                self._saw_death = True
        else:
            self._saw_death = False

    def on_frame(self, env, obs, action, frame: int) -> None:
        del obs, action
        ram, self._ram = self._ram, None
        if ram is None:
            ram = env.get_ram()
        self.reads += 1
        self._record(int(frame), (int(ram[ADDR_HEALTH]), int(ram[ADDR_HEART_PARTIAL])))

    def _record(self, frame: int, sample: tuple[int, int]) -> None:
        if self._values and self._values[-1] == sample:
            return
        self._frames.append(frame)
        self._values.append(sample)

    def seed(self, env) -> None:
        self.on_frame(env, None, None, 0)

    @property
    def samples(self) -> dict[int, tuple[int, int]]:
        """The change log as ``{frame: sample}`` (one row per change)."""
        return dict(zip(self._frames, self._values, strict=True))

    def at(self, frame: int) -> tuple[int, int] | None:
        """Health in effect at ``frame`` (stage boundaries are exact)."""
        i = bisect_right(self._frames, int(frame)) - 1
        if i < 0:
            return None
        return self._values[i]


def _hearts(sample: tuple[int, int] | None) -> dict | None:
    """``$066F``/``$0670`` as a quotable row. lo nibble is whole hearts."""
    if sample is None:
        return None
    health, partial = sample
    return {
        "health_hex": f"0x{health:02x}",
        "hearts": health & 0x0F,
        "containers": (health >> 4) + 1,
        "partial": f"0x{partial:02x}",
        "value": ((health & 0x0F) << 8) + partial,
    }


def _stage_row(stage, ledger: HealthLedger | None = None) -> dict:
    """Name/frames plus the damage attribution, when the controller kept one.

    A red stage that only reports a tile costs the next sitting a whole
    trial working out what hit Link. ``dungeon.postmortem`` already knows.
    """
    row = {"name": stage.name, "success": stage.success, "frames": stage.frames}
    if ledger is not None:
        row["frame_base"] = stage.frame_base
        row["end_frame"] = stage.end_frame
        row["health_in"] = _hearts(ledger.at(stage.frame_base))
        row["health_out"] = _hearts(ledger.at(stage.end_frame))
    report = getattr(stage.controller, "report", None)
    damage = report().get("damage") if callable(report) else None
    if damage and damage.get("hits"):
        row["damage"] = damage
    return row


def _death_cause(stages) -> str | None:
    for stage in stages:
        report = getattr(stage.controller, "report", None)
        damage = report().get("damage") if callable(report) else None
        if damage and damage.get("death_cause"):
            return damage["death_cause"]
    return None


def _glance(env) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    tf = int(snap.triforce)
    health = int(snap.health)
    return {
        "level": int(snap.level),
        "room": int(snap.screen),
        "mode": int(snap.mode),
        "xy": [int(snap.link_x), int(snap.link_y)],
        "triforce": tf,
        "tf_l6": bool(tf & TF_BIT_L6),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "health": health,
        "health_hex": f"0x{health:02x}",
        "hearts_lo": int(snap.filled_hearts),
        "hearts_hi": int(snap.heart_containers) - 1,
        "health_full": bool(snap.health_is_full),
        "health_coherent": health_byte_is_coherent(health),
        "bow": int(snap.bow),
        "arrows": int(snap.arrows),
        "rod": int(snap.rod),
        "raft": int(snap.raft),
        "ladder": int(snap.ladder),
        "play": snap.mode == PLAY_MODE and not snap.transitioning,
    }


def _open(from_state: str):
    """Load a named pin. ``runner.open_env`` currently ImportErrors (no load_state)."""
    configure_headless()
    env = make_env(GAME, from_state, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    resync_custom_state(env, GAME_DIR, GAME, from_state)
    return env


def run_once(*, from_state: str, tag: str, through: str, clean: bool, video) -> dict:
    env = _open(from_state)
    ledger = HealthLedger()
    orig_step = env.step

    def _step(action, *args, **kwargs):
        # One ``get_ram`` for the whole runner: the ledger grades the death
        # mode off this array and reuses it for the frame's health sample.
        result = orig_step(action, *args, **kwargs)
        ledger.observe_ram(env.get_ram())
        return result

    env.step = _step
    tap = VideoTap(
        video[0],
        video[1],
        tag=tag,
        intro_summary="Clean L6 Entrance -> TF 0x20",
        intro_frames=video[2],
        intervention="Clean no-assist no-pokes",
    )
    try:
        obs, *_ = env.step(nes_idle_action())
        tap.attach(env, obs)
        start = _glance(env)
        assist = None if clean else make_assist(True)
        ledger.seed(env)
        run = run_level6_from_entrance(
            env,
            obs,
            assist=assist,
            through=through,
            poke_arrows=False,
            on_frame=ledger.on_frame,
        )
        end = _glance(env)
        png = RECORDINGS_DIR / f"{tag}_final.png"
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        final_obs, *_ = env.step(nes_idle_action())
        save_rgb_png(final_obs, png)
        leftover = {
            "room": end["room"],
            "mode": end["mode"],
            "x": end["xy"][0],
            "y": end["xy"][1],
            "triforce": end["triforce"],
            "keys": end["keys"],
            "bombs": end["bombs"],
            "health": end["health"],
            "bow": end["bow"],
            "arrows": end["arrows"],
            "rod": end["rod"],
        }
        failed = next((s for s in run.stages if not s.success), None)
        ctrl_rep = {}
        if failed is not None:
            rep = getattr(failed.controller, "report", None)
            ctrl_rep = rep() if callable(rep) else {}
        leftover["objects"] = list(ctrl_rep.get("objects") or [])
        leftover["notes"] = list(ctrl_rep.get("notes") or [])
        leftover["shot_types"] = list(ctrl_rep.get("shot_types") or [])
        leftover["phase"] = ctrl_rep.get("phase")
        leftover["reason_counts"] = dict(ctrl_rep.get("reason_counts") or {})
        leftover["tail"] = list(ctrl_rep.get("tail") or [])
        deaths = ledger.deaths
        ok = bool(run.success) and deaths == 0
        if through == "level6":
            ok = ok and bool(end["tf_l6"])
        body = {
            "ok": ok,
            "route_eligible": False,
            "status_claim": None,
            "track": "clean" if clean else "assisted",
            "from_state": from_state,
            "through": through,
            "frames": run.end_frame,
            "deaths": deaths,
            "failed_stage": run.failed_stage,
            "start": start,
            "final": end,
            "leftover": leftover,
            "stages": [_stage_row(s, ledger) for s in run.stages],
            "death_cause": _death_cause(run.stages),
            "screenshot": str(png.resolve()),
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
        return body
    finally:
        tap.close()
        env.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_state=START, default_tag="l6_entrance_tf_clean")
    parser.set_defaults(infinite_life=False)
    parser.add_argument(
        "--through",
        default="level6",
        choices=tuple(t for t in L6_THROUGH if t != "level6-entry"),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="No infinite-life, no pokes (this runner never assists by default).",
    )
    parser.add_argument(
        "--allow-incoherent-pin",
        action="store_true",
        help=(
            "Run anyway from a pin whose $066F holds more whole hearts than "
            "containers. Every heart number from such a run is against a "
            "fake denominator (rr-d6v); the default refuses."
        ),
    )
    add_video_args(parser, default_on=False)
    args = parser.parse_args(argv)
    clean = bool(args.clean or not args.infinite_life)
    if not args.allow_incoherent_pin:
        env = _open(args.from_state)
        try:
            health = int(read_snapshot(env.get_ram()).health)
        finally:
            env.close()
        if not health_byte_is_coherent(health):
            print(
                f"refusing {args.from_state}: $066F=0x{health:02x} is "
                f"{health & 0x0F} whole hearts in {(health >> 4) + 1} "
                "containers, which normal play cannot reach. Rebuild the pin "
                "with scripts/fixtures/capture_level6_entrance_fixture.py, or "
                "pass --allow-incoherent-pin to measure something else.",
                flush=True,
            )
            return 2
    video = resolve_video(args, default_path=RECORDINGS_DIR / f"{args.tag}.mp4")
    results = []
    for trial in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{trial}"
        body = run_once(
            from_state=args.from_state,
            tag=tag,
            through=args.through,
            clean=clean,
            video=video,
        )
        results.append(body)
        print(
            "OK", body["ok"],
            "TF", body["final"]["tf_l6"],
            "deaths", body["deaths"],
            "failed", body["failed_stage"],
            "leftover", body["leftover"],
            "health", body["final"]["health_hex"],
            flush=True,
        )
        if not body["ok"]:
            return 1
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
