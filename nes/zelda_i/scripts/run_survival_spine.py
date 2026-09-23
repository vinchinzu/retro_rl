"""Continuous Survival spine: power-on, one emulator session, no stitch.

    uv run python nes/zelda_i/scripts/run_survival_spine.py --trials 1
    uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1
    uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --no-video --trials 1
    uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --headed --no-video --trials 1
    uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --rollout --headed --no-video --trials 1
    uv run python nes/zelda_i/scripts/run_survival_spine.py --through gather --no-video --trials 1
    uv run python nes/zelda_i/scripts/run_survival_spine.py --no-gather --no-video --trials 1

Gathering is the default prefix: pre-l1 bombs, the gather chain to the L1
mouth 0x37 (6 containers, White Sword, Blue Ring), then L1 from its door. The chain's
own health refill engages at ``--gather-engage-hearts`` (default 2; 1 is
last-heart, 0 is off; ``--clean`` sets 0). ``--no-gather`` is the legacy
wooden-sword prefix.

Power-on first file slot / first quest. Records MP4 + room-transition PNGs
unless ``--no-video``. ``--headed`` opens a pygame window (``[ ]`` speed,
TAB turbo, ESC quit) and skips dummy SDL. Heart assist is on by default;
``--no-infinite-life`` turns it off for combat practice. ``--through pre-l1``
forces it off: the refill hides the ``$0670`` chip that zeros the 10-kill
5-rupee. Pre-l1 still forces heart assist and bomb/key/food pokes off, and
writes the rupee count up to the 20R pack price before ``bomb_topup``.
Inventory pokes stay on unless ``--no-pokes``; ``--clean`` is both
off. Does not overwrite Clean M5.
No ``--from-state``. Stop at first failed stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs
from retro_harness.headed import (
    HEADED_ATTR,
    add_headed_flag,
    attach_headed,
    idle_headed,
)
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.combat import facing_to_direction
from zelda_i.overworld.path import OverworldPathController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_HELP_DROP_COUNT, ADDR_WORLD_KILL_COUNT, read_snapshot
from zelda_i.spine.ledger import RunLedger
from zelda_i.runner import VideoTap, add_video_args, resolve_video
from zelda_i.spine.survival import (
    GATHER_ENGAGE_HEARTS,
    SPINE_THROUGH,
    gather_assist,
    run_survival_spine,
    spine_final_fields,
)
from zelda_i.level5.spine import validate_l5_endpoint


def _spine_kills(payload: dict) -> int:
    """Kills banked across every hunting stage of this trial.

    The census lives on the stage controller (``overworld.hunt.ScreenHunter``),
    so a stage that does not hunt contributes nothing rather than a zero that
    reads like a measurement.
    """
    return sum(
        int((stage.get("controller") or {}).get("kills", 0))
        for stage in payload.get("stages", [])
    )


def _headed_hud(env) -> str:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    headed = getattr(env, HEADED_ATTR, None)
    frame = int(getattr(headed, "frame", 0) or 0)
    try:
        face = facing_to_direction(int(snap.facing))[0]
    except ValueError:
        face = "?"
    assist = getattr(env, "_zelda_assist", "?")
    return (
        f"f{frame} assist={assist} "
        f"0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) {face} "
        f"{snap.rupees}R {snap.whole_hearts}/{snap.heart_containers}H "
        f"b{snap.bombs} k{int(ram[ADDR_HELP_DROP_COUNT])}/"
        f"s{int(ram[ADDR_WORLD_KILL_COUNT])}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=SPINE_THROUGH, default="level1")
    parser.add_argument("--tag", default="survival_spine")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--infinite-life",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Survival health refill (default on). "
            "--no-infinite-life turns assist off for combat practice."
        ),
    )
    parser.add_argument(
        "--no-pokes",
        action="store_true",
        help=(
            "Skip Survival inventory pokes (bomb/key/rupee top-ups, Food, "
            "wooden arrows). Heart assist stays on unless --no-infinite-life."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Alias: --no-infinite-life and --no-pokes. "
            "Does not change Survival defaults when omitted."
        ),
    )
    parser.add_argument(
        "--gather",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gathering prefix before L1 (default on). --no-gather is legacy.",
    )
    parser.add_argument(
        "--gather-engage-hearts",
        type=int,
        default=GATHER_ENGAGE_HEARTS,
        help="Gather-chain health refill at N whole hearts (1 last-heart, 0 off).",
    )
    parser.add_argument(
        "--engage-hearts",
        type=int,
        default=None,
        help=(
            "Whole-run refill only at N whole hearts (1 = last heart), so the "
            "refill count is deaths prevented. Default: refill after every hit."
        ),
    )
    parser.add_argument(
        "--observed-damage-guard",
        action="store_true",
        help=(
            "Raise the refill floor to the largest survived single hit so far; "
            "report safety refills separately from the requested threshold."
        ),
    )
    parser.add_argument(
        "--save-points",
        nargs="?",
        const="Spine",
        default=None,
        help="Write <PREFIX>_<stage>.state at every stage start (default prefix Spine).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="STAGE",
        help="Load the STAGE save point and play on from it (dev tape, not continuous).",
    )
    parser.add_argument(
        "--trace",
        default=None,
        metavar="PATH",
        help="Write the per-frame (frame, room, x, y, mode, buttons) tape as JSON.",
    )
    add_video_args(parser, default_on=True)
    add_headed_flag(parser)
    parser.add_argument(
        "--rollout",
        action="store_true",
        help=(
            "Opt-in ROM-truth evader on OverworldPathController. "
            "Default reactive arm is unchanged. A/B only."
        ),
    )
    args = parser.parse_args(argv)

    headed = bool(args.headed)
    if not headed:
        configure_headless()
    results: list[dict] = []
    for trial in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{trial}"
        video_path, video_config, intro = resolve_video(
            args,
            default_path=RECORDINGS_DIR / f"{tag}.mp4",
        )
        env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
        if args.rollout:
            # Bind the inner emulator: a rollout is a restore, not a STATUS
            # mid-run load. The honest cost is controller.report()["rollout"].
            raw = env
            _orig_step = OverworldPathController.step

            def _step(self, snap, **kw):  # type: ignore[no-untyped-def]
                if self._rollout is None:
                    self.attach_rollout(raw)
                return _orig_step(self, snap, **kw)

            OverworldPathController.step = _step  # type: ignore[assignment]
        tap = VideoTap(
            video_path,
            video_config,
            tag=tag,
            intro_summary="Survival continuous spine, first quest, first file",
            intro_frames=intro,
        )
        allow_pokes = not args.no_pokes and not args.clean
        infinite_life = bool(args.infinite_life) and not args.clean
        gather_engage = 0 if args.clean else int(args.gather_engage_hearts)
        if args.through == "pre-l1":
            # Survival refill writes $0670 back to $FF the same frame
            # Link_BeHarmed zeros $50/$627. The bomb walk is a Clean farm.
            # No heart assist, no inventory pokes. Ever.
            infinite_life = False
            allow_pokes = False
        if not infinite_life:
            assist = None
        elif args.engage_hearts:
            assist = gather_assist(
                args.engage_hearts,
                observed_damage_guard=args.observed_damage_guard,
            )
        else:
            assist = UnlimitedHealthAssist(enabled=True)
        payload: dict | None = None
        pygame_mod = None
        try:
            obs, _ = reset_obs(env)
            env = AuditedEnv(
                env,
                capabilities=AuditCapabilities.all("zelda_i.survival_spine"),
            )
            env._zelda_assist = "off" if assist is None else "on"
            if headed:
                pygame_mod = attach_headed(
                    env,
                    title=f"Zelda I BOT: {args.through} (assist {env._zelda_assist})",
                    hud=_headed_hud,
                )
            tap.attach(env, obs)
            ledger = RunLedger(trace=[] if args.trace else None)
            ledger.attach(env)
            # VideoTap wraps env.step; do not also pass on_frame (double encode).
            run = run_survival_spine(
                env,
                obs,
                assist=assist,
                through=args.through,
                allow_pokes=allow_pokes,
                gather=bool(args.gather),
                gather_engage_hearts=gather_engage,
                save_points=args.save_points,
                resume_from=args.resume,
            )
            run.apply_state_audit(int(env.audit().mid_run_loads or 0))
            final_ram = env.get_ram()
            snap = read_snapshot(final_ram)
            screenshot = RECORDINGS_DIR / f"{tag}_final.png"
            save_rgb_png(run.obs, screenshot)
            payload = {
                **run.report(),
                "trial": trial,
                "final": spine_final_fields(snap, final_ram),
                "screenshot": str(screenshot),
                "assist": None if assist is None else assist.report(),
                "ledger": ledger.report(),
            }
            for line in ledger.summary_lines():
                print(line)
        finally:
            try:
                video_info = tap.close()
            except Exception:
                tap.abort()
                video_info = {
                    "path": None,
                    "encoded_frames": 0,
                    "intro_frames": tap.intro_written,
                    "gameplay_frames": tap.frame,
                    "transitions": list(tap.transitions),
                }
            if pygame_mod is not None:
                try:
                    idle_headed(env, pygame_mod)
                except KeyboardInterrupt:
                    pass
            env.close()
        if payload is None:
            raise RuntimeError("survival spine trial ended before a report")
        payload["video"] = video_info
        if args.through == "level5" and payload.get("ok") and not args.resume:
            validate_l5_endpoint(payload)
        write_json_report(RECORDINGS_DIR / f"{tag}.json", payload)
        if args.trace:
            Path(args.trace).write_text(json.dumps(ledger.trace))
        results.append(payload)
        video = payload.get("video") or {}
        kills = _spine_kills(payload)
        print(
            f"trial{trial}: ok={payload['ok']} failed={payload.get('failed_stage')} "
            f"tf={payload['final']['triforce']} room=0x{payload['final']['room']:02x} "
            f"keys={payload['final']['keys']} bombs={payload['final']['bombs']} "
            f"rupees={payload['final']['rupees']} kills={kills} "
            f"set_state={payload.get('set_state_count')} "
            f"boot={payload.get('boot_policy')} video={video.get('path')}"
        )
    n_ok = sum(1 for row in results if row.get("ok"))
    print(f"summary: {n_ok}/{len(results)} continuous Survival {args.through}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
