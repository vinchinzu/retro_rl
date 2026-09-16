"""Continuous Survival spine: power-on, one emulator session, no stitch.

    uv run python nes/zelda_i/scripts/run_survival_spine.py --trials 1
    uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1
    uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --no-video --trials 1
    uv run python nes/zelda_i/scripts/run_survival_spine.py --through pre-l1 --headed --no-video --trials 1

Power-on first file slot / first quest. Records MP4 + room-transition PNGs
unless ``--no-video``. ``--headed`` opens a pygame window (``[ ]`` speed,
TAB turbo, ESC quit) and skips dummy SDL. Heart assist is on by default;
``--no-infinite-life`` turns it off for combat practice. ``--through pre-l1``
forces it off: the refill hides the ``$0670`` chip that zeros the 10-kill
5-rupee. Inventory pokes stay on unless ``--no-pokes``; ``--clean`` is both
off. Does not overwrite Clean M5.
No ``--from-state``. Stop at first failed stage.
"""

from __future__ import annotations

import argparse

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
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_HELP_DROP_COUNT, ADDR_WORLD_KILL_COUNT, read_snapshot
from zelda_i.runner import VideoTap, add_video_args, resolve_video
from zelda_i.spine.survival import (
    SPINE_THROUGH,
    run_survival_spine,
    spine_final_fields,
    validate_l5_endpoint,
)


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
    add_video_args(parser, default_on=True)
    add_headed_flag(parser)
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
        tap = VideoTap(
            video_path,
            video_config,
            tag=tag,
            intro_summary="Survival continuous spine, first quest, first file",
            intro_frames=intro,
        )
        allow_pokes = not args.no_pokes and not args.clean
        infinite_life = bool(args.infinite_life) and not args.clean
        if args.through == "pre-l1":
            # Survival refill writes $0670 back to $FF the same frame
            # Link_BeHarmed zeros $50/$627. The bomb walk is a Clean farm.
            # No heart assist, no inventory pokes. Ever.
            infinite_life = False
            allow_pokes = False
        assist = (
            UnlimitedHealthAssist(enabled=True) if infinite_life else None
        )
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
                    title=f"Zelda I BOT: {args.through} (no assist)",
                    hud=_headed_hud,
                )
            tap.attach(env, obs)
            # VideoTap wraps env.step; do not also pass on_frame (double encode).
            run = run_survival_spine(
                env,
                obs,
                assist=assist,
                through=args.through,
                allow_pokes=allow_pokes,
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
            }
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
        if args.through == "level5" and payload.get("ok"):
            validate_l5_endpoint(payload)
        write_json_report(RECORDINGS_DIR / f"{tag}.json", payload)
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
