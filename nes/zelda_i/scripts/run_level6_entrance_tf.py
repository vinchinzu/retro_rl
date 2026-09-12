"""Fixture-live Clean L6: Level6Entrance → TF 0x20.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/run_level6_entrance_tf.py \
        --no-infinite-life --no-video --trials 1

Pin is ``Level6Entrance`` (play 0x79, (120,205), mode 5). Spine CLI has no
``--from-state``. ``route_eligible=false``. Not a Clean STATUS claim.
No health refill. No wooden-arrow poke.
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs, resync_custom_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.anchors import TF_BIT_L6
from zelda_i.level6.spine import L6_THROUGH, run_level6_from_entrance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import (
    add_common_args,
    add_video_args,
    make_assist,
    resolve_video,
    VideoTap,
)

START = "Level6Entrance"


def _stage_row(stage) -> dict:
    """Name/frames plus the damage attribution, when the controller kept one.

    A red stage that only reports a tile costs the next sitting a whole
    trial working out what hit Link. ``dungeon.postmortem`` already knows.
    """
    row = {"name": stage.name, "success": stage.success, "frames": stage.frames}
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
    deaths = 0
    saw_death = False
    orig_step = env.step

    def _step(action, *args, **kwargs):
        nonlocal deaths, saw_death
        result = orig_step(action, *args, **kwargs)
        mode = int(read_snapshot(env.get_ram()).mode)
        if mode == 17 and not saw_death:
            deaths += 1
            saw_death = True
        elif mode != 17:
            saw_death = False
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
        run = run_level6_from_entrance(
            env,
            obs,
            assist=assist,
            through=through,
            poke_arrows=False,
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
            "stages": [_stage_row(s) for s in run.stages],
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
    add_video_args(parser, default_on=False)
    args = parser.parse_args(argv)
    clean = bool(args.clean or not args.infinite_life)
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
