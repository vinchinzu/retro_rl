"""Assisted Survival compose: L2 exit → L3 Raft → Manhandla → TF 0x04.

Thin env+assist+report wrapper over existing library controllers. One
emulator session, no mid-run state load, no inventory pokes.

Default start ``Level2ExitOverworld`` (post-Moon TF). ``Level3Entrance``
skips the OW hop. ``Level3Raft`` is the longest isolated suffix (0 bombs
on that fixture — expect bomb-wall fail without a poke).

Not Clean STATUS. Examples::

    uv run python nes/zelda_i/scripts/run_level3_complete.py \\
        --infinite-life --video --trials 1
    uv run python nes/zelda_i/scripts/run_level3_complete.py \\
        --from-state Level3Entrance --infinite-life --video
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level3_boss_path import Level3BossPathController
from zelda_i.level3_dungeon import (
    NORTH_ENTER_MAX_FRAMES,
    NORTH_EXIT_6B_MAX_FRAMES,
    RAFT_PATH_MAX_FRAMES,
    ROOM_6B_SPEC,
    ROOM_7B_SPEC,
    WEST_ENTER_MAX_FRAMES,
    Level3NorthChainController,
    Level3RaftPathController,
    Level3WestKeyController,
    level3_has_raft,
    level3_reached_5b,
    level3_room_7b_key_success,
)
from zelda_i.level3_overworld import (
    LEVEL3,
    POST_L2_PATH_MAX_FRAMES,
    OverworldPostL2ToLevel3Controller,
    post_l2_overworld_ready,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_RAFT, ADDR_TRIFORCE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.runner import VideoTap, add_video_args, controller_stopped, resolve_video

LEVEL3_TF_BIT = 0x04


def _brief(env) -> dict:
    snap = read_snapshot(env.get_ram())
    ram = env.get_ram()
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "health": snap.health,
        "triforce": int(read_u8(ram, ADDR_TRIFORCE)),
        "raft": int(read_u8(ram, ADDR_RAFT)),
    }


def _run_frames(env, controller, assist, *, max_frames: int, step=None) -> str:
    last = ""
    for _ in range(max_frames):
        snap = read_snapshot(env.get_ram())
        action = step(snap) if step is not None else controller.step(snap)
        last = action.reason
        env.step(action.action)
        if assist is not None:
            assist.apply_env(env, frame=controller.frames)
        if controller_stopped(controller):
            break
    return last


def run_once(
    *,
    start_state: str = "Level2ExitOverworld",
    infinite_life: bool = True,
    tag: str = "level3_complete_assisted",
    video_path=None,
    video_config=None,
    intro_frames: int = 0,
) -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    tap = VideoTap(
        video_path,
        video_config,
        tag=tag,
        intro_summary="Survival L2 exit -> L3 Raft -> Manhandla -> TF 0x04",
        intro_frames=intro_frames,
    )
    stages: dict = {}
    skipped: list[str] = []
    error = None
    obs = None
    try:
        env.reset()
        obs, *_ = env.step(nes_idle_action())
        tap.attach(env, obs)
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = _brief(env)
        snap = read_snapshot(env.get_ram())

        # --- OW hop ---
        in_l3 = snap.level == LEVEL3 and snap.mode == PLAY_MODE
        if not in_l3:
            if not post_l2_overworld_ready(env.get_ram()) and snap.level != 0:
                error = (
                    f"unexpected start loc level={snap.level} "
                    f"sc=0x{snap.screen:02x} mode={snap.mode}"
                )
            else:
                nav = OverworldPostL2ToLevel3Controller(require_dungeon=True)
                _run_frames(env, nav, assist, max_frames=POST_L2_PATH_MAX_FRAMES)
                stages["ow"] = nav.report()
                snap = read_snapshot(env.get_ram())
                in_l3 = (
                    snap.level == LEVEL3
                    and snap.mode == PLAY_MODE
                    and snap.screen == 0x7C
                )
                if not in_l3:
                    error = "ow_enter_failed"
        else:
            skipped.append("ow")

        # --- west key ---
        if error is None and not level3_has_raft(env.get_ram()):
            snap = read_snapshot(env.get_ram())
            if snap.screen == 0x7C:
                west = Level3WestKeyController()
                _run_frames(
                    env,
                    west,
                    assist,
                    max_frames=WEST_ENTER_MAX_FRAMES + ROOM_7B_SPEC.max_frames,
                )
                stages["west_key"] = west.report()
                if not (west.success or level3_room_7b_key_success(env.get_ram())):
                    error = "west_key_failed"
            else:
                skipped.append("west_key")

        # --- north chain ---
        if error is None and not level3_has_raft(env.get_ram()):
            snap = read_snapshot(env.get_ram())
            if snap.screen in (0x7B, 0x6B) or (
                snap.screen == 0x5B and not level3_reached_5b(env.get_ram())
            ):
                north = Level3NorthChainController()
                _run_frames(
                    env,
                    north,
                    assist,
                    max_frames=(
                        NORTH_ENTER_MAX_FRAMES
                        + ROOM_6B_SPEC.max_frames
                        + NORTH_EXIT_6B_MAX_FRAMES
                    ),
                )
                for _ in range(180):
                    env.step(nes_idle_action())
                    if assist is not None:
                        assist.apply_env(env, frame=north.frames)
                stages["north_chain"] = north.report()
                if not (north.success or level3_reached_5b(env.get_ram())):
                    # x=112 north-wall trap: south-band, center x≈120, push UP.
                    nudged = False
                    for _ in range(1600):
                        s = read_snapshot(env.get_ram())
                        if (
                            s.level == LEVEL3
                            and s.screen == 0x5B
                            and s.mode == PLAY_MODE
                            and not s.transitioning
                        ):
                            nudged = True
                            break
                        if s.mode in (6, 7, 16) or s.transitioning:
                            env.step(nes_action("UP"))
                        elif s.mode != PLAY_MODE:
                            env.step(nes_idle_action())
                        elif abs(s.link_x - 120) > 4 and s.link_y < 180:
                            env.step(nes_action("DOWN"))
                        elif abs(s.link_x - 120) > 4:
                            env.step(
                                nes_action("RIGHT" if s.link_x < 120 else "LEFT")
                            )
                        else:
                            env.step(nes_action("UP"))
                        if assist is not None:
                            assist.apply_env(env, frame=north.frames)
                    stages["north_nudge"] = {
                        "ok": nudged or level3_reached_5b(env.get_ram())
                    }
                    if not (nudged or level3_reached_5b(env.get_ram())):
                        error = "north_chain_failed"
            elif snap.screen == 0x5B:
                skipped.append("north_chain")
            elif snap.screen not in (0x0F, 0x59, 0x69, 0x5A):
                if not level3_reached_5b(env.get_ram()):
                    skipped.append("north_chain")

        # --- raft ---
        if error is None and not level3_has_raft(env.get_ram()):
            raft = Level3RaftPathController()

            def _raft_step(s):
                return raft.step(s, has_raft=level3_has_raft(env.get_ram()))

            _run_frames(
                env, raft, assist, max_frames=RAFT_PATH_MAX_FRAMES, step=_raft_step
            )
            for _ in range(30):
                env.step(nes_idle_action())
                if assist is not None:
                    assist.apply_env(env, frame=raft.frames)
            stages["raft"] = raft.report()
            if not level3_has_raft(env.get_ram()):
                error = "raft_failed"
        elif error is None:
            skipped.append("raft")

        # --- boss / TF ---
        tf04 = bool(int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & LEVEL3_TF_BIT)
        if error is None and not tf04:
            boss = Level3BossPathController(poke_bombs=None, tag=tag)
            total = [tap.frame]
            if read_snapshot(env.get_ram()).screen != 0x4D:
                p5 = boss.path_to_5d(env, assist, total)
                stages["path_to_5d"] = {
                    "ok": p5.get("ok"),
                    "error": p5.get("error"),
                    "final": p5.get("final"),
                }
                if not p5.get("ok"):
                    error = p5.get("error") or "path_to_5d_failed"
                else:
                    gate = boss.open_5d_up(env, assist, total)
                    stages["gate_5d"] = {
                        "ok": gate.get("ok"),
                        "method": gate.get("method"),
                        "error": gate.get("error"),
                    }
                    if not gate.get("ok"):
                        error = gate.get("error") or "gate_5d_failed"
            if error is None:
                fight = boss.fight_manhandla(env, assist, total, max_frames=16000)
                stages["fight"] = {
                    "ok": fight.get("ok"),
                    "tf04": fight.get("tf04"),
                    "frames": fight.get("frames"),
                    "error": fight.get("error"),
                    "notes": fight.get("notes"),
                }
                tf04 = bool(fight.get("tf04"))
                if not tf04:
                    error = fight.get("error") or "tf04_failed"
            stages["boss"] = boss.report()

        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(
            obs, RECORDINGS_DIR / f"{tag}_{'ok' if tf04 else 'fail'}.png"
        )
        final = _brief(env)
        out = {
            "ok": bool(tf04),
            "bead": "rr-4d53.3",
            "track": "assisted" if infinite_life else "clean",
            "intervention_class": "survival" if infinite_life else "clean",
            "status_promote": False,
            "natural_entry": False,
            "start_state": start_state,
            "compose_scope": f"{start_state} -> L3 TF 0x04 continuous",
            "poke": False,
            "entry": entry,
            "skipped": skipped,
            "stages": stages,
            "error": error,
            "final": final,
            "triforce": final["triforce"],
            "triforce_bit_0x04": tf04,
            "assist": assist.report() if assist is not None else None,
            "video": tap.close(),
            "evidence": [f"recordings/{tag}.json"],
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
        return out
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level2ExitOverworld")
    p.add_argument("--infinite-life", action="store_true", default=True)
    p.add_argument("--no-infinite-life", action="store_true")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--tag", default="level3_complete_assisted")
    add_video_args(p)
    args = p.parse_args(argv)
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
            video_path=trial_video,
            video_config=video_config,
            intro_frames=intro_frames,
        )
        results.append(r)
        print(
            f"trial{t}: ok={r.get('ok')} tf={r.get('triforce')} "
            f"err={r.get('error')} final={r.get('final')} "
            f"video={(r.get('video') or {}).get('path')}"
        )
    n_ok = sum(1 for r in results if r.get("ok"))
    payload = {
        "ok": n_ok == len(results) and n_ok > 0,
        "bead": "rr-4d53.3",
        "track": "assisted" if inf else "clean",
        "intervention_class": "survival" if inf else "clean",
        "status_promote": False,
        "natural_entry": False,
        "start_state": args.from_state,
        "trials": len(results),
        "ok_count": n_ok,
        "triforce_bit_0x04": n_ok > 0,
        "poke": False,
        "assist": next((r.get("assist") for r in results if r.get("assist")), None),
        "video": next((r.get("video") for r in results if r.get("video")), None),
        "results": [
            {
                "ok": r.get("ok"),
                "triforce": r.get("triforce"),
                "error": r.get("error"),
                "final": r.get("final"),
                "skipped": r.get("skipped"),
                "video": r.get("video"),
            }
            for r in results
        ],
        "evidence": [f"recordings/{args.tag}.json"],
    }
    write_json_report(RECORDINGS_DIR / f"{args.tag}_summary.json", payload)
    print(f"summary: {n_ok}/{len(results)} TF 0x04")
    return 0 if n_ok == len(results) and n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
