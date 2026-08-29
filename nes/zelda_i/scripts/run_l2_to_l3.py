"""Assisted post-L2 triforce → Level 3 Manji entry (rr-rnx).

Pipeline::

    Level2 TF (0x02) → settle fanfare → OW **0x3C** ~(112,125)
    → reverse L2 door path + 5C maze west → 5B bush leave
    → west forest → door **0x74** → enter level==3 room **0x7c**

Examples::

    # From post-TF settle checkpoint (fastest enter path)
    uv run python nes/zelda_i/scripts/run_l2_to_l3.py --infinite-life \\
        --from-state Level2ExitOverworld --trials 2

    # From mid-fanfare TF collect state
    uv run python nes/zelda_i/scripts/run_l2_to_l3.py --infinite-life \\
        --from-state Level2Complete --trials 1

    # From boom / boss (runs TF collect first)
    uv run python nes/zelda_i/scripts/run_l2_to_l3.py --infinite-life \\
        --from-state Level2Boom --save-state

    # Door screen only (no enter)
    uv run python nes/zelda_i/scripts/run_l2_to_l3.py --infinite-life \\
        --from-state Level2ExitOverworld --door-only
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon.trace import write_state_provenance
from zelda_i.level2.boss_combat import fight_dodongo
from zelda_i.level2.boss_path import BossPathStart, run_boss_path
from zelda_i.level2.boss_tf import collect_and_tf
from zelda_i.level3.overworld import (
    LEVEL3,
    LEVEL3_HOPS_FROM_POST_L2,
    LEVEL3_POST_L2_SCREENS,
    LEVEL2_TRIFORCE_BIT,
    POST_L2_PATH_MAX_FRAMES,
    POST_L2_SETTLE_MAX_FRAMES,
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_LEVEL3_ENTRY_ROOM,
    SCREEN_POST_L2_RETURN,
    OverworldPostL2ToLevel3Controller,
    PostL2TriforceSettleController,
    level3_entrance_success,
    post_l2_overworld_ready,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, PLAY_MODE, read_snapshot, read_u8

# States that already have TF bit 0x02 (or mid-fanfare with it).
_POST_TF_STATES = frozenset(
    {
        "Level2Complete",
        "Level2ExitOverworld",
        "Level2_0D_PostBoss",  # may still need collect
    }
)

def _snap_dict(snap) -> dict:
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "triforce": snap.triforce,
        "tf02": bool(snap.triforce & LEVEL2_TRIFORCE_BIT),
        "sword": snap.sword,
        "health": snap.health,
    }

def _ensure_tf_and_settle(
    env,
    *,
    start_state: str,
    assist: UnlimitedHealthAssist | None,
    tag: str,
) -> dict:
    """Return to OW 0x3C with tf&0x02 from various L2 checkpoints."""
    stages: dict = {}
    snap = read_snapshot(env.get_ram())
    tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))

    # Already on post-L2 OW
    if post_l2_overworld_ready(env.get_ram()):
        stages["skip"] = "already_ow_3c"
        return {"ok": True, "stages": stages, "entry": _snap_dict(snap)}

    # Need TF collect?
    need_collect = not (tf & LEVEL2_TRIFORCE_BIT)
    if need_collect or (
        start_state in ("Level2Boom", "Level2_0E", "Level2_0D_PostBoss")
        and not (tf & LEVEL2_TRIFORCE_BIT)
    ):
        if start_state == "Level2_0D_PostBoss" or snap.screen == 0x0D:
            stages["tf"] = collect_and_tf(env, assist, budget=4000)
        elif start_state == "Level2_0E" or snap.screen == 0x0E:
            fight = fight_dodongo(env, assist, max_frames=14000)
            stages["fight"] = {k: v for k, v in fight.items() if k != "log"}
            if not fight.get("success"):
                return {"ok": False, "stage": "dodongo", "stages": stages}
            stages["tf"] = collect_and_tf(env, assist, budget=4000)
        else:
            path = run_boss_path(env, start=BossPathStart.BOOM, assist=assist)
            stages["dodongo_run"] = {
                "ok": path.get("ok"),
                "reason": path.get("reason"),
            }
            if not path.get("ok"):
                return {"ok": False, "stage": "tf_run", "stages": stages}

        tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        if not (tf & LEVEL2_TRIFORCE_BIT):
            return {
                "ok": False,
                "stage": "tf_collect",
                "stages": stages,
                "final": _snap_dict(read_snapshot(env.get_ram())),
            }

    # Settle fanfare → OW 0x3C
    settle = PostL2TriforceSettleController()
    for fr in range(POST_L2_SETTLE_MAX_FRAMES):
        snap = read_snapshot(env.get_ram())
        act = settle.step(snap)
        env.step(act.action)
        if assist is not None and fr % 15 == 0:
            assist.apply_env(env, frame=fr)
        if settle.success or settle.phase.name == "FAILED":
            break
    # Brief stable play
    for _ in range(30):
        env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

    stages["settle"] = settle.report()
    snap = read_snapshot(env.get_ram())
    ok = post_l2_overworld_ready(env.get_ram())
    return {
        "ok": ok,
        "stages": stages,
        "final": _snap_dict(snap),
    }

def run_once(
    *,
    start_state: str = "Level2ExitOverworld",
    infinite_life: bool = True,
    require_dungeon: bool = True,
    door_only: bool = False,
    save_checkpoint: bool = False,
    max_frames: int = POST_L2_PATH_MAX_FRAMES,
    tag: str = "l2_to_l3",
) -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    track = "assisted" if infinite_life else "clean"
    boom_pre: dict | None = None

    # Level2Boom: library boss path → Level2Complete, then resume this env.
    if start_state == "Level2Boom":
        env_tf = make_env(GAME, "Level2Boom", GAME_DIR, render_mode="rgb_array")
        assist_tf = UnlimitedHealthAssist(enabled=True) if infinite_life else None
        try:
            env_tf.reset()
            env_tf.step(nes_idle_action())
            if assist_tf is not None:
                assist_tf.apply_env(env_tf, frame=0)
            boom_pre = run_boss_path(
                env_tf, start=BossPathStart.BOOM, assist=assist_tf
            )
            if boom_pre.get("ok"):
                save_state(env_tf, GAME_DIR, GAME, "Level2Complete")
        finally:
            env_tf.close()
        if not boom_pre or not boom_pre.get("ok"):
            out = {
                "ok": False,
                "bead": "rr-rnx",
                "track": track,
                "stage": "boom_tf",
                "pre": boom_pre,
            }
            write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
            return out
        start_state = "Level2Complete"

    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    try:
        env.reset()
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = _snap_dict(read_snapshot(env.get_ram()))

        pre = _ensure_tf_and_settle(
            env, start_state=start_state, assist=assist, tag=tag
        )
        if boom_pre is not None:
            pre = {"boom_tf": boom_pre, **pre}
        if not pre.get("ok"):
            snap = read_snapshot(env.get_ram())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_pre_fail.png")
            out = {
                "ok": False,
                "bead": "rr-rnx",
                "track": track,
                "stage": pre.get("stage", "pre"),
                "entry": entry,
                "pre": pre,
                "final": _snap_dict(snap),
            }
            write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
            return out

        # Optional save Level2ExitOverworld after settle
        exit_path = None
        if save_checkpoint and post_l2_overworld_ready(env.get_ram()):
            exit_path = str(save_state(env, GAME_DIR, GAME, "Level2ExitOverworld"))

        snap = read_snapshot(env.get_ram())
        ow_entry = _snap_dict(snap)
        nav = OverworldPostL2ToLevel3Controller(
            require_dungeon=require_dungeon and not door_only,
            require_level3_screen=door_only,
            max_frames=max_frames,
        )
        frames = 0
        trail: list[dict] = []
        last_sc = snap.screen
        while frames < max_frames:
            snap = read_snapshot(env.get_ram())
            if snap.screen != last_sc or snap.level != 0:
                trail.append({"f": frames, **_snap_dict(snap)})
                last_sc = snap.screen
            if nav.success or nav.phase.name == "FAILED":
                break
            if door_only and (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == SCREEN_LEVEL3_ENTRANCE
            ):
                # stop predicate also via nav
                pass
            act = nav.step(snap)
            obs, *_ = env.step(act.action)
            frames += 1
            if assist is not None and frames % 15 == 0:
                assist.apply_env(env, frame=frames)

        snap = read_snapshot(env.get_ram())
        entered = (
            snap.level == LEVEL3
            and snap.mode == PLAY_MODE
            and snap.screen == SCREEN_LEVEL3_ENTRY_ROOM
        )
        door = snap.level == 0 and snap.screen == SCREEN_LEVEL3_ENTRANCE
        ok = entered if (require_dungeon and not door_only) else (door or entered)

        # Settle frames inside dungeon for checkpoint
        state_path = None
        prov = None
        if entered and save_checkpoint:
            for sf in range(200):
                obs, *_ = env.step(nes_idle_action())
                if assist is not None and sf % 15 == 0:
                    assist.apply_env(env, frame=frames + sf)
            snap = read_snapshot(env.get_ram())
            path = save_state(env, GAME_DIR, GAME, "Level3Entrance")
            state_path = str(path)
            prov = str(
                write_state_provenance(
                    path,
                    source_state_path=GAME_DIR
                    / "custom_integrations"
                    / GAME
                    / f"{start_state}.state",
                    request={
                        "segment": "l2_to_l3_enter",
                        "bead": "rr-rnx",
                        "track": track,
                    },
                    selected_trial={
                        "ok": True,
                        "entered_level3": True,
                        "final": _snap_dict(snap),
                        "frames": frames,
                        "nav": nav.report(),
                    },
                )
            )

        save_rgb_png(
            obs, RECORDINGS_DIR / f"{tag}_{'ok' if ok else 'fail'}.png"
        )
        if entered:
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_entrance.png")

        out = {
            "ok": ok,
            "bead": "rr-rnx",
            "track": track,
            "intervention_class": "survival" if infinite_life else "clean",
            "start_state": start_state,
            "path_screens": [f"0x{s:02x}" for s in LEVEL3_POST_L2_SCREENS],
            "n_hops": len(LEVEL3_HOPS_FROM_POST_L2),
            "return_screen": f"0x{SCREEN_POST_L2_RETURN:02x}",
            "door_screen": f"0x{SCREEN_LEVEL3_ENTRANCE:02x}",
            "entry_room": f"0x{SCREEN_LEVEL3_ENTRY_ROOM:02x}",
            "entry": entry,
            "pre": pre,
            "ow_entry": ow_entry,
            "nav": nav.report(),
            "trail": trail[-60:],
            "frames": frames,
            "entered_level3": entered,
            "door_reached": door or entered,
            "final": _snap_dict(snap),
            "level2_exit_state": exit_path,
            "level3_entrance_state": state_path,
            "provenance": prov,
            "natural_entry": False,
            "evidence": [f"recordings/{tag}.json"],
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
        return out
    finally:
        env.close()

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-state",
        default="Level2ExitOverworld",
        help="Level2ExitOverworld | Level2Complete | Level2_0E | Level2_0D_PostBoss | …",
    )
    p.add_argument("--infinite-life", action="store_true", default=True)
    p.add_argument("--no-infinite-life", action="store_true")
    p.add_argument("--door-only", action="store_true")
    p.add_argument(
        "--enter-only",
        action="store_true",
        help="Alias: assume already on post-L2 OW (same as default path from Exit)",
    )
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--max-frames", type=int, default=POST_L2_PATH_MAX_FRAMES)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--tag", default="l2_to_l3")
    args = p.parse_args(argv)

    inf = not args.no_infinite_life
    results = []
    for t in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{t}"
        r = run_once(
            start_state=args.from_state,
            infinite_life=inf,
            require_dungeon=not args.door_only,
            door_only=args.door_only,
            save_checkpoint=args.save_state and t == 0,
            max_frames=args.max_frames,
            tag=tag,
        )
        results.append(r)
        fin = r.get("final") or {}
        print(
            f"trial{t}: ok={r.get('ok')} entered={r.get('entered_level3')} "
            f"lv={fin.get('level')} sc={fin.get('screen')} "
            f"nav_phase={(r.get('nav') or {}).get('phase')} "
            f"hops={(r.get('nav') or {}).get('hop_index')}"
        )

    n_ok = sum(1 for r in results if r.get("ok"))
    summary = {
        "ok": n_ok == len(results) and n_ok > 0,
        "bead": "rr-rnx",
        "track": "assisted" if inf else "clean",
        "trials": len(results),
        "n_ok": n_ok,
        "start_state": args.from_state,
        "return_screen": f"0x{SCREEN_POST_L2_RETURN:02x}",
        "path_screens": [f"0x{s:02x}" for s in LEVEL3_POST_L2_SCREENS],
        "results": [
            {
                "ok": r.get("ok"),
                "entered": r.get("entered_level3"),
                "final": r.get("final"),
                "frames": r.get("frames"),
                "nav_phase": (r.get("nav") or {}).get("phase"),
            }
            for r in results
        ],
        "evidence": [f"recordings/{args.tag}.json"]
        + (
            [f"recordings/{args.tag}_t{i}.json" for i in range(args.trials)]
            if args.trials > 1
            else []
        ),
    }
    # Canonical evidence name for the bead
    out_path = RECORDINGS_DIR / "l2_to_l3_assisted.json"
    if args.trials >= 2 or args.tag == "l2_to_l3":
        write_json_report(out_path, summary)
        print(f"wrote {out_path}")
    print(f"summary: {n_ok}/{len(results)} ok")
    return 0 if n_ok == len(results) and n_ok > 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
