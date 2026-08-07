"""Probe post-0x4A overworld path toward Level 2 door (0x3C) and entry.

Clean path (default, no assist)::

    farm hearts on 0x4A → rejoin 0x49→0x59→0x5A @y≈140 → corridor clear on
    0x5A → LEVEL2_CLEAN_FROM_5A_TO_3C (maze on 0x5C) → Moon door 0x3C.

Assisted first-pass still available with ``--infinite-life`` (skips farm/clear).

Examples::

    uv run python zelda_i/scripts/probe_level2_suffix.py --tag l2_clean_t0
    uv run python zelda_i/scripts/probe_level2_suffix.py --from-state At4A --tag l2_clean_at4a_t0
    uv run python zelda_i/scripts/probe_level2_suffix.py --infinite-life --enter-dungeon
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.chain import run_controller_stage
from zelda_i.heart_farm import HeartFarmController
from zelda_i.level2_clean_door import run_clean_door_from_env
from zelda_i.level2_overworld import (
    SEGMENT_MAX_FRAMES,
    OverworldToLevel2Controller,
    level2_door_hops_from,
    level2_path_prefix_success,
    post_triforce_overworld_ready,
    PostTriforceSettleController,
    SETTLE_MAX_FRAMES,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot


def _snapshot_dict(snap) -> dict:
    objs = [
        {
            "slot": o.slot,
            "type": o.type_id,
            "x": o.x,
            "y": o.y,
            "hp": o.hp,
        }
        for o in snap.objects
        if o.slot >= 1 and o.type_id not in (0, 0xFF) and o.y > 0
    ][:12]
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "health": snap.health,
        "hearts": f"{snap.filled_hearts}/{snap.heart_containers}",
        "filled_hearts": snap.filled_hearts,
        "heart_containers": snap.heart_containers,
        "sword": snap.sword,
        "bombs": snap.bombs,
        "keys": snap.keys,
        "triforce": snap.triforce,
        "objects": objs,
    }


def _run_nav_loop(env, obs, nav, *, max_frames, assist, trail, tag, stop_pred=None):
    """Step nav until success/fail/death/max_frames. Mutates trail; returns obs."""
    frames = 0
    snap = read_snapshot(env.get_ram())
    last_screen = snap.screen
    while frames < max_frames:
        snap = read_snapshot(env.get_ram())
        if snap.screen != last_screen:
            trail.append({"f": frames, **_snapshot_dict(snap)})
            last_screen = snap.screen
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_sc{snap.screen:02x}.png")
        if snap.mode == 17:
            break
        if stop_pred is not None and stop_pred(snap):
            break
        if nav.success or nav.phase.name == "FAILED":
            break
        act = nav.step(snap)
        obs, *_ = env.step(act.action)
        frames += 1
        if assist is not None:
            assist.apply_env(env, frame=frames)
        if nav.success or nav.phase.name == "FAILED":
            break
    return obs, frames


def run_probe(
    *,
    start_state: str,
    max_frames: int,
    stop_screen: int,
    enter_dungeon: bool,
    farm_hearts_min: int,
    farm_max_frames: int,
    corridor_clear_frames: int,
    save_checkpoint: bool,
    tag: str,
    infinite_life: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    if infinite_life:
        farm_hearts_min = 0
        corridor_clear_frames = 0
    track = "assisted" if infinite_life else "clean"
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        entry = _snapshot_dict(read_snapshot(env.get_ram()))
        trail: list[dict] = []
        total_frames = 0

        if start_state == "Level1ExitOverworld" and not post_triforce_overworld_ready(
            env.get_ram()
        ):
            settle = PostTriforceSettleController()
            obs, _ = run_controller_stage(
                env,
                obs,
                name="settle",
                controller=settle,
                max_frames=SETTLE_MAX_FRAMES,
                assist=assist,
            )

        snap = read_snapshot(env.get_ram())
        prefix_report = None
        door_or_farm_screens = {
            0x4A,
            0x49,
            0x59,
            0x5A,
            0x5B,
            0x5C,
            0x5D,
            0x4D,
            0x4C,
            0x3C,
        }
        if snap.level == 0 and snap.screen not in door_or_farm_screens:
            nav = OverworldToLevel2Controller()
            obs, stage = run_controller_stage(
                env,
                obs,
                name="prefix",
                controller=nav,
                max_frames=SEGMENT_MAX_FRAMES,
                assist=assist,
            )
            prefix_report = stage.report()
            snap = read_snapshot(env.get_ram())
            if not (level2_path_prefix_success(env.get_ram()) or nav.success):
                png = RECORDINGS_DIR / f"{tag}_prefix_fail.png"
                save_rgb_png(obs, png)
                return {
                    "ok": False,
                    "track": track,
                    "stage": "prefix",
                    "entry": entry,
                    "prefix": prefix_report,
                    "assist": assist.report() if assist else None,
                    "final": _snapshot_dict(snap),
                    "screenshot": str(png),
                }

        # --- Clean path: farm + y140 east + 0x5A clear + maze door (timing-locked) ---
        if track == "clean" and not enter_dungeon and snap.screen in (
            0x4A,
            0x49,
            0x59,
        ):
            # Ensure we start clean runner from 0x4A when possible.
            if snap.screen != 0x4A and snap.screen in (0x49, 0x59):
                # Fall through: runner expects 0x4A farm; rejoin still works from
                # mid-path if already past farm (farm no-ops when hearts ok / off 4A).
                pass
            trail_c: list[dict] = []
            obs, clean_rep = run_clean_door_from_env(
                env,
                obs,
                farm_hearts_min=farm_hearts_min,
                farm_max_frames=farm_max_frames,
                corridor_clear_frames=corridor_clear_frames,
                max_door_frames=max_frames,
                trail=trail_c,
            )
            snap = read_snapshot(env.get_ram())
            png = RECORDINGS_DIR / f"{tag}_final.png"
            save_rgb_png(obs, png)
            ok = bool(clean_rep.get("ok"))
            fin = clean_rep.get("final") or _snapshot_dict(snap)
            if isinstance(fin, dict) and "hearts" in fin and "filled_hearts" not in fin:
                pass
            else:
                fin = _snapshot_dict(snap)
            checkpoint = None
            if ok and save_checkpoint:
                checkpoint = str(save_state(env, GAME_DIR, GAME, f"OW_{snap.screen:02X}"))
            return {
                "ok": ok,
                "track": track,
                "infinite_life": False,
                "entry": entry,
                "prefix": prefix_report,
                "farm": clean_rep.get("farm"),
                "corridor_clear": {
                    "notes": [
                        n for n in (clean_rep.get("notes") or []) if n.startswith("clear")
                    ]
                },
                "trail": trail_c,
                "nav": clean_rep.get("door"),
                "notes": clean_rep.get("notes"),
                "assist": None,
                "final": fin,
                "screenshot": str(png),
                "checkpoint": checkpoint,
                "frames": len(trail_c),
                "farm_hearts_min": farm_hearts_min,
                "corridor_clear_frames": corridor_clear_frames,
            }

        # --- Assisted / generic door hops ---
        remaining = level2_door_hops_from(snap.screen)
        nav2 = OverworldToLevel2Controller(
            hops=remaining,
            require_level2_screen=stop_screen == 0x3C and not enter_dungeon,
            require_dungeon=enter_dungeon,
        )
        if not remaining and snap.screen == 0x3C:
            nav2 = OverworldToLevel2Controller(
                hops=(),
                require_level2_screen=not enter_dungeon,
                require_dungeon=enter_dungeon,
            )

        def _stop(snap) -> bool:
            if enter_dungeon and snap.level == 2:
                nav2.success = True
                return True
            if (
                not enter_dungeon
                and snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == stop_screen
                and snap.filled_hearts > 0
            ):
                nav2.success = True
                return True
            return False

        obs, fr = _run_nav_loop(
            env,
            obs,
            nav2,
            max_frames=max_frames,
            assist=assist,
            trail=trail,
            tag=tag,
            stop_pred=_stop,
        )
        total_frames += fr

        snap = read_snapshot(env.get_ram())
        png = RECORDINGS_DIR / f"{tag}_final.png"
        save_rgb_png(obs, png)
        ok = (
            (enter_dungeon and snap.level == 2)
            or (
                not enter_dungeon
                and snap.level == 0
                and snap.screen == stop_screen
                and snap.mode == PLAY_MODE
                and snap.filled_hearts > 0
            )
        )
        checkpoint = None
        if ok and save_checkpoint:
            name = "Level2Entrance" if snap.level == 2 else f"OW_{snap.screen:02X}"
            checkpoint = str(save_state(env, GAME_DIR, GAME, name))

        return {
            "ok": ok,
            "track": track,
            "infinite_life": infinite_life,
            "entry": entry,
            "prefix": prefix_report,
            "farm": None,
            "corridor_clear": None,
            "trail": trail,
            "nav": nav2.report(),
            "remaining_hops": [
                {"t": f"0x{h.target:02x}", "d": h.direction} for h in remaining
            ],
            "assist": assist.report() if assist else None,
            "final": _snapshot_dict(snap),
            "screenshot": str(png),
            "checkpoint": checkpoint,
            "frames": total_frames,
            "farm_hearts_min": farm_hearts_min,
            "corridor_clear_frames": corridor_clear_frames,
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level1ExitOverworld")
    p.add_argument("--max-frames", type=int, default=20000)
    p.add_argument("--stop-screen", type=lambda s: int(s, 0), default=0x3C)
    p.add_argument("--enter-dungeon", action="store_true")
    p.add_argument(
        "--farm-hearts",
        type=int,
        default=3,
        help="Min filled hearts before leaving 0x4A (0=skip). Clean default 3.",
    )
    p.add_argument("--farm-max-frames", type=int, default=3600)
    p.add_argument(
        "--corridor-clear",
        type=int,
        default=201,
        help="Frames of kite/clear on 0x5A before east (0=skip). Clean default 201.",
    )
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--tag", default="l2_suffix")
    p.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (ASSIST_CONTRACT). Not Clean.",
    )
    args = p.parse_args(argv)

    rep = run_probe(
        start_state=args.from_state,
        max_frames=args.max_frames,
        stop_screen=args.stop_screen,
        enter_dungeon=args.enter_dungeon,
        farm_hearts_min=args.farm_hearts,
        farm_max_frames=args.farm_max_frames,
        corridor_clear_frames=args.corridor_clear,
        save_checkpoint=args.save_state,
        tag=args.tag,
        infinite_life=args.infinite_life,
    )
    out_name = f"{args.tag}_probe.json"
    if rep.get("track") == "clean" and "clean" not in args.tag.lower():
        out_name = f"{args.tag}_clean_probe.json"
    out = RECORDINGS_DIR / out_name
    write_json_report(out, rep)
    fin = rep["final"]
    assist_s = ""
    if rep.get("assist"):
        a = rep["assist"]
        h = a.get("health") or {}
        assist_s = (
            f" assist_writes={h.get('writes', 0)}"
            f" dmg={a.get('total_damage', 0)}"
            f" dmg_ev={a.get('damage_events', 0)}"
        )
        by_loc = a.get("damage_by_location") or {}
        if by_loc:
            top = list(by_loc.items())[:3]
            hot = ",".join(f"{k}:{v}" for k, v in top)
            assist_s += f" hot=[{hot}]"
    farm_s = ""
    if rep.get("farm"):
        farm_s = (
            f" farm_ok={rep['farm'].get('success')} "
            f"peak={rep['farm'].get('peak_filled')} "
            f"farm_f={rep['farm'].get('frames')}"
        )
    clear_s = ""
    if rep.get("corridor_clear"):
        clear_s = (
            f" clear={rep['corridor_clear'].get('start_filled')}→"
            f"{rep['corridor_clear'].get('end_filled')}"
        )
    print(
        f"ok={rep['ok']} track={rep.get('track')} sc={fin['screen']:#04x} "
        f"lvl={fin['level']} mode={fin['mode']} hp={fin['health']:#04x} "
        f"hearts={fin['hearts']} xy=({fin['x']},{fin['y']}) "
        f"frames={rep.get('frames')} trail={len(rep.get('trail', []))} "
        f"{farm_s}{clear_s}{assist_s}"
    )
    for t in rep.get("trail", [])[-14:]:
        sc = t.get("screen", 0)
        hp = t.get("health", 0)
        hearts = t.get("hearts", t.get("filled_hearts", "?"))
        print(
            f"  stage={t.get('stage', t.get('f', ''))} "
            f"sc={sc:#04x} hp={hp:#04x} hearts={hearts} "
            f"xy=({t.get('x')},{t.get('y')})"
        )
    if rep.get("nav"):
        print(f"nav_notes={(rep['nav'].get('notes') or [])[-16:]}")
    if rep.get("notes"):
        print(f"notes={rep['notes']}")
    print(f"report={out}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
