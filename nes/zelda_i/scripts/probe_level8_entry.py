"""Assisted recon: overworld walk to Level 8 (Lion) candle-bush + entry.

Does **not** poke inventory (ASSIST_CONTRACT). Candle must be bought in-game
or the probe stops on the bush screen and records the blocker.

Examples::

    # Walk hop table from PostSwordStart (Survival assist)
    uv run python zelda_i/scripts/probe_level8_entry.py --infinite-life

    # Try both hop tables; save bush OW checkpoint if reached
    uv run python zelda_i/scripts/probe_level8_entry.py --infinite-life \\
        --save-state --tag l8_recon

    # From a mid-path OW state
    uv run python zelda_i/scripts/probe_level8_entry.py --from-state OW_5B \\
        --infinite-life --path via_6b_east

    # Blue Candle shop path (0x5E cave) — rr-ccx
    uv run python zelda_i/scripts/probe_level8_entry.py --to-shop \\
        --infinite-life --save-state --tag l8_shop
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.level8_overworld import (
    CANDLE_SHOP_HOPS,
    CANDLE_SHOP_STATE,
    DEFAULT_BUSH_X,
    DEFAULT_BUSH_Y,
    LEVEL8_BUSH_HOPS,
    LEVEL8_BUSH_HOPS_VIA_58,
    LEVEL_8,
    SCREEN_CANDLE_SHOP,
    SCREEN_LEVEL8_BUSH,
    SEGMENT_MAX_FRAMES,
    OverworldToCandleShopController,
    OverworldToLevel8Controller,
    candle_shop_cave_entered,
    has_candle,
    level8_entered,
    poke_candle_for_recon,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

# Door bits (same layout as L2 probe)
DOOR_RIGHT = 0x01
DOOR_LEFT = 0x02
DOOR_DOWN = 0x04
DOOR_UP = 0x08


def _objs(snap: ZeldaSnapshot) -> list[dict]:
    out: list[dict] = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10):
            continue
        if o.type_id in (0, 0xFF):
            continue
        out.append(
            {
                "slot": o.slot,
                "type": o.type_id,
                "type_name": object_name(o.type_id),
                "x": o.x,
                "y": o.y,
                "hp": o.hp,
            }
        )
    return out


def _snap_dict(snap: ZeldaSnapshot, ram=None) -> dict:
    candle = read_u8(ram, ADDR_CANDLE) if ram is not None else None
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "screen_hex": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "facing": snap.facing,
        "health": snap.health,
        "rupees": snap.rupees,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "sword": snap.sword,
        "candle": candle,
        "triforce": snap.triforce,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "room_obj_count": snap.room_obj_count,
        "cur_opened_doors": snap.cur_opened_doors,
        "objects": _objs(snap),
        "type_counts": dict(Counter(o["type"] for o in _objs(snap))),
    }


def _select_path(name: str):
    # Both names resolve to the live-verified 0x5C-maze path (0x6D).
    if name in ("via_6b_east", "via_58", "default"):
        return LEVEL8_BUSH_HOPS if name != "via_58" else LEVEL8_BUSH_HOPS_VIA_58
    return LEVEL8_BUSH_HOPS


def _probe_cardinals(env, obs, *, frames_each: int, assist, tag: str) -> dict:
    """Push N/E/S/W briefly from current dungeon room; log screen changes."""
    results: dict[str, list] = {}
    for direction in ("UP", "RIGHT", "DOWN", "LEFT"):
        trail: list[dict] = []
        start = read_snapshot(env.get_ram())
        start_screen = start.screen
        for f in range(frames_each):
            snap = read_snapshot(env.get_ram())
            if snap.screen != start_screen or snap.level != start.level:
                trail.append({"f": f, **_snap_dict(snap, env.get_ram())})
                save_rgb_png(
                    obs,
                    RECORDINGS_DIR
                    / f"{tag}_dir_{direction.lower()}_sc{snap.screen:02x}.png",
                )
                break
            act = nes_action(direction)
            if f % 10 < 3:
                act = nes_action(direction, "A")
            obs, *_ = env.step(act)
            if assist is not None:
                assist.apply_env(env, frame=f)
        # Walk back toward start roughly
        back = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[
            direction
        ]
        for f in range(frames_each // 2):
            obs, *_ = env.step(nes_action(back))
            if assist is not None:
                assist.apply_env(env, frame=f)
            snap = read_snapshot(env.get_ram())
            if snap.screen == start_screen and snap.level == start.level:
                break
        results[direction] = trail
    return results


def run_shop_probe(
    *,
    start_state: str,
    max_frames: int,
    infinite_life: bool,
    enter_cave: bool,
    buy_candle: bool,
    save_checkpoint: bool,
    tag: str,
) -> dict:
    """Assisted walk to Blue Candle shop 0x5E (+ optional cave / buy)."""
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    hops = CANDLE_SHOP_HOPS

    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

        entry = _snap_dict(read_snapshot(env.get_ram()), env.get_ram())
        trail: list[dict] = [{"f": 0, **entry}]
        last_screen = entry["screen"]
        last_level = entry["level"]
        last_mode = entry["mode"]

        # Mid-path / cave starts: skip hops already completed.
        if start_state in (CANDLE_SHOP_STATE, "BFS_5E", "OW_5E") or entry.get(
            "screen"
        ) == SCREEN_CANDLE_SHOP:
            use_hops: tuple = ()
        else:
            use_hops = hops

        nav = OverworldToCandleShopController(
            hops=use_hops if use_hops else hops,
            enter_cave=enter_cave,
            buy_candle=buy_candle,
        )
        if not use_hops:
            nav.hop_index = len(nav.hops)
            if entry.get("mode") == 11:
                from zelda_i.level8_overworld import CandleShopNavPhase

                if buy_candle:
                    nav.phase = CandleShopNavPhase.BUY
                    nav._rupees_at_buy = entry.get("rupees", 0)
                else:
                    nav.phase = CandleShopNavPhase.DONE
                    nav.success = True
            else:
                from zelda_i.level8_overworld import CandleShopNavPhase

                nav.phase = CandleShopNavPhase.DOOR

        frames = 0
        while frames < max_frames:
            snap = read_snapshot(env.get_ram())
            nav.candle_value = read_u8(env.get_ram(), ADDR_CANDLE)
            if (
                snap.screen != last_screen
                or snap.level != last_level
                or snap.mode != last_mode
            ):
                trail.append({"f": frames, **_snap_dict(snap, env.get_ram())})
                last_screen = snap.screen
                last_level = snap.level
                last_mode = snap.mode
                save_rgb_png(
                    obs,
                    RECORDINGS_DIR
                    / f"{tag}_sc{snap.screen:02x}_m{snap.mode}_lv{snap.level}.png",
                )
            if snap.mode == 17:
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

        final = _snap_dict(read_snapshot(env.get_ram()), env.get_ram())
        ow_shop = (
            final["level"] == 0
            and final["mode"] == PLAY_MODE
            and final["screen"] == SCREEN_CANDLE_SHOP
        )
        in_cave = candle_shop_cave_entered(env.get_ram())
        candle_final = has_candle(env.get_ram())

        saved = []
        if save_checkpoint and in_cave:
            path = save_state(env, GAME_DIR, GAME, CANDLE_SHOP_STATE)
            saved.append(str(path))
        if save_checkpoint and ow_shop and not in_cave:
            path = save_state(env, GAME_DIR, GAME, "OW_5E")
            saved.append(str(path))
        if save_checkpoint and candle_final:
            path = save_state(env, GAME_DIR, GAME, "CandleOwned")
            saved.append(str(path))

        report = {
            "tag": tag,
            "track": track,
            "mode": "candle_shop",
            "assist_contract": "nes/zelda_i/docs/ASSIST_CONTRACT.md",
            "start_state": start_state,
            "path": "candle_shop_5e",
            "hops": [
                {
                    "target": h.target,
                    "direction": h.direction,
                    "align_x": h.align_x,
                    "align_y": h.align_y,
                    "y_band": h.y_band,
                }
                for h in hops
            ],
            "entry": entry,
            "final": final,
            "trail": trail,
            "nav": nav.report(),
            "frames": frames,
            "shop_screen": SCREEN_CANDLE_SHOP,
            "shop_ow_reached": ow_shop or in_cave,
            "shop_cave_entered": in_cave,
            "candle_final": candle_final,
            "rupees_final": final.get("rupees"),
            "buy_candle_requested": buy_candle,
            "need_rupees": in_cave and not candle_final and (final.get("rupees") or 0) < 60,
            "saved_states": saved,
            "assist": assist.report() if assist is not None else None,
            "success": ow_shop or in_cave or candle_final,
        }
        out = RECORDINGS_DIR / f"{tag}.json"
        write_json_report(out, report)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_final.png")
        print(
            f"[{tag}] shop track={track} ow={ow_shop} cave={in_cave} "
            f"candle={candle_final} R={final.get('rupees')} frames={frames} "
            f"final=0x{final['screen']:02x} m={final['mode']} saved={saved}"
        )
        return report
    finally:
        env.close()


def run_probe(
    *,
    start_state: str,
    path_name: str,
    max_frames: int,
    infinite_life: bool,
    burn_bush: bool,
    enter_dungeon: bool,
    save_checkpoint: bool,
    probe_rooms: bool,
    bush_x: int,
    bush_y: int,
    poke_candle: bool,
    candle_value: int,
    tag: str,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    if poke_candle:
        track = f"{track}+candle_poke_recon"
    hops = _select_path(path_name)
    poke_notes: list[str] = []

    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if poke_candle:
            # RECON only — not Clean / not ASSIST_CONTRACT progression.
            poke_notes = poke_candle_for_recon(env, candle=candle_value)
        if assist is not None:
            assist.apply_env(env, frame=0)

        entry = _snap_dict(read_snapshot(env.get_ram()), env.get_ram())
        candle_at_start = has_candle(env.get_ram())
        trail: list[dict] = [{"f": 0, **entry}]
        last_screen = entry["screen"]
        last_level = entry["level"]

        # If no candle, force stop on bush screen (still walk OW).
        do_burn = burn_bush and candle_at_start
        do_enter = enter_dungeon and candle_at_start
        if burn_bush and not candle_at_start:
            # Still allow enter flag only if somehow mouth already open.
            do_burn = False

        # From bush-only start states, hops already done — empty hop table.
        if start_state in ("Level8BushOW", "OW_6D", "BFS_6D") and hops:
            # Keep hops for report but start in post-hop burn if requested.
            use_hops = hops if entry.get("screen") != hops[-1].target else ()
        else:
            use_hops = hops

        nav = OverworldToLevel8Controller(
            hops=use_hops if use_hops is not None else hops,
            burn_bush=do_burn,
            enter_dungeon=do_enter,
            bush_x=bush_x,
            bush_y=bush_y,
        )
        if not use_hops and do_burn:
            from zelda_i.level8_overworld import Level8NavPhase

            nav.hop_index = len(nav.hops)
            nav.phase = Level8NavPhase.BURN

        frames = 0
        while frames < max_frames:
            snap = read_snapshot(env.get_ram())
            if snap.screen != last_screen or snap.level != last_level:
                trail.append({"f": frames, **_snap_dict(snap, env.get_ram())})
                last_screen = snap.screen
                last_level = snap.level
                save_rgb_png(
                    obs,
                    RECORDINGS_DIR
                    / f"{tag}_sc{snap.screen:02x}_lv{snap.level}.png",
                )
            if snap.mode == 17:
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

        final = _snap_dict(read_snapshot(env.get_ram()), env.get_ram())
        bush_reached = (
            final["level"] == 0
            and final["mode"] == PLAY_MODE
            and final["screen"] == hops[-1].target
        )
        entered = level8_entered(env.get_ram())

        saved = []
        if save_checkpoint and bush_reached and not entered:
            name = "Level8BushOW"
            path = save_state(env, GAME_DIR, GAME, name)
            saved.append(str(path))
        if save_checkpoint and entered:
            name = "Level8Entrance"
            path = save_state(env, GAME_DIR, GAME, name)
            saved.append(str(path))

        room_probe = None
        if probe_rooms and entered:
            room_probe = _probe_cardinals(
                env, obs, frames_each=280, assist=assist, tag=tag
            )
            final = _snap_dict(read_snapshot(env.get_ram()), env.get_ram())

        report = {
            "tag": tag,
            "track": track,
            "assist_contract": "nes/zelda_i/docs/ASSIST_CONTRACT.md",
            "start_state": start_state,
            "path": path_name,
            "hops": [
                {
                    "target": h.target,
                    "direction": h.direction,
                    "align_x": h.align_x,
                    "align_y": h.align_y,
                    "y_band": h.y_band,
                }
                for h in hops
            ],
            "entry": entry,
            "final": final,
            "trail": trail,
            "nav": nav.report(),
            "frames": frames,
            "candle_at_start": candle_at_start,
            "candle_final": has_candle(env.get_ram()),
            "poke_candle": poke_candle,
            "poke_notes": poke_notes,
            "bush_screen_planned": hops[-1].target if hops else SCREEN_LEVEL8_BUSH,
            "bush_xy": [bush_x, bush_y],
            "bush_reached": bush_reached,
            "level8_entered": entered,
            "candle_blocker": not candle_at_start and not entered,
            "saved_states": saved,
            "room_probe": room_probe,
            "assist": assist.report() if assist is not None else None,
            "success": bush_reached or entered,
        }
        out = RECORDINGS_DIR / f"{tag}.json"
        write_json_report(out, report)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_final.png")
        print(
            f"[{tag}] track={track} bush={bush_reached} entered={entered} "
            f"candle={candle_at_start} frames={frames} "
            f"final=0x{final['screen']:02x} lv={final['level']} "
            f"saved={saved}"
        )
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="PostSwordStart")
    p.add_argument(
        "--path",
        choices=("default", "via_6b_east", "via_58"),
        default="default",
        help="Hop table variant (all map to live 0x5C-maze → 0x6D path)",
    )
    p.add_argument(
        "--to-shop",
        action="store_true",
        help="Walk CANDLE_SHOP_HOPS to Blue Candle cave 0x5E (rr-ccx)",
    )
    p.add_argument(
        "--enter-cave",
        action="store_true",
        default=None,
        help="With --to-shop: enter cave mouth (default on). Use --no-enter-cave to stop on OW.",
    )
    p.add_argument(
        "--no-enter-cave",
        action="store_true",
        help="With --to-shop: stop on OW 0x5E without cave enter",
    )
    p.add_argument(
        "--buy-candle",
        action="store_true",
        help="With --to-shop: attempt natural buy if rupees>=60 (no inventory poke)",
    )
    p.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    p.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (not Clean STATUS)",
    )
    p.add_argument(
        "--burn",
        action="store_true",
        help="Attempt candle burn if candle already in inventory",
    )
    p.add_argument(
        "--enter-dungeon",
        action="store_true",
        help="After burn, push into Level 8 mouth",
    )
    p.add_argument(
        "--save-state",
        action="store_true",
        help="Write Level8BushOW / Level8Entrance / CandleShop5E under custom_integrations",
    )
    p.add_argument(
        "--probe-rooms",
        action="store_true",
        help="If entered, push cardinals ~280f each and log rooms",
    )
    p.add_argument("--bush-x", type=int, default=DEFAULT_BUSH_X)
    p.add_argument("--bush-y", type=int, default=DEFAULT_BUSH_Y)
    p.add_argument(
        "--poke-candle",
        action="store_true",
        help="RECON: poke ADDR_CANDLE + B-select (not Clean / not contract)",
    )
    p.add_argument(
        "--candle-value",
        type=int,
        default=1,
        help="With --poke-candle: 1=blue, 2=red (multi-use per screen)",
    )
    p.add_argument("--tag", default="l8_recon")
    args = p.parse_args(argv)

    if args.to_shop:
        enter_cave = not args.no_enter_cave
        report = run_shop_probe(
            start_state=args.from_state,
            max_frames=args.max_frames,
            infinite_life=args.infinite_life,
            enter_cave=enter_cave,
            buy_candle=args.buy_candle,
            save_checkpoint=args.save_state,
            tag=args.tag if args.tag != "l8_recon" else "l8_shop",
        )
        return 0 if report.get("success") else 1

    report = run_probe(
        start_state=args.from_state,
        path_name=args.path,
        max_frames=args.max_frames,
        infinite_life=args.infinite_life,
        burn_bush=args.burn or args.enter_dungeon or args.poke_candle,
        enter_dungeon=args.enter_dungeon,
        save_checkpoint=args.save_state,
        probe_rooms=args.probe_rooms,
        bush_x=args.bush_x,
        bush_y=args.bush_y,
        poke_candle=args.poke_candle,
        candle_value=args.candle_value,
        tag=args.tag,
    )
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
