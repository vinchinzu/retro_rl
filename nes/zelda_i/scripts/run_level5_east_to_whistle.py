"""Survival: Level5EastKey 0x77 → natural Recorder → Whistle basement 0x04.

0x77 LEFT → 0x76 UP → 0x66, bomb-west → 0x65, then the proven 0x65 bomb-west
→ 0x64 stairs → 0x07 → 0x06 key-west → 0x05 block-stairs → 0x04 Recorder.

--infinite-life only. No door/key/inventory pokes. Not Clean STATUS.

    PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/run_level5_east_to_whistle.py \
        --from-state Level5EastKey --infinite-life --save-state
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import (
    BLUE_DARKNUT_TYPE,
    ROOM_L5_BLUE_64,
    ROOM_L5_PASSAGE_06,
    ROOM_L5_WHISTLE_05,
    ROOM_L5_WHISTLE_ITEM,
    bomb_west_from_65,
    bomb_west_from_66,
    cellar_other_mouth,
    fight_blue_darknuts,
    hunt_whistle,
    key_west_to,
    make_return_66_controller,
    push_block_stairs,
    take_center_stairs_64,
    take_whistle_04,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_MAP,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)


def _inv(ram) -> dict:
    snap = read_snapshot(ram)
    return {
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "candle_0x065B": int(read_u8(ram, ADDR_CANDLE)),
        "map_0x0668": int(read_u8(ram, ADDR_MAP)),
        "triforce_0x0671": int(read_u8(ram, ADDR_TRIFORCE)),
        "tf_l5_bit": bool(int(read_u8(ram, ADDR_TRIFORCE)) & 0x10),
        "bombs": int(snap.bombs),
        "keys": int(snap.keys),
        "room": snap.screen,
        "room_hex": f"0x{snap.screen:02x}",
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "doors": int(snap.cur_opened_doors),
        "health": int(snap.health),
    }


def _shot(env, tag: str, name: str, obs=None):
    path = RECORDINGS_DIR / f"{tag}_{name}.png"
    if obs is None:
        obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, path)
    return str(path)


def _save_ckpt(env, name: str, source: str, via: str, extra: dict) -> str:
    path = write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
    write_state_provenance(
        path,
        source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{source}.state",
        request={
            "segment": name,
            "predecessor_entry": True,
            "start_state": source,
            "via": via,
            "key_poke": False,
            "door_poke": False,
            "bomb_count_poke": False,
            "selected_item_poke": False,
            "route_eligible": False,
        },
        selected_trial={"success": True, **extra},
        natural_entry=False,
    )
    return name


def run_once(
    *,
    start_state: str,
    infinite_life: bool,
    save_checkpoint: bool,
    tag: str,
    to_65_only: bool,
) -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [1]
    hops: list[dict] = []
    checkpoints: list[str] = []
    shots: list[str] = []
    trail: list[dict] = []
    failed = None
    reason = None
    whistle_before = 0
    whistle_after = 0
    try:
        reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        whistle_before = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        start = {**_inv(env.get_ram()), "state": start_state}
        print("START", start, flush=True)
        shots.append(_shot(env, tag, "start", obs))

        ctl = make_return_66_controller()
        last_room = snap.screen
        stuck_at = (snap.link_x, snap.link_y)
        stuck_n = 0
        for frame in range(ctl.max_frames):
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            obs, *_ = env.step(action.action)
            total[0] += 1
            after = read_snapshot(env.get_ram())
            pos = (after.link_x, after.link_y)
            if after.screen != last_room:
                rec = {
                    "event": "transition",
                    "frame": total[0],
                    "room": after.screen,
                    "room_hex": f"0x{after.screen:02x}",
                    "mode": after.mode,
                    "xy": [after.link_x, after.link_y],
                    "keys": after.keys,
                    "reason": action.reason,
                }
                trail.append(rec)
                shots.append(_shot(env, tag, f"r{after.screen:02x}_f{total[0]}", obs))
                print("ROOM", rec, flush=True)
                last_room = after.screen
                stuck_n = 0
                stuck_at = pos
            else:
                if pos == stuck_at:
                    stuck_n += 1
                    if stuck_n % 250 == 0:
                        trail.append(
                            {
                                "event": "stuck",
                                "frame": total[0],
                                "room": after.screen,
                                "xy": [after.link_x, after.link_y],
                                "reason": action.reason,
                                "stuck": stuck_n,
                            }
                        )
                        shots.append(_shot(env, tag, f"stuck_f{total[0]}", obs))
                else:
                    stuck_n = 0
                    stuck_at = pos
            if ctl.success or ctl.failed:
                break
        hops.append({"hop": "return_66", **ctl.report()})
        snap = read_snapshot(env.get_ram())
        print(
            "AT66",
            ctl.success,
            f"0x{snap.screen:02x}",
            [snap.link_x, snap.link_y],
            "k",
            snap.keys,
            "b",
            snap.bombs,
            flush=True,
        )
        shots.append(_shot(env, tag, "at66"))
        if not ctl.success or snap.screen != 0x66:
            failed, reason = f"0x{snap.screen:02x}", "return_66_missed"
            return _finish(
                env, assist, tag, hops, checkpoints, shots, trail, start,
                whistle_before, whistle_after, failed, reason, start_state,
            )

        bomb66 = bomb_west_from_66(env, assist, total)
        hops.append({"hop": "bomb_west_from_66", **{k: bomb66[k] for k in bomb66 if k != "menu"}})
        snap = read_snapshot(env.get_ram())
        print(
            "BOMB66",
            bomb66.get("success"),
            "via",
            bomb66.get("via"),
            "dest",
            f"0x{bomb66.get('dest'):02x}",
            "xy",
            bomb66.get("xy"),
            "spent",
            bomb66.get("bombs_spent"),
            flush=True,
        )
        shots.append(_shot(env, tag, "after_bomb66"))
        if not bomb66.get("success"):
            failed, reason = f"0x{snap.screen:02x}", "bomb_west_66_did_not_enter_65"
            return _finish(
                env, assist, tag, hops, checkpoints, shots, trail, start,
                whistle_before, whistle_after, failed, reason, start_state,
            )
        if save_checkpoint:
            checkpoints.append(
                _save_ckpt(
                    env,
                    "Level5Entered65From77",
                    start_state,
                    "0x77 return 0x66 bomb-west",
                    {**_inv(env.get_ram()), "via": bomb66.get("via")},
                )
            )
        if to_65_only:
            whistle_after = int(read_u8(env.get_ram(), ADDR_WHISTLE))
            return _finish(
                env, assist, tag, hops, checkpoints, shots, trail, start,
                whistle_before, whistle_after, failed, reason, start_state,
                extra={"to_65_only": True, "prefix_ok": True},
            )

        # Proven 0x65 → 0x04 suffix (same hops as run_level5_whistle).
        bomb65 = bomb_west_from_65(env, assist, total)
        hops.append({"hop": "bomb_west_from_65", **{k: bomb65[k] for k in bomb65 if k != "menu"}})
        print("BOMB65", bomb65.get("success"), "dest", f"0x{bomb65.get('dest'):02x}", flush=True)
        shots.append(_shot(env, tag, "after_bomb65"))
        if not bomb65.get("success"):
            failed, reason = "0x65", "bomb_west_did_not_enter_64"
            return _finish(
                env, assist, tag, hops, checkpoints, shots, trail, start,
                whistle_before, whistle_after, failed, reason, start_state,
            )

        snap = read_snapshot(env.get_ram())
        n_dn = sum(
            1
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id == BLUE_DARKNUT_TYPE and o.hp > 0
        )
        if n_dn:
            fight64 = fight_blue_darknuts(
                env, assist, total, ROOM_L5_BLUE_64, expected=n_dn, source=0x65
            )
            hops.append({"hop": "fight_64", **{k: fight64[k] for k in fight64 if k != "progress"}})
            print("FIGHT64", fight64.get("ok"), "end_n", fight64.get("end_n"), flush=True)
            idle(env, assist, total, 16)
            pushed64 = push_block_stairs(env, assist, total, ROOM_L5_BLUE_64)
            hops.append({"hop": "push_64", **{k: pushed64[k] for k in pushed64 if k != "log"}})
            if pushed64.get("success"):
                stairs = {
                    "success": True,
                    "dest": read_snapshot(env.get_ram()).screen,
                    "mode": read_snapshot(env.get_ram()).mode,
                    "via": "push_block_stairs",
                }
            else:
                stairs = take_center_stairs_64(env, assist, total)
        else:
            stairs = take_center_stairs_64(env, assist, total)
        hops.append({"hop": "take_center_stairs_64", **{k: stairs[k] for k in stairs if k != "log"}})
        print("STAIRS64", stairs.get("success"), "dest", f"0x{stairs.get('dest', 0):02x}", flush=True)
        shots.append(_shot(env, tag, "after_stairs64"))
        if not stairs.get("success"):
            failed, reason = "0x64", "center_stairs_not_taken"
            return _finish(
                env, assist, tag, hops, checkpoints, shots, trail, start,
                whistle_before, whistle_after, failed, reason, start_state,
            )

        cellar = cellar_other_mouth(env, assist, total)
        hops.append({"hop": "cellar_other_mouth", **{k: cellar[k] for k in cellar if k not in ("start",)}})
        print("CELLAR", cellar.get("success"), "dest", f"0x{cellar.get('dest', 0):02x}", flush=True)
        shots.append(_shot(env, tag, "after_cellar"))
        if not cellar.get("success"):
            failed, reason = "0x07", "other_mouth_did_not_enter_0x06"
            return _finish(
                env, assist, tag, hops, checkpoints, shots, trail, start,
                whistle_before, whistle_after, failed, reason, start_state,
            )

        west = key_west_to(env, assist, total, ROOM_L5_WHISTLE_05)
        hops.append({"hop": "key_west_to_05", **west})
        print("KEYWEST", west.get("success"), "dest", f"0x{west.get('dest', 0):02x}", flush=True)
        shots.append(_shot(env, tag, "after_keywest"))
        if not west.get("success"):
            failed, reason = "0x06", "key_west_did_not_enter_0x05"
            return _finish(
                env, assist, tag, hops, checkpoints, shots, trail, start,
                whistle_before, whistle_after, failed, reason, start_state,
            )

        snap = read_snapshot(env.get_ram())
        n_dn = sum(
            1
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id == BLUE_DARKNUT_TYPE and o.hp > 0
        )
        fight = fight_blue_darknuts(
            env, assist, total, ROOM_L5_WHISTLE_05, expected=max(6, n_dn), source=ROOM_L5_PASSAGE_06
        )
        hops.append({"hop": "fight_05", **{k: fight[k] for k in fight if k != "progress"}})
        print("FIGHT05", fight.get("ok"), "end_n", fight.get("end_n"), flush=True)
        shots.append(_shot(env, tag, "after_fight05"))
        if not fight.get("ok"):
            failed, reason = "0x05", "darknuts_not_cleared"
            return _finish(
                env, assist, tag, hops, checkpoints, shots, trail, start,
                whistle_before, whistle_after, failed, reason, start_state,
            )

        pushed = push_block_stairs(env, assist, total, ROOM_L5_WHISTLE_05)
        hops.append({"hop": "push_block_stairs_05", **{k: pushed[k] for k in pushed if k != "log"}})
        print("PUSH05", pushed.get("success"), "dest", pushed.get("dest"), flush=True)
        shots.append(_shot(env, tag, "after_push05"))
        if not pushed.get("success"):
            failed, reason = "0x05", "block_stairs_not_taken"
            return _finish(
                env, assist, tag, hops, checkpoints, shots, trail, start,
                whistle_before, whistle_after, failed, reason, start_state,
            )

        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_L5_WHISTLE_ITEM or snap.mode in (9, 11):
            walk = take_whistle_04(env, assist, total)
        else:
            walk = hunt_whistle(env, assist, total)
        whistle_after = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        hops.append({"hop": "take_whistle", "in": walk["in"], "out": walk["out"], "got": walk["got"]})
        print("WHISTLE", walk["in"], "->", walk["out"], "now", whistle_after, flush=True)
        shots.append(_shot(env, tag, "after_whistle"))
        if whistle_after < 1:
            failed, reason = "0x04", "whistle_still_0"
            return _finish(
                env, assist, tag, hops, checkpoints, shots, trail, start,
                whistle_before, whistle_after, failed, reason, start_state,
            )
        if save_checkpoint:
            snap = read_snapshot(env.get_ram())
            checkpoints.append(
                _save_ckpt(
                    env,
                    "Level5WhistleFrom77",
                    start_state,
                    "0x77 return 0x66 bomb-west 0x65 bomb-west 0x64 stairs 0x07 0x06 0x05 0x04",
                    {
                        **_inv(env.get_ram()),
                        "room": int(snap.screen),
                        "whistle_0x065C": whistle_after,
                    },
                )
            )
        return _finish(
            env, assist, tag, hops, checkpoints, shots, trail, start,
            whistle_before, whistle_after, failed, reason, start_state,
        )
    finally:
        env.close()


def _finish(
    env, assist, tag, hops, checkpoints, shots, trail, start,
    w0, w1, failed, reason, source, extra=None,
):
    snap = read_snapshot(env.get_ram())
    png = RECORDINGS_DIR / f"{tag}_final.png"
    obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, png)
    shots.append(str(png))
    assist_rep = assist.report() if assist is not None else None
    extra = extra or {}
    prefix_ok = bool(extra.get("prefix_ok"))
    body = {
        "ok": (w1 >= 1 and failed is None) or (prefix_ok and failed is None),
        "status_claim": None,
        "route_eligible": False,
        "pokes": False,
        "track": "assisted",
        "intervention_class": "survival",
        "start_state": source,
        "start": start,
        "hops": hops,
        "trail": trail,
        "checkpoints": checkpoints,
        "screenshots": shots,
        "whistle_before_0x065C": w0,
        "whistle_after_0x065C": w1,
        "failed_room": failed,
        "reason": reason,
        "final": _inv(env.get_ram()),
        "screenshot": str(png.resolve()),
        "assist": assist_rep,
        "progression_writes": 0 if assist_rep is None else assist_rep.get("progression_writes", 0),
        "capacity_writes": 0 if assist_rep is None else assist_rep.get("capacity_writes", 0),
        "deaths": 0 if assist_rep is None else assist_rep.get("deaths", 0),
        "commands": [
            "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/run_level5_east_to_whistle.py "
            f"--from-state {source} --infinite-life --save-state --tag {tag}"
        ],
    }
    body.update(extra)
    # snap used so unused-var lints stay quiet if env is mid-close
    body["final"]["room_confirm"] = snap.screen
    write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-state", default="Level5EastKey")
    ap.add_argument("--infinite-life", action="store_true")
    ap.add_argument("--save-state", action="store_true")
    ap.add_argument("--tag", default="l5_east_to_whistle")
    ap.add_argument(
        "--to-65-only",
        action="store_true",
        help="Stop after 0x66 bomb-west lands 0x65 (prefix debug).",
    )
    args = ap.parse_args()
    r = run_once(
        start_state=args.from_state,
        infinite_life=args.infinite_life,
        save_checkpoint=args.save_state,
        tag=args.tag,
        to_65_only=args.to_65_only,
    )
    print("OK", r.get("ok"))
    print("WHISTLE", r.get("whistle_before_0x065C"), "->", r.get("whistle_after_0x065C"))
    print("FAILED", r.get("failed_room"), r.get("reason"))
    print("FINAL", r.get("final"))
    print("CKPT", r.get("checkpoints"))
    print("ASSIST", {k: r.get("assist", {} or {}).get(k) for k in (
        "progression_writes", "capacity_writes", "deaths", "total_damage"
    )} if r.get("assist") else None)
    return 0 if r.get("ok") or (args.to_65_only and r.get("reason") is None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
