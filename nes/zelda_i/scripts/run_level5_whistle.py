"""Assisted L5 Whistle path: Cleared65 bomb-west → 0x64 stairs → 0x04 Recorder.

Survival / --infinite-life. No key/door pokes. Not a Clean STATUS claim.

    PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/run_level5_whistle.py \
        --from-state Level5Cleared65 --infinite-life --save-state
    PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/run_level5_whistle.py \
        --from-state Level5Entered64 --infinite-life --save-state
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
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.level5_dungeon import LEVEL_5
from zelda_i.level5_path import (
    BLUE_DARKNUT_TYPE,
    ROOM_L5_BLUE_64,
    ROOM_L5_CELLAR_07,
    ROOM_L5_PASSAGE_06,
    ROOM_L5_WHISTLE_05,
    ROOM_L5_WHISTLE_ITEM,
    bomb_west_from_65,
    cellar_other_mouth,
    exit_whistle_04,
    fight_blue_darknuts,
    hunt_whistle,
    take_whistle_04,
    key_west_to,
    push_block_stairs,
    select_b_item_menu,
    take_center_stairs_64,
    walk_west_from_25,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
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
        "triforce_0x0671": int(read_u8(ram, ADDR_TRIFORCE)),
        "tf_l5_bit": bool(int(read_u8(ram, ADDR_TRIFORCE)) & 0x10),
        "bombs": int(snap.bombs),
        "keys": int(snap.keys),
        "room": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
    }


def _objs(snap) -> list[dict]:
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 12) or o.type_id in (0, 0xFF):
            continue
        out.append(
            {
                "slot": o.slot,
                "type": o.type_id,
                "hp": o.hp,
                "x": o.x,
                "y": o.y,
            }
        )
    return out


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
    to_stairs_only: bool,
    do_boss: bool,
) -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [1]
    hops = []
    checkpoints = []
    whistle_before = 0
    whistle_after = 0
    failed = None
    reason = None
    boss = None
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        whistle_before = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        start = {
            **_inv(env.get_ram()),
            "objects": _objs(snap),
            "state": start_state,
        }
        print(
            "START",
            f"0x{snap.screen:02x}",
            "mode",
            snap.mode,
            "xy",
            [snap.link_x, snap.link_y],
            "bombs",
            snap.bombs,
            "keys",
            snap.keys,
            "whistle",
            whistle_before,
            "candle",
            int(read_u8(env.get_ram(), ADDR_CANDLE)),
            flush=True,
        )

        if snap.screen == ROOM_L5_WHISTLE_ITEM and whistle_before >= 1:
            walk = exit_whistle_04(env, assist, total)
            hops.append({"hop": "exit_whistle_04", **{k: walk[k] for k in walk if k != "log"}})
            print("EXIT04", walk.get("success"), "dest", f"0x{walk.get('dest'):02x}", "xy", walk.get("xy"), flush=True)
            whistle_after = int(read_u8(env.get_ram(), ADDR_WHISTLE))
            snap = read_snapshot(env.get_ram())
            if not walk.get("left_cellar"):
                failed = "Whistle basement"
                reason = "exit_04_still_in_cellar"
                return _finish(
                    env, assist, total, tag, hops, checkpoints, start,
                    whistle_before, whistle_after, failed, reason, boss, start_state,
                    extra={"exit04": walk},
                )
            if save_checkpoint:
                checkpoints.append(
                    _save_ckpt(
                        env,
                        "Level5WhistleFloor",
                        start_state,
                        "0x04 short ladder x=176 DOWN, pit y=189, left mouth x=48 UP",
                        {
                            "room": int(snap.screen),
                            "mode": int(snap.mode),
                            "whistle_0x065C": whistle_after,
                            "xy": [snap.link_x, snap.link_y],
                        },
                    )
                )
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
                extra={"exit04": walk},
            )
        if snap.screen == 0x65:
            bomb = bomb_west_from_65(env, assist, total)
            hops.append({"hop": "bomb_west_from_65", **{k: bomb[k] for k in bomb if k != "menu"}})
            print("BOMB65", bomb.get("success"), "dest", f"0x{bomb.get('dest'):02x}", flush=True)
            if not bomb.get("success"):
                failed = "west Gibdo pocket"
                reason = "bomb_west_did_not_enter_blue_darknut_stairs"
                return _finish(
                    env, assist, total, tag, hops, checkpoints, start,
                    whistle_before, whistle_after, failed, reason, boss, start_state,
                )
            if save_checkpoint:
                checkpoints.append(
                    _save_ckpt(
                        env,
                        "Level5Entered64",
                        start_state,
                        "0x65 WEST bomb settled",
                        {"room": ROOM_L5_BLUE_64, "whistle_0x065C": whistle_before},
                    )
                )
        elif snap.screen == ROOM_L5_CELLAR_07 or snap.mode in (9, 10, 11):
            hops.append({"hop": "take_center_stairs_64", "ok": True, "resumed": True, "dest": snap.screen, "mode": snap.mode})
            cellar = cellar_other_mouth(env, assist, total)
            hops.append({"hop": "cellar_other_mouth", **{k: cellar[k] for k in cellar if k != "start"}})
            print("CELLAR", cellar.get("success"), "dest", f"0x{cellar.get('dest'):02x}", "side", cellar.get("chose_side"), flush=True)
            if not cellar.get("success"):
                failed = "Whistle basement passage"
                reason = "other_mouth_did_not_enter_0x06"
                return _finish(
                    env, assist, total, tag, hops, checkpoints, start,
                    whistle_before, whistle_after, failed, reason, boss, start_state,
                )
            west = key_west_to(env, assist, total, ROOM_L5_WHISTLE_05)
            hops.append({"hop": "key_west_to_05", **west})
            print("KEYWEST", west.get("success"), "dest", f"0x{west.get('dest'):02x}", "spent", west.get("key_spent"), flush=True)
            if not west.get("success"):
                failed = "six-Darknut whistle room"
                reason = "key_west_did_not_enter_0x05"
                return _finish(
                    env, assist, total, tag, hops, checkpoints, start,
                    whistle_before, whistle_after, failed, reason, boss, start_state,
                )
            snap = read_snapshot(env.get_ram())
            n_dn = sum(
                1
                for o in snap.objects
                if 1 <= o.slot <= 12 and o.type_id == BLUE_DARKNUT_TYPE and o.hp > 0
            )
            print("FIGHT05 start_n", n_dn, flush=True)
            fight = fight_blue_darknuts(
                env, assist, total, ROOM_L5_WHISTLE_05, expected=max(6, n_dn), source=ROOM_L5_PASSAGE_06
            )
            hops.append({"hop": "fight_05", **{k: fight[k] for k in fight if k != "progress"}})
            print("FIGHT05", fight.get("ok"), "end_n", fight.get("end_n"), "f", fight.get("frames"), flush=True)
            if not fight.get("ok"):
                failed = "six-Darknut whistle room"
                reason = "darknuts_not_cleared"
                return _finish(
                    env, assist, total, tag, hops, checkpoints, start,
                    whistle_before, whistle_after, failed, reason, boss, start_state,
                    extra={"fight05": fight},
                )
            if save_checkpoint:
                checkpoints.append(
                    _save_ckpt(
                        env,
                        "Level5Cleared05",
                        start_state,
                        "0x05 6/6 blue darknuts",
                        {"room": ROOM_L5_WHISTLE_05, "whistle_0x065C": whistle_before},
                    )
                )
            pushed = push_block_stairs(env, assist, total, ROOM_L5_WHISTLE_05)
            hops.append({"hop": "push_block_stairs_05", **{k: pushed[k] for k in pushed if k != "log"}})
            print("PUSH05", pushed.get("success"), "dest", pushed.get("dest"), flush=True)
            if not pushed.get("success"):
                failed = "six-Darknut whistle room"
                reason = "block_stairs_not_taken"
                return _finish(
                    env, assist, total, tag, hops, checkpoints, start,
                    whistle_before, whistle_after, failed, reason, boss, start_state,
                )
            walk = hunt_whistle(env, assist, total)
            whistle_after = int(read_u8(env.get_ram(), ADDR_WHISTLE))
            hops.append({"hop": "hunt_whistle", "in": walk["in"], "out": walk["out"], "got": walk["got"], "via": walk.get("via")})
            print("WHISTLE", walk["in"], "->", walk["out"], "now", whistle_after, flush=True)
            if whistle_after < 1:
                failed = "Whistle basement"
                reason = "whistle_still_0"
                return _finish(
                    env, assist, total, tag, hops, checkpoints, start,
                    whistle_before, whistle_after, failed, reason, boss, start_state,
                )
            if save_checkpoint:
                snap = read_snapshot(env.get_ram())
                checkpoints.append(
                    _save_ckpt(
                        env,
                        "Level5Whistle",
                        start_state,
                        "cellar 0x07 other mouth -> 0x06 key -> 0x05 block -> 0x04",
                        {
                            "room": int(snap.screen),
                            "whistle_0x065C": whistle_after,
                            "bombs": int(snap.bombs),
                            "keys": int(snap.keys),
                        },
                    )
                )
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
            )
        elif snap.screen == ROOM_L5_WHISTLE_05:
            hops.append({"hop": "resume_cleared05", "ok": True, "dest": ROOM_L5_WHISTLE_05})
            pushed = push_block_stairs(env, assist, total, ROOM_L5_WHISTLE_05)
            hops.append({"hop": "push_block_stairs_05", **{k: pushed[k] for k in pushed if k != "log"}})
            print("PUSH05", pushed.get("success"), "dest", pushed.get("dest"), flush=True)
            if not pushed.get("success"):
                failed = "six-Darknut whistle room"
                reason = "block_stairs_not_taken"
                return _finish(
                    env, assist, total, tag, hops, checkpoints, start,
                    whistle_before, whistle_after, failed, reason, boss, start_state,
                )
            snap = read_snapshot(env.get_ram())
            if snap.screen == ROOM_L5_WHISTLE_ITEM or snap.mode in (9, 11):
                walk = take_whistle_04(env, assist, total)
            else:
                walk = hunt_whistle(env, assist, total)
            whistle_after = int(read_u8(env.get_ram(), ADDR_WHISTLE))
            hops.append({"hop": "take_whistle", "in": walk["in"], "out": walk["out"], "got": walk["got"]})
            print("WHISTLE", walk["in"], "->", walk["out"], "now", whistle_after, flush=True)
            if whistle_after < 1:
                failed = "Whistle basement"
                reason = "whistle_still_0"
                return _finish(
                    env, assist, total, tag, hops, checkpoints, start,
                    whistle_before, whistle_after, failed, reason, boss, start_state,
                )
            if save_checkpoint:
                snap = read_snapshot(env.get_ram())
                checkpoints.append(
                    _save_ckpt(
                        env,
                        "Level5Whistle",
                        start_state,
                        "0x05 block stairs -> 0x04 recorder platform",
                        {
                            "room": int(snap.screen),
                            "whistle_0x065C": whistle_after,
                            "bombs": int(snap.bombs),
                            "keys": int(snap.keys),
                        },
                    )
                )
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
            )
        elif snap.screen != ROOM_L5_BLUE_64:
            failed = "start"
            reason = f"unexpected_start_room_0x{snap.screen:02x}"
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
            )
        else:
            hops.append({"hop": "bomb_west_from_65", "ok": True, "resumed": True, "dest": ROOM_L5_BLUE_64})

        arrive64 = {
            **_inv(env.get_ram()),
            "objects": _objs(read_snapshot(env.get_ram())),
        }
        png64 = RECORDINGS_DIR / f"{tag}_64_arrive.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png64)
        print(
            "ARRIVE64",
            "xy",
            arrive64["xy"],
            "objs",
            [(o["type"], o["hp"]) for o in arrive64["objects"]],
            flush=True,
        )

        snap = read_snapshot(env.get_ram())
        n_dn = sum(
            1
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id == BLUE_DARKNUT_TYPE and o.hp > 0
        )
        fight64 = None
        if n_dn:
            print("FIGHT64 start_n", n_dn, flush=True)
            fight64 = fight_blue_darknuts(
                env, assist, total, ROOM_L5_BLUE_64, expected=n_dn, source=0x65
            )
            hops.append({"hop": "fight_64", **{k: fight64[k] for k in fight64 if k != "progress"}})
            print("FIGHT64", fight64.get("ok"), "end_n", fight64.get("end_n"), "f", fight64.get("frames"), flush=True)
            idle(env, assist, total, 16)
            pushed64 = push_block_stairs(env, assist, total, ROOM_L5_BLUE_64)
            hops.append({"hop": "push_64", **{k: pushed64[k] for k in pushed64 if k != "log"}})
            print("PUSH64", pushed64.get("success"), "dest", pushed64.get("dest"), flush=True)
            if pushed64.get("success"):
                stairs = {
                    "success": True,
                    "dest": read_snapshot(env.get_ram()).screen,
                    "mode": read_snapshot(env.get_ram()).mode,
                    "xy": [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y],
                    "via": "push_block_stairs",
                }
            else:
                stairs = take_center_stairs_64(env, assist, total)
        else:
            stairs = take_center_stairs_64(env, assist, total)
        hops.append({"hop": "take_center_stairs_64", **{k: stairs[k] for k in stairs if k != "log"}})
        print(
            "STAIRS64",
            stairs.get("success"),
            "dest",
            f"0x{stairs.get('dest'):02x}",
            "mode",
            stairs.get("mode"),
            flush=True,
        )
        if not stairs.get("success"):
            failed = "Blue Darknut stairs"
            reason = "center_stairs_not_taken"
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
                extra={"arrive64": arrive64},
            )
        if save_checkpoint:
            checkpoints.append(
                _save_ckpt(
                    env,
                    "Level5Entered07",
                    start_state,
                    "0x64 center stairs",
                    {"room": read_snapshot(env.get_ram()).screen, "mode": read_snapshot(env.get_ram()).mode},
                )
            )
        if to_stairs_only:
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
                extra={"arrive64": arrive64, "to_stairs_only": True},
            )

        cellar = cellar_other_mouth(env, assist, total)
        hops.append({"hop": "cellar_other_mouth", **{k: cellar[k] for k in cellar if k != "start"}})
        print("CELLAR", cellar.get("success"), "dest", f"0x{cellar.get('dest'):02x}", "side", cellar.get("chose_side"), flush=True)
        if not cellar.get("success"):
            failed = "Whistle basement passage"
            reason = "other_mouth_did_not_enter_0x06"
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
            )

        west = key_west_to(env, assist, total, ROOM_L5_WHISTLE_05)
        hops.append({"hop": "key_west_to_05", **west})
        print("KEYWEST", west.get("success"), "dest", f"0x{west.get('dest'):02x}", "spent", west.get("key_spent"), flush=True)
        if not west.get("success"):
            failed = "six-Darknut whistle room"
            reason = "key_west_did_not_enter_0x05"
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
            )

        snap = read_snapshot(env.get_ram())
        n_dn = sum(
            1
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id == BLUE_DARKNUT_TYPE and o.hp > 0
        )
        print("FIGHT05 start_n", n_dn, flush=True)
        fight = fight_blue_darknuts(
            env, assist, total, ROOM_L5_WHISTLE_05, expected=max(6, n_dn), source=ROOM_L5_PASSAGE_06
        )
        hops.append({"hop": "fight_05", **{k: fight[k] for k in fight if k != "progress"}})
        print("FIGHT05", fight.get("ok"), "end_n", fight.get("end_n"), "f", fight.get("frames"), flush=True)
        if not fight.get("ok"):
            failed = "six-Darknut whistle room"
            reason = "darknuts_not_cleared"
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
                extra={"fight05": fight},
            )
        if save_checkpoint:
            checkpoints.append(
                _save_ckpt(
                    env,
                    "Level5Cleared05",
                    start_state,
                    "0x05 6/6 blue darknuts",
                    {"room": ROOM_L5_WHISTLE_05, "whistle_0x065C": whistle_before},
                )
            )

        pushed = push_block_stairs(env, assist, total, ROOM_L5_WHISTLE_05)
        hops.append({"hop": "push_block_stairs_05", **{k: pushed[k] for k in pushed if k != "log"}})
        print("PUSH05", pushed.get("success"), "dest", pushed.get("dest"), flush=True)
        if not pushed.get("success"):
            failed = "six-Darknut whistle room"
            reason = "block_stairs_not_taken"
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
            )

        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_L5_WHISTLE_ITEM or snap.mode in (9, 11):
            walk = take_whistle_04(env, assist, total)
        else:
            walk = hunt_whistle(env, assist, total)
        whistle_after = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        hops.append({"hop": "take_whistle", "in": walk["in"], "out": walk["out"], "got": walk["got"]})
        print("WHISTLE", walk["in"], "->", walk["out"], "now", whistle_after, flush=True)
        if whistle_after < 1:
            failed = "Whistle basement"
            reason = "whistle_still_0"
            return _finish(
                env, assist, total, tag, hops, checkpoints, start,
                whistle_before, whistle_after, failed, reason, boss, start_state,
            )
        if save_checkpoint:
            snap = read_snapshot(env.get_ram())
            checkpoints.append(
                _save_ckpt(
                    env,
                    "Level5Whistle",
                    start_state,
                    "0x65 bomb -> 0x64 stairs -> 0x07 -> 0x06 key -> 0x05 block -> 0x04",
                    {
                        "room": int(snap.screen),
                        "whistle_0x065C": whistle_after,
                        "bombs": int(snap.bombs),
                        "keys": int(snap.keys),
                    },
                )
            )
    finally:
        env.close()

    if whistle_after >= 1 and do_boss:
        boss = _digdogger(start_state, infinite_life)

    return _report(
        tag, hops, checkpoints, start if "start" in dir() else {},
        whistle_before, whistle_after, failed, reason, boss, start_state,
    )


def _finish(env, assist, total, tag, hops, checkpoints, start, w0, w1, failed, reason, boss, source, extra=None):
    snap = read_snapshot(env.get_ram())
    png = RECORDINGS_DIR / f"{tag}_final.png"
    obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, png)
    extra = extra or {}
    extra["final"] = {**_inv(env.get_ram()), "objects": _objs(snap)}
    extra["screenshot"] = str(png.resolve())
    return _report(tag, hops, checkpoints, start, w0, w1, failed, reason, boss, source, extra)


def _report(tag, hops, checkpoints, start, w0, w1, failed, reason, boss, source, extra=None):
    body = {
        "ok": w1 >= 1,
        "status_claim": None,
        "pokes": False,
        "track": "assisted",
        "start_state": source,
        "start": start,
        "hops": hops,
        "checkpoints": checkpoints,
        "whistle_before_0x065C": w0,
        "whistle_after_0x065C": w1,
        "failed_room": failed,
        "reason": reason,
        "digdogger": boss,
        "commands": [
            "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/run_level5_whistle.py "
            f"--from-state {source} --infinite-life --save-state --tag {tag}"
        ],
    }
    if extra:
        body.update(extra)
    write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
    return body


def _digdogger(source: str, infinite_life: bool) -> dict:
    """From source (Level5Whistle24 or Cleared25 west) into Digdogger.

    One Recorder song (~180f) shrinks type 0x38 HP=240 into 0x18 HP=128.
    Do not tap B every 40f — that cuts the melody and the boss never shrinks.
    Sword the small form, grab HC 0x1A, north 0x14, TF bit 0x10.
    """
    from zelda_i.dungeon_ops import push_dir
    from zelda_i.level5_path import fight_blue_darknuts as _unused  # noqa: F401
    from zelda_i.dungeon import DoorRoute, DungeonPhase, GenericDungeonRoomController, RewardKind, RewardSpec
    from dataclasses import replace
    from zelda_i.level3_dungeon import ROOM_5B_SPEC, ROOM_59_SPEC
    from retro_harness.nes import nes_action as _nes_action

    env = make_env(GAME, source, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [1]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        west = {"success": snap.screen == 0x24, "dest": snap.screen, "skipped": True}
        if snap.screen != 0x24:
            west = walk_west_from_25(env, assist, total)
            print("AT24", west.get("success"), "dest", f"0x{west.get('dest'):02x}", flush=True)
            if not west.get("success"):
                return {"ok": False, "reason": "west_from_cleared25_missed_24", "west": west}
        menu = select_b_item_menu(env, assist, total, 5)
        walk_axis = __import__("zelda_i.level5_path", fromlist=["walk_axis"]).walk_axis
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=400)
        idle(env, assist, total, 12)
        for _ in range(16):
            env.step(_nes_action("B"))
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
        # Full melody; 0x38 -> 0x18 around 180f.
        for _ in range(8):
            idle(env, assist, total, 30)
            snap = read_snapshot(env.get_ram())
            if any(1 <= o.slot <= 12 and o.type_id == 0x18 and o.hp > 0 for o in snap.objects):
                break
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        after_b = _objs(snap)
        print("WHISTLE_B", menu, "objs", after_b, flush=True)
        bosses = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x18 and o.hp > 0]
        if not bosses:
            bosses = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x38 and o.hp > 0]
        fight = None
        if bosses:
            spec = replace(
                ROOM_5B_SPEC,
                spec_id="level5_digdogger",
                source_room=0x25,
                room_id=0x24,
                entry=DoorRoute("LEFT", ((224, 141),)),
                enemy_types=(bosses[0].type_id,),
                expected_enemy_count=len(bosses),
                required_open_doors=0,
                reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
                combat=ROOM_59_SPEC.combat,
                exit_routes=(DoorRoute("UP", ((120, 93),)),),
                max_frames=20000,
                level=LEVEL_5,
            )
            ctl = GenericDungeonRoomController(spec)
            for _ in range(spec.max_frames):
                snap = read_snapshot(env.get_ram())
                action = ctl.step(snap)
                env.step(action.action)
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                if ctl.success or ctl.phase is DungeonPhase.FAILED:
                    break
            fight = {"ok": bool(ctl.success), "frames": ctl.frames}
        leftovers = [
            o
            for o in read_snapshot(env.get_ram()).objects
            if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40) and o.hp > 0
        ]
        extra = None
        if leftovers:
            extra = fight_blue_darknuts(env, assist, total, 0x24, expected=len(leftovers), source=0x25)
        walk_axis = __import__("zelda_i.level5_path", fromlist=["walk_axis"]).walk_axis
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 224, max_f=400)
        idle(env, assist, total, 12)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 93, max_f=400)
        push_dir(env, assist, total, "UP", frames=220)
        idle(env, assist, total, 20)
        snap = read_snapshot(env.get_ram())
        tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        if snap.screen == 0x14 or snap.room_item_id == 0x1B:
            hunt = hunt_whistle  # reuse stands; walk TF via ADDR_TRIFORCE
            from zelda_i.ram import ADDR_TRIFORCE as _TF
            w0 = int(read_u8(env.get_ram(), _TF))
            for tx, ty in ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141)):
                walk_axis(env, assist, total, "y", ty, max_f=200)
                walk_axis(env, assist, total, "x", tx, max_f=200)
                idle(env, assist, total, 8)
                if int(read_u8(env.get_ram(), _TF)) > w0:
                    break
        ram = env.get_ram()
        snap = read_snapshot(ram)
        tf1 = int(read_u8(ram, ADDR_TRIFORCE))
        rec = {
            "ok": bool(tf1 & 0x10),
            "west": {k: west[k] for k in west if k != "log"},
            "menu": menu,
            "after_whistle_objs": after_b,
            "fight": fight,
            "extra": extra,
            "tf_in": tf0,
            "tf_out": tf1,
            "tf_l5": bool(tf1 & 0x10),
            "room": snap.screen,
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        }
        write_json_report(RECORDINGS_DIR / "l5_24_whistle_boss.json", rec)
        print("DIGDOGGER", rec.get("ok"), "room", f"0x{snap.screen:02x}", "tf", hex(tf1), flush=True)
        return rec
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-state", default="Level5Cleared65")
    ap.add_argument("--infinite-life", action="store_true")
    ap.add_argument("--save-state", action="store_true")
    ap.add_argument("--tag", default="l5_whistle_path")
    ap.add_argument("--to-stairs-only", action="store_true")
    ap.add_argument("--no-boss", action="store_true")
    args = ap.parse_args()
    r = run_once(
        start_state=args.from_state,
        infinite_life=args.infinite_life,
        save_checkpoint=args.save_state,
        tag=args.tag,
        to_stairs_only=args.to_stairs_only,
        do_boss=not args.no_boss,
    )
    print("OK", r.get("ok"))
    print("HOPS", r.get("hops"))
    print("WHISTLE", r.get("whistle_before_0x065C"), "->", r.get("whistle_after_0x065C"))
    print("FAILED", r.get("failed_room"), r.get("reason"))
    print("CKPT", r.get("checkpoints"))
    print("DIGDOGGER", r.get("digdogger"))
    print("status_claim", None)


if __name__ == "__main__":
    main()
