"""Assisted recon: Level 6 post-east-key graph from ``Level6EastKey``.

Maps key doors / Old Man trap / correct west path toward Compass/Rod.
Survival (``--infinite-life``) only — not Clean STATUS.

Live graph (2026-08-07)::

    0x7a -LEFT free-> 0x79
    0x7a -UP KEY-> 0x6a Old Man ⚠ (do not)
    0x79 -LEFT KEY (fire-bypass y157→141)-> 0x78 (5×0x24)
    0x78 clear -UP-> 0x68 compass Zols → 0x58 → 0x48 → 0x38 → 0x28 → Rod residual

Segment runner: ``run_level6_west_wizzrobes.py --infinite-life``.

Examples::

    uv run python nes/zelda_i/scripts/probe_level6_past_east_key.py --infinite-life
    uv run python nes/zelda_i/scripts/probe_level6_past_east_key.py \\
        --from-state Level6EastKey --infinite-life --tag l6_post_key
    uv run python nes/zelda_i/scripts/probe_level6_past_east_key.py \\
        --infinite-life --try-old-man --max-hops 6
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.level6_dungeon import Level6EastKeyController, ROOM_7A_SPEC
from zelda_i.level6_overworld import LEVEL6, LEVEL6_EAST_KEY_ROOM, LEVEL6_ENTRY_ROOM
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

DOOR_RIGHT = 0x01
DOOR_LEFT = 0x02
DOOR_DOWN = 0x04
DOOR_UP = 0x08

DOOR_TARGETS: dict[str, tuple[int, int]] = {
    "RIGHT": (208, 141),
    "LEFT": (32, 141),
    "UP": (120, 93),
    "DOWN": (120, 205),
}

# Center door push after y/x align — used when locked doors need a long press.
PUSH_FRAMES = 90
SETTLE_FRAMES = 80

def _objs(snap: ZeldaSnapshot) -> list[dict]:
    out: list[dict] = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10) or o.type_id in (0, 0xFF):
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

def _room_fields(snap: ZeldaSnapshot) -> dict:
    types = Counter(
        o.type_id
        for o in snap.objects
        if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
    )
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "sc": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "health": snap.health,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "room_obj_count": snap.room_obj_count,
        "cur_opened_doors": snap.cur_opened_doors,
        "open_doorway_mask": snap.open_doorway_mask,
        "doors": {
            "R": bool(snap.cur_opened_doors & DOOR_RIGHT),
            "L": bool(snap.cur_opened_doors & DOOR_LEFT),
            "D": bool(snap.cur_opened_doors & DOOR_DOWN),
            "U": bool(snap.cur_opened_doors & DOOR_UP),
            "raw": snap.cur_opened_doors,
        },
        "type_counts": {f"0x{k:02x}": v for k, v in sorted(types.items())},
        "type_names": {f"0x{k:02x}": object_name(k) for k in sorted(types)},
        "objects": _objs(snap),
    }

def _goto(env, assist, total: list[int], tx: int, ty: int, *, tol: int = 4, max_f: int = 400):
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - tx) <= tol and abs(snap.link_y - ty) <= tol:
            return True
        if abs(snap.link_x - tx) > tol:
            act = nes_action("RIGHT" if snap.link_x < tx else "LEFT")
        else:
            act = nes_action("DOWN" if snap.link_y < ty else "UP")
        env.step(act.action if hasattr(act, "action") else act)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    return False

def _push_dir(env, assist, total: list[int], direction: str, frames: int = PUSH_FRAMES):
    keys0 = read_snapshot(env.get_ram()).keys
    room0 = read_snapshot(env.get_ram()).screen
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        if snap.keys < keys0:
            # Key consumed — keep pushing a bit for scroll.
            pass
        env.step(nes_action(direction))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    # Settle transition.
    for _ in range(SETTLE_FRAMES):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.level == LEVEL6:
            # wait a bit more if still settling
            if snap.screen != room0:
                for _ in range(40):
                    env.step(nes_idle_action())
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                break
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])

def _try_exit(
    env,
    assist,
    total: list[int],
    direction: str,
    *,
    tag: str,
    stem: str,
) -> dict:
    """Walk to door target then push; report whether room/keys changed."""
    snap0 = read_snapshot(env.get_ram())
    before = _room_fields(snap0)
    tx, ty = DOOR_TARGETS[direction]
    # Approach: first center y/x then door.
    if direction in ("LEFT", "RIGHT"):
        _goto(env, assist, total, snap0.link_x, ty, tol=3)
        _goto(env, assist, total, tx, ty, tol=4)
    else:
        _goto(env, assist, total, tx, snap0.link_y, tol=4)
        _goto(env, assist, total, tx, ty, tol=4)

    mid = _room_fields(read_snapshot(env.get_ram()))
    _push_dir(env, assist, total, direction, frames=PUSH_FRAMES + 40)
    after_snap = read_snapshot(env.get_ram())
    after = _room_fields(after_snap)
    changed_room = after["screen"] != before["screen"]
    keys_spent = before["keys"] - after["keys"]
    png = RECORDINGS_DIR / f"{tag}_{stem}_{direction.lower()}.png"
    # Need last obs — re-step idle and capture via env; screenshot helper needs rgb.
    # Callers pass obs; we re-fetch with idle.
    return {
        "direction": direction,
        "before": before,
        "at_door": mid,
        "after": after,
        "changed_room": changed_room,
        "keys_spent": keys_spent,
        "result": (
            "room_change"
            if changed_room
            else ("key_spent_no_room" if keys_spent else "blocked")
        ),
        "screenshot": str(png),
    }

def _idle(env, assist, total: list[int], frames: int = 30):
    for _ in range(frames):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])

def _fight_clear(
    env,
    assist,
    total: list[int],
    *,
    enemy_types: tuple[int, ...],
    max_frames: int = 10000,
) -> dict:
    """Simple center-patrol sword clear for typed enemies (wizzrobe-ish)."""
    # Reuse east-key controller patterns: patrol mid, swing when close.
    from zelda_i.dungeon import (
        AliveRule,
        CombatTuning,
        DoorRoute,
        DungeonPhase,
        DungeonRoomSpec,
        GenericDungeonRoomController,
        RewardKind,
        RewardSpec,
    )

    snap = read_snapshot(env.get_ram())
    room = snap.screen
    patrol = (
        (64, 109),
        (120, 109),
        (176, 109),
        (176, 141),
        (176, 173),
        (120, 173),
        (64, 173),
        (64, 141),
        (120, 141),
    )
    spec = DungeonRoomSpec(
        spec_id=f"l6_probe_0x{room:02x}",
        source_room=room,
        room_id=room,
        entry=DoorRoute("LEFT", ((120, 141),)),
        enemy_types=enemy_types,
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE_AND_HP,
        combat=CombatTuning(
            patrol=patrol,
            engage_distance=48,
            attack_phase=2,
            patrol_attack_period=8,
            patrol_attack_hold=3,
            engage_attack_period=6,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        max_frames=max_frames,
        level=LEVEL6,
    )
    ctl = GenericDungeonRoomController(spec)
    # Backstep when stuck (wizzrobes).
    last_progress = 0
    prev_live = -1
    backstep = 0
    for frame in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return {"ok": False, "error": "death", "frames": frame}
        live = [
            o
            for o in snap.objects
            if 1 <= o.slot <= 10
            and o.type_id in enemy_types
            and o.hp > 0
        ]
        n = len(live)
        if prev_live < 0:
            prev_live = n
            last_progress = frame
        elif n < prev_live:
            prev_live = n
            last_progress = frame
            backstep = 0
        if n == 0 and frame > 60:
            # settle
            for _ in range(40):
                env.step(nes_idle_action())
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
            return {
                "ok": True,
                "frames": frame,
                "final": _room_fields(read_snapshot(env.get_ram())),
            }

        if live:
            nearest = min(
                live,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
            stuck = dist < 16 and (frame - last_progress) > 100
            if stuck or backstep > 0:
                if backstep <= 0:
                    backstep = 24
                backstep -= 1
                if backstep == 0:
                    last_progress = frame
                if snap.link_x < 40:
                    d = "RIGHT"
                elif snap.link_x > 200:
                    d = "LEFT"
                else:
                    dx = nearest.x - snap.link_x
                    dy = nearest.y - snap.link_y
                    if abs(dx) >= abs(dy):
                        d = "LEFT" if dx >= 0 else "RIGHT"
                    else:
                        d = "UP" if dy >= 0 else "DOWN"
                env.step(nes_action(d))
            else:
                act = ctl.step(snap)
                env.step(act.action)
        else:
            # wander center until spawn
            act = ctl.step(snap)
            env.step(act.action)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    return {
        "ok": False,
        "error": "timeout",
        "frames": max_frames,
        "final": _room_fields(read_snapshot(env.get_ram())),
    }

def run_probe(
    *,
    start_state: str,
    infinite_life: bool,
    try_old_man: bool,
    max_hops: int,
    save_checkpoints: bool,
    tag: str,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    track = "assisted" if infinite_life else "clean"
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        total[0] = 1

        entry = _room_fields(read_snapshot(env.get_ram()))
        reports: dict = {
            "ok": False,
            "track": track,
            "start_state": start_state,
            "entry": entry,
            "graph_edges": [],
            "room_notes": {},
            "path_log": [],
            "trap_notes": [],
        }
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_start.png")

        if not (
            entry["level"] == LEVEL6
            and entry["mode"] == PLAY_MODE
            and entry["screen"] == LEVEL6_EAST_KEY_ROOM
            and entry["keys"] >= 1
        ):
            reports["error"] = "expected_Level6EastKey_0x7a_keys>=1"
            out = RECORDINGS_DIR / f"{tag}_recon.json"
            write_json_report(out, reports)
            reports["report_path"] = str(out)
            return reports

        # --- Phase A: probe all 4 doors from 0x7a (fresh state per direction) ---
        door_probes_7a: list[dict] = []
        for direction in ("LEFT", "RIGHT", "UP", "DOWN"):
            # reload state for isolation
            env.close()
            env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
            obs, _ = reset_obs(env)
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=0)
            total = [1]
            probe = _try_exit(
                env, assist, total, direction, tag=tag, stem="7a"
            )
            # capture screenshot
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, Path(probe["screenshot"]))
            door_probes_7a.append(probe)
            edge = {
                "from": "0x7a",
                "dir": direction,
                "to": (
                    f"0x{probe['after']['screen']:02x}"
                    if probe["changed_room"]
                    else None
                ),
                "keys_spent": probe["keys_spent"],
                "result": probe["result"],
                "after_types": probe["after"].get("type_counts"),
                "after_item": probe["after"].get("room_item_id"),
            }
            reports["graph_edges"].append(edge)
            reports["path_log"].append(
                {
                    "phase": "7a_door_probe",
                    "dir": direction,
                    **edge,
                    "after": {
                        "sc": probe["after"]["sc"],
                        "keys": probe["after"]["keys"],
                        "xy": [probe["after"]["x"], probe["after"]["y"]],
                        "types": probe["after"].get("type_counts"),
                    },
                }
            )

        reports["door_probes_7a"] = door_probes_7a

        # Identify Old Man candidate: key spent or room with no combat types.
        for p in door_probes_7a:
            if p["direction"] == "LEFT":
                continue
            if p["keys_spent"] or (
                p["changed_room"] and p["after"]["keys"] < p["before"]["keys"]
            ):
                reports["trap_notes"].append(
                    {
                        "kind": "possible_old_man_or_key_door",
                        "from": "0x7a",
                        "dir": p["direction"],
                        "to": f"0x{p['after']['screen']:02x}"
                        if p["changed_room"]
                        else None,
                        "keys_spent": p["keys_spent"],
                        "types": p["after"].get("type_counts"),
                        "item": p["after"].get("room_item_id"),
                    }
                )

        # --- Phase B: LEFT 0x7a → 0x79 (free), then probe 0x79 doors ---
        env.close()
        env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        total = [1]

        left = _try_exit(env, assist, total, "LEFT", tag=tag, stem="7a_to_79")
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, Path(left["screenshot"]))
        reports["path_log"].append(
            {
                "phase": "return_entry",
                "dir": "LEFT",
                "changed_room": left["changed_room"],
                "keys_spent": left["keys_spent"],
                "after": {
                    "sc": left["after"]["sc"],
                    "keys": left["after"]["keys"],
                    "xy": [left["after"]["x"], left["after"]["y"]],
                },
            }
        )
        if not (
            left["changed_room"] and left["after"]["screen"] == LEVEL6_ENTRY_ROOM
        ):
            reports["error"] = "failed_left_7a_to_79"
            reports["left_7a"] = left
            out = RECORDINGS_DIR / f"{tag}_recon.json"
            write_json_report(out, reports)
            reports["report_path"] = str(out)
            return reports

        reports["room_notes"]["0x79"] = left["after"]
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_79_from_7a.png")
        if save_checkpoints:
            path = save_state(env, GAME_DIR, GAME, "L6Room_79_keys1")
            reports["saved_79_keys1"] = str(path)

        # Snapshot doors on 0x79 with keys=1, then probe each direction
        # from a saved mid-state by reloading... but we only have one env.
        # Probe directions carefully: save state bytes via get_state.
        state_79 = env.em.get_state()
        door_probes_79: list[dict] = []
        for direction in ("LEFT", "RIGHT", "UP", "DOWN"):
            env.em.set_state(state_79)
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            probe = _try_exit(
                env, assist, total, direction, tag=tag, stem="79"
            )
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, Path(probe["screenshot"]))
            door_probes_79.append(probe)
            edge = {
                "from": "0x79",
                "dir": direction,
                "to": (
                    f"0x{probe['after']['screen']:02x}"
                    if probe["changed_room"]
                    else None
                ),
                "keys_spent": probe["keys_spent"],
                "result": probe["result"],
                "after_types": probe["after"].get("type_counts"),
                "after_item": probe["after"].get("room_item_id"),
                "after_keys": probe["after"]["keys"],
            }
            reports["graph_edges"].append(edge)
            reports["path_log"].append(
                {
                    "phase": "79_door_probe",
                    "dir": direction,
                    **edge,
                    "after": {
                        "sc": probe["after"]["sc"],
                        "keys": probe["after"]["keys"],
                        "xy": [probe["after"]["x"], probe["after"]["y"]],
                        "types": probe["after"].get("type_counts"),
                        "item": probe["after"].get("room_item_id"),
                    },
                }
            )
            if (
                direction != "RIGHT"
                and probe["changed_room"]
                and probe["keys_spent"]
            ):
                # Candidate route room — note Old Man vs combat.
                types = probe["after"].get("type_counts") or {}
                if not types or all(
                    t in ("0x4f", "0x4e", "0x5a")  # fireballs/statues-ish
                    for t in types
                ):
                    reports["trap_notes"].append(
                        {
                            "kind": "possible_old_man_room",
                            "from": "0x79",
                            "dir": direction,
                            "to": f"0x{probe['after']['screen']:02x}",
                            "keys_spent": probe["keys_spent"],
                            "types": types,
                        }
                    )

        reports["door_probes_79"] = door_probes_79

        # --- Phase C: correct path — LEFT from 0x79 (key spend) if live ---
        left_79 = next(
            (p for p in door_probes_79 if p["direction"] == "LEFT"), None
        )
        if left_79 and left_79["changed_room"]:
            west_room = left_79["after"]["screen"]
            reports["west_of_entry"] = {
                "room": west_room,
                "sc": f"0x{west_room:02x}",
                "keys_spent": left_79["keys_spent"],
                "keys_after": left_79["after"]["keys"],
                "types": left_79["after"].get("type_counts"),
                "type_names": left_79["after"].get("type_names"),
                "item": left_79["after"].get("room_item_id"),
                "doors": left_79["after"].get("doors"),
            }
            reports["room_notes"][f"0x{west_room:02x}"] = left_79["after"]

            # Reload onto west room via 79 left path and clear/explore hops.
            env.em.set_state(state_79)
            obs, *_ = env.step(nes_idle_action())
            left2 = _try_exit(
                env, assist, total, "LEFT", tag=tag, stem="79_west"
            )
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_west_0x{west_room:02x}.png")
            if save_checkpoints and left2["changed_room"]:
                path = save_state(
                    env, GAME_DIR, GAME, f"L6Room_{west_room:02X}"
                )
                reports["saved_west"] = str(path)

            # Optional: try clear wizzrobes if present (0x24).
            snap = read_snapshot(env.get_ram())
            types_present = {
                o.type_id
                for o in snap.objects
                if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
            }
            if 0x24 in types_present or 0x23 in types_present:
                enemy = tuple(
                    t for t in (0x24, 0x23) if t in types_present
                ) or (0x24,)
                clear = _fight_clear(env, assist, total, enemy_types=enemy)
                reports["west_clear"] = clear
                obs, *_ = env.step(nes_idle_action())
                save_rgb_png(
                    obs, RECORDINGS_DIR / f"{tag}_west_cleared.png"
                )
                if clear.get("ok"):
                    state_west = env.em.get_state()
                    # Probe exits after clear
                    west_exits = []
                    for direction in ("LEFT", "RIGHT", "UP", "DOWN"):
                        env.em.set_state(state_west)
                        obs, *_ = env.step(nes_idle_action())
                        pr = _try_exit(
                            env,
                            assist,
                            total,
                            direction,
                            tag=tag,
                            stem=f"west_{west_room:02x}",
                        )
                        obs, *_ = env.step(nes_idle_action())
                        save_rgb_png(obs, Path(pr["screenshot"]))
                        west_exits.append(pr)
                        reports["graph_edges"].append(
                            {
                                "from": f"0x{west_room:02x}",
                                "dir": direction,
                                "to": (
                                    f"0x{pr['after']['screen']:02x}"
                                    if pr["changed_room"]
                                    else None
                                ),
                                "keys_spent": pr["keys_spent"],
                                "result": pr["result"],
                                "after_types": pr["after"].get("type_counts"),
                                "after_item": pr["after"].get("room_item_id"),
                            }
                        )
                    reports["door_probes_west"] = [
                        {
                            "direction": p["direction"],
                            "changed_room": p["changed_room"],
                            "keys_spent": p["keys_spent"],
                            "to": f"0x{p['after']['screen']:02x}"
                            if p["changed_room"]
                            else None,
                            "types": p["after"].get("type_counts"),
                            "item": p["after"].get("room_item_id"),
                        }
                        for p in west_exits
                    ]

                    # Hop UP chain (walkthrough: west room UP → compass Zols…)
                    env.em.set_state(state_west)
                    obs, *_ = env.step(nes_idle_action())
                    hops = []
                    for hop_i in range(max_hops):
                        snap = read_snapshot(env.get_ram())
                        room_before = snap.screen
                        keys_before = snap.keys
                        # clear current room lightly if enemies
                        types_now = {
                            o.type_id
                            for o in snap.objects
                            if 1 <= o.slot <= 10
                            and o.type_id not in (0, 0xFF)
                            and o.hp > 0
                        }
                        combat_types = tuple(
                            t
                            for t in types_now
                            if t not in {0x5a, 0x4f, 0x4e, 0x60, 0x61, 0x62}
                        )
                        if combat_types:
                            clr = _fight_clear(
                                env,
                                assist,
                                total,
                                enemy_types=combat_types,
                                max_frames=8000,
                            )
                        else:
                            clr = {"ok": True, "skipped": True}
                        # Prefer UP then LEFT free doors after clear.
                        moved = False
                        for direction in ("UP", "LEFT", "RIGHT", "DOWN"):
                            st = env.em.get_state()
                            pr = _try_exit(
                                env,
                                assist,
                                total,
                                direction,
                                tag=tag,
                                stem=f"hop{hop_i}",
                            )
                            if (
                                pr["changed_room"]
                                and pr["after"]["screen"] != room_before
                            ):
                                hops.append(
                                    {
                                        "hop": hop_i,
                                        "from": f"0x{room_before:02x}",
                                        "dir": direction,
                                        "to": f"0x{pr['after']['screen']:02x}",
                                        "keys_before": keys_before,
                                        "keys_after": pr["after"]["keys"],
                                        "keys_spent": pr["keys_spent"],
                                        "types": pr["after"].get(
                                            "type_counts"
                                        ),
                                        "item": pr["after"].get(
                                            "room_item_id"
                                        ),
                                        "clear": clr.get("ok"),
                                    }
                                )
                                reports["graph_edges"].append(
                                    {
                                        "from": f"0x{room_before:02x}",
                                        "dir": direction,
                                        "to": f"0x{pr['after']['screen']:02x}",
                                        "keys_spent": pr["keys_spent"],
                                        "result": pr["result"],
                                        "after_types": pr["after"].get(
                                            "type_counts"
                                        ),
                                        "after_item": pr["after"].get(
                                            "room_item_id"
                                        ),
                                    }
                                )
                                reports["room_notes"][
                                    f"0x{pr['after']['screen']:02x}"
                                ] = pr["after"]
                                obs, *_ = env.step(nes_idle_action())
                                save_rgb_png(
                                    obs,
                                    RECORDINGS_DIR
                                    / f"{tag}_hop{hop_i}_0x{pr['after']['screen']:02x}.png",
                                )
                                moved = True
                                # Prefer UP first success; don't restore
                                break
                            # restore and try next dir
                            env.em.set_state(st)
                            obs, *_ = env.step(nes_idle_action())
                        if not moved:
                            hops.append(
                                {
                                    "hop": hop_i,
                                    "from": f"0x{room_before:02x}",
                                    "stuck": True,
                                    "clear": clr,
                                    "final": _room_fields(
                                        read_snapshot(env.get_ram())
                                    ),
                                }
                            )
                            break
                    reports["north_west_hops"] = hops

        # --- Phase D: optional Old Man confirmation on 0x7a non-LEFT ---
        if try_old_man:
            env.close()
            env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
            obs, _ = reset_obs(env)
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=0)
            total = [1]
            old_man_hits = []
            for direction in ("UP", "RIGHT", "DOWN"):
                env.close()
                env = make_env(
                    GAME, start_state, GAME_DIR, render_mode="rgb_array"
                )
                obs, _ = reset_obs(env)
                obs, *_ = env.step(nes_idle_action())
                if assist is not None:
                    assist.apply_env(env, frame=0)
                total = [1]
                pr = _try_exit(
                    env, assist, total, direction, tag=tag, stem="oldman"
                )
                obs, *_ = env.step(nes_idle_action())
                save_rgb_png(obs, Path(pr["screenshot"]))
                if pr["keys_spent"] or pr["changed_room"]:
                    old_man_hits.append(
                        {
                            "dir": direction,
                            "keys_spent": pr["keys_spent"],
                            "to": f"0x{pr['after']['screen']:02x}"
                            if pr["changed_room"]
                            else None,
                            "types": pr["after"].get("type_counts"),
                            "item": pr["after"].get("room_item_id"),
                            "keys_after": pr["after"]["keys"],
                        }
                    )
            reports["old_man_probe"] = old_man_hits

        reports["ok"] = True
        reports["frames"] = total[0]
        reports["assist"] = assist.report() if assist else None
        reports["final"] = _room_fields(read_snapshot(env.get_ram()))
        # Dedupe graph edges
        seen = set()
        deduped = []
        for e in reports["graph_edges"]:
            key = (e.get("from"), e.get("dir"), e.get("to"), e.get("result"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)
        reports["graph_edges"] = deduped

        out = RECORDINGS_DIR / f"{tag}_recon.json"
        write_json_report(out, reports)
        reports["report_path"] = str(out)
        return reports
    finally:
        env.close()

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level6EastKey")
    p.add_argument("--infinite-life", action="store_true")
    p.add_argument(
        "--try-old-man",
        action="store_true",
        help="Also probe 0x7a UP/RIGHT/DOWN for key-waste Old Man door",
    )
    p.add_argument("--max-hops", type=int, default=6)
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--tag", default="l6_post_key")
    args = p.parse_args(argv)

    report = run_probe(
        start_state=args.from_state,
        infinite_life=args.infinite_life,
        try_old_man=args.try_old_man,
        max_hops=args.max_hops,
        save_checkpoints=args.save_state,
        tag=args.tag,
    )
    print(f"ok={report.get('ok')} track={report.get('track')}")
    if report.get("error"):
        print(f"error={report['error']}")
    print("edges:")
    for e in report.get("graph_edges") or []:
        print(
            f"  {e.get('from')} -{e.get('dir')}-> {e.get('to')} "
            f"keys_spent={e.get('keys_spent')} result={e.get('result')} "
            f"types={e.get('after_types')} item={e.get('after_item')}"
        )
    if report.get("west_of_entry"):
        print(f"west_of_entry={report['west_of_entry']}")
    if report.get("trap_notes"):
        print(f"traps={report['trap_notes']}")
    if report.get("north_west_hops"):
        print("hops:")
        for h in report["north_west_hops"]:
            print(f"  {h}")
    print(f"wrote {report.get('report_path')}")
    return 0 if report.get("ok") else 1

if __name__ == "__main__":
    raise SystemExit(main())
