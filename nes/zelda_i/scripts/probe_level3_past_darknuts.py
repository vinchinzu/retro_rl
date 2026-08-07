"""Assisted recon: Level 3 Manji past Darknuts (0x5b) toward Raft.

Maps doors from ``Level3Darknuts`` (3× Darknut type 0x0b), optional combat
clear, hop north into 0x4b, and multi-hop free-explore toward Compass / Raft.
Survival (``--infinite-life``) default for recon — not Clean STATUS.

Known graph (pre-probe)::

    0x7c --west LEFT+UP--> 0x7b (6 Zol + key, pure Clean)
    0x7b --UP x≈120--> 0x6b (5 Zol, pure clear)
    0x6b --UP after clear--> 0x5b Darknuts  checkpoint Level3Darknuts
    0x5b --north open (graph)--> 0x4b (3× Zol + key)

Source walkthrough after: bombs, Compass west path, staircase → Raft,
backtrack → Manhandla → TF 0x04.

Examples::

    uv run python nes/zelda_i/scripts/probe_level3_past_darknuts.py --infinite-life
    uv run python nes/zelda_i/scripts/probe_level3_past_darknuts.py \\
        --from-state Level3Darknuts --infinite-life --tag l3_past_5b
    uv run python nes/zelda_i/scripts/probe_level3_past_darknuts.py \\
        --infinite-life --skip-clear --max-hops 4
    uv run python nes/zelda_i/scripts/probe_level3_past_darknuts.py \\
        --infinite-life --poke-bombs 8 --try-bombs
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
from zelda_i.level3_dungeon import (
    DARKNUT_OBJECT_TYPE,
    ROOM_L3_DARKNUTS,
)
from zelda_i.level3_overworld import LEVEL3
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_COMPASS,
    ADDR_RAFT,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

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

# Bomb stand candidates (recon). L2 N-wall style + side walls.
BOMB_STANDS: list[tuple[str, int, int]] = [
    ("UP", 120, 101),
    ("DOWN", 120, 189),
    ("LEFT", 48, 141),
    ("RIGHT", 192, 141),
    ("UP", 96, 101),
    ("UP", 144, 101),
    ("LEFT", 48, 117),
    ("LEFT", 48, 165),
    ("RIGHT", 192, 117),
    ("RIGHT", 192, 165),
]

PUSH_FRAMES = 90
SETTLE_FRAMES = 80

# Non-combat object types (skip in free-explore clear).
_NON_COMBAT_TYPES = {0x5A, 0x4F, 0x4E, 0x60, 0x61, 0x62, 0x5B, 0x5C, 0x49}


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


def _room_fields(snap: ZeldaSnapshot, ram=None) -> dict:
    types = Counter(
        o.type_id
        for o in snap.objects
        if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
    )
    raft = int(read_u8(ram, ADDR_RAFT)) if ram is not None else None
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
        "raft": raft,
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
            pass
        env.step(nes_action(direction))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    for _ in range(SETTLE_FRAMES):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.level == LEVEL3:
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
    before = _room_fields(snap0, env.get_ram())
    tx, ty = DOOR_TARGETS[direction]
    if direction in ("LEFT", "RIGHT"):
        _goto(env, assist, total, snap0.link_x, ty, tol=3)
        _goto(env, assist, total, tx, ty, tol=4)
    else:
        # Strict x≈120 for north/south like L3 pure residual.
        _goto(env, assist, total, tx, snap0.link_y, tol=3)
        _goto(env, assist, total, tx, ty, tol=4)

    mid = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    _push_dir(env, assist, total, direction, frames=PUSH_FRAMES + 40)
    after_snap = read_snapshot(env.get_ram())
    after = _room_fields(after_snap, env.get_ram())
    changed_room = after["screen"] != before["screen"]
    keys_spent = before["keys"] - after["keys"]
    png = RECORDINGS_DIR / f"{tag}_{stem}_{direction.lower()}.png"
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


def _poke_bombs(env, n: int = 8) -> str:
    """RECON-ONLY inventory poke for bomb-wall mapping. Document in report."""
    try:
        env.unwrapped.data.set_value("bombs", int(n) & 0xFF)
        return f"bombs={n}"
    except Exception as exc:
        # Fallback: direct WRAM write via retro data if available.
        try:
            ram = env.get_ram()
            # Some integrations expose set_value only; try data dict.
            env.unwrapped.data.set_value("bombs", n)
            return f"bombs={n}"
        except Exception as exc2:
            return f"poke_fail={exc!r}/{exc2!r}"


def _try_bomb_stand(
    env,
    assist,
    total: list[int],
    face: str,
    x: int,
    y: int,
    *,
    tag: str,
    stem: str,
) -> dict:
    """Stand at (x,y), face wall, press B, wait for hole, push into new room."""
    snap0 = read_snapshot(env.get_ram())
    before = _room_fields(snap0, env.get_ram())
    bombs0 = before["bombs"]
    _goto(env, assist, total, x, y, tol=3, max_f=500)
    # Face wall and place bomb.
    for _ in range(8):
        env.step(nes_action(face))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    env.step(nes_action(face, "B"))
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])
    # Wait for fuse (~60–90f) then push.
    for _ in range(100):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    room0 = before["screen"]
    for _ in range(PUSH_FRAMES + 60):
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        env.step(nes_action(face))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    for _ in range(SETTLE_FRAMES):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    after = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    changed = after["screen"] != before["screen"]
    png = RECORDINGS_DIR / f"{tag}_{stem}_bomb_{face.lower()}_{x}_{y}.png"
    return {
        "face": face,
        "stand": [x, y],
        "before": before,
        "after": after,
        "bombs_before": bombs0,
        "bombs_after": after["bombs"],
        "bombs_spent": bombs0 - after["bombs"],
        "changed_room": changed,
        "result": "room_change" if changed else "no_open",
        "screenshot": str(png),
    }


def _fight_clear(
    env,
    assist,
    total: list[int],
    *,
    enemy_types: tuple[int, ...],
    max_frames: int = 12000,
) -> dict:
    """Generic center-patrol sword clear (assist OK — Darknuts side/back hits)."""
    from zelda_i.dungeon import (
        AliveRule,
        CombatTuning,
        DoorRoute,
        DungeonRoomSpec,
        GenericDungeonRoomController,
        RewardKind,
        RewardSpec,
    )

    snap = read_snapshot(env.get_ram())
    room = snap.screen
    # Wider patrol — Darknuts need side/back; circle mid + corners.
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
        (100, 125),
        (140, 157),
        (80, 157),
        (160, 125),
    )
    spec = DungeonRoomSpec(
        spec_id=f"l3_probe_0x{room:02x}",
        source_room=room,
        room_id=room,
        entry=DoorRoute("DOWN", ((120, 205),)),
        enemy_types=enemy_types,
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE_AND_HP,
        combat=CombatTuning(
            patrol=patrol,
            engage_distance=40,
            attack_phase=2,
            patrol_attack_period=6,
            patrol_attack_hold=3,
            engage_attack_period=5,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        max_frames=max_frames,
        level=LEVEL3,
    )
    ctl = GenericDungeonRoomController(spec)
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
            for _ in range(50):
                env.step(nes_idle_action())
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
            return {
                "ok": True,
                "frames": frame,
                "final": _room_fields(read_snapshot(env.get_ram()), env.get_ram()),
            }

        if live:
            nearest = min(
                live,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
            stuck = dist < 20 and (frame - last_progress) > 120
            # Prefer approach from side (not face-to-face with Darknut).
            if stuck or backstep > 0:
                if backstep <= 0:
                    backstep = 28
                backstep -= 1
                if backstep == 0:
                    last_progress = frame
                if snap.link_x < 48:
                    d = "RIGHT"
                elif snap.link_x > 192:
                    d = "LEFT"
                elif snap.link_y < 100:
                    d = "DOWN"
                elif snap.link_y > 190:
                    d = "UP"
                else:
                    dx = nearest.x - snap.link_x
                    dy = nearest.y - snap.link_y
                    # Circle: move perpendicular to enemy vector.
                    if abs(dx) >= abs(dy):
                        d = "UP" if (frame // 20) % 2 == 0 else "DOWN"
                    else:
                        d = "LEFT" if (frame // 20) % 2 == 0 else "RIGHT"
                env.step(nes_action(d))
            else:
                act = ctl.step(snap)
                env.step(act.action)
        else:
            act = ctl.step(snap)
            env.step(act.action)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    return {
        "ok": False,
        "error": "timeout",
        "frames": max_frames,
        "final": _room_fields(read_snapshot(env.get_ram()), env.get_ram()),
    }


def _edge(
    from_room: int | str,
    direction: str,
    probe: dict,
) -> dict:
    fr = from_room if isinstance(from_room, str) else f"0x{from_room:02x}"
    return {
        "from": fr,
        "dir": direction,
        "to": (
            f"0x{probe['after']['screen']:02x}" if probe["changed_room"] else None
        ),
        "keys_spent": probe["keys_spent"],
        "result": probe["result"],
        "after_types": probe["after"].get("type_counts"),
        "after_item": probe["after"].get("room_item_id"),
        "after_keys": probe["after"].get("keys"),
        "after_bombs": probe["after"].get("bombs"),
        "after_doors": probe["after"].get("doors"),
        "after_raft": probe["after"].get("raft"),
    }


def _reload(env, start_state: str, assist, total: list[int]):
    env.close()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    obs, *_ = env.step(nes_idle_action())
    if assist is not None:
        assist.apply_env(env, frame=0)
    total[0] = 1
    return env, obs


def run_probe(
    *,
    start_state: str,
    infinite_life: bool,
    skip_clear: bool,
    try_bombs: bool,
    poke_bombs: int | None,
    max_hops: int,
    save_checkpoints: bool,
    tag: str,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    track = "assisted" if infinite_life else "clean"
    recon_notes: list[str] = []
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        total[0] = 1

        # Settle a bit so Darknuts appear.
        _idle(env, assist, total, 40)
        entry = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
        reports: dict = {
            "ok": False,
            "track": track,
            "start_state": start_state,
            "entry": entry,
            "graph_edges": [],
            "room_notes": {},
            "path_log": [],
            "trap_notes": [],
            "recon_notes": recon_notes,
            "intervention_class": "survival" if infinite_life else "clean",
        }
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_start.png")
        reports["room_notes"][entry["sc"]] = entry

        if not (
            entry["level"] == LEVEL3
            and entry["mode"] == PLAY_MODE
            and entry["screen"] == ROOM_L3_DARKNUTS
        ):
            reports["error"] = (
                f"expected Level3Darknuts room 0x5b play; got "
                f"level={entry['level']} mode={entry['mode']} "
                f"screen=0x{entry['screen']:02x}"
            )
            out = RECORDINGS_DIR / f"{tag}_recon.json"
            write_json_report(out, reports)
            reports["report_path"] = str(out)
            return reports

        # --- Phase A: probe 4 doors from 0x5b WITHOUT clear (fresh each dir) ---
        door_probes_5b: list[dict] = []
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            env, obs = _reload(env, start_state, assist, total)
            _idle(env, assist, total, 30)
            probe = _try_exit(
                env, assist, total, direction, tag=tag, stem="5b_raw"
            )
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, Path(probe["screenshot"]))
            door_probes_5b.append(probe)
            edge = _edge(ROOM_L3_DARKNUTS, direction, probe)
            edge["phase"] = "5b_raw"
            reports["graph_edges"].append(edge)
            reports["path_log"].append(
                {
                    "phase": "5b_raw_door",
                    "dir": direction,
                    **edge,
                    "after": {
                        "sc": probe["after"]["sc"],
                        "keys": probe["after"]["keys"],
                        "xy": [probe["after"]["x"], probe["after"]["y"]],
                        "types": probe["after"].get("type_counts"),
                        "doors": probe["after"].get("doors"),
                    },
                }
            )
            if probe["changed_room"]:
                sc = probe["after"]["sc"]
                reports["room_notes"][sc] = probe["after"]

        reports["door_probes_5b_raw"] = [
            {
                "direction": p["direction"],
                "changed_room": p["changed_room"],
                "keys_spent": p["keys_spent"],
                "to": f"0x{p['after']['screen']:02x}" if p["changed_room"] else None,
                "types": p["after"].get("type_counts"),
                "item": p["after"].get("room_item_id"),
                "doors": p["after"].get("doors"),
                "result": p["result"],
            }
            for p in door_probes_5b
        ]

        # --- Phase B: clear Darknuts (optional), re-probe doors ---
        clear_report = None
        if not skip_clear:
            env, obs = _reload(env, start_state, assist, total)
            _idle(env, assist, total, 40)
            clear_report = _fight_clear(
                env,
                assist,
                total,
                enemy_types=(DARKNUT_OBJECT_TYPE,),
                max_frames=15000,
            )
            reports["darknut_clear"] = {
                "ok": clear_report.get("ok"),
                "error": clear_report.get("error"),
                "frames": clear_report.get("frames"),
                "final": clear_report.get("final"),
            }
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_5b_cleared.png")

            if clear_report.get("ok"):
                if save_checkpoints:
                    path = save_state(env, GAME_DIR, GAME, "Level3_5B_Cleared")
                    reports["saved_5b_cleared"] = str(path)
                state_cleared = env.em.get_state()
                door_probes_cleared: list[dict] = []
                for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                    env.em.set_state(state_cleared)
                    obs, *_ = env.step(nes_idle_action())
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                    probe = _try_exit(
                        env,
                        assist,
                        total,
                        direction,
                        tag=tag,
                        stem="5b_clear",
                    )
                    obs, *_ = env.step(nes_idle_action())
                    save_rgb_png(obs, Path(probe["screenshot"]))
                    door_probes_cleared.append(probe)
                    edge = _edge(ROOM_L3_DARKNUTS, direction, probe)
                    edge["phase"] = "5b_after_clear"
                    reports["graph_edges"].append(edge)
                    reports["path_log"].append(
                        {
                            "phase": "5b_cleared_door",
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
                    if probe["changed_room"]:
                        reports["room_notes"][probe["after"]["sc"]] = probe["after"]

                reports["door_probes_5b_cleared"] = [
                    {
                        "direction": p["direction"],
                        "changed_room": p["changed_room"],
                        "keys_spent": p["keys_spent"],
                        "to": (
                            f"0x{p['after']['screen']:02x}"
                            if p["changed_room"]
                            else None
                        ),
                        "types": p["after"].get("type_counts"),
                        "item": p["after"].get("room_item_id"),
                        "result": p["result"],
                    }
                    for p in door_probes_cleared
                ]

                # Optional bomb stands from cleared 0x5b.
                if try_bombs:
                    env.em.set_state(state_cleared)
                    obs, *_ = env.step(nes_idle_action())
                    poke_note = None
                    if poke_bombs is not None:
                        poke_note = _poke_bombs(env, poke_bombs)
                        recon_notes.append(
                            f"RECON inventory poke: {poke_note} (not Clean)"
                        )
                        _idle(env, assist, total, 5)
                    bombs_now = read_snapshot(env.get_ram()).bombs
                    bomb_hits = []
                    if bombs_now > 0:
                        state_bombs = env.em.get_state()
                        for face, bx, by in BOMB_STANDS:
                            env.em.set_state(state_bombs)
                            obs, *_ = env.step(nes_idle_action())
                            if assist is not None:
                                assist.apply_env(env, frame=total[0])
                            # re-poke if bombs depleted mid-loop
                            if read_snapshot(env.get_ram()).bombs <= 0 and poke_bombs:
                                _poke_bombs(env, poke_bombs)
                            br = _try_bomb_stand(
                                env,
                                assist,
                                total,
                                face,
                                bx,
                                by,
                                tag=tag,
                                stem="5b",
                            )
                            obs, *_ = env.step(nes_idle_action())
                            save_rgb_png(obs, Path(br["screenshot"]))
                            bomb_hits.append(
                                {
                                    "face": br["face"],
                                    "stand": br["stand"],
                                    "result": br["result"],
                                    "to": (
                                        f"0x{br['after']['screen']:02x}"
                                        if br["changed_room"]
                                        else None
                                    ),
                                    "bombs_spent": br["bombs_spent"],
                                    "types": br["after"].get("type_counts"),
                                }
                            )
                            if br["changed_room"]:
                                reports["graph_edges"].append(
                                    {
                                        "from": "0x5b",
                                        "dir": f"BOMB_{face}",
                                        "to": f"0x{br['after']['screen']:02x}",
                                        "keys_spent": 0,
                                        "result": "bomb_open",
                                        "stand": br["stand"],
                                        "after_types": br["after"].get(
                                            "type_counts"
                                        ),
                                        "after_item": br["after"].get(
                                            "room_item_id"
                                        ),
                                        "phase": "5b_bomb",
                                    }
                                )
                                reports["room_notes"][
                                    f"0x{br['after']['screen']:02x}"
                                ] = br["after"]
                                # keep exploring from this hole if wanted later
                    else:
                        recon_notes.append(
                            "no bombs for wall probe "
                            "(use --poke-bombs N for recon only)"
                        )
                    reports["bomb_probes_5b"] = bomb_hits

        # --- Phase C: north hop 0x5b → 0x4b (3× Zol + key) ---
        env, obs = _reload(env, start_state, assist, total)
        _idle(env, assist, total, 30)
        north = _try_exit(env, assist, total, "UP", tag=tag, stem="5b_to_4b")
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, Path(north["screenshot"]))
        reports["path_log"].append(
            {
                "phase": "north_to_4b",
                "changed_room": north["changed_room"],
                "to": (
                    f"0x{north['after']['screen']:02x}"
                    if north["changed_room"]
                    else None
                ),
                "types": north["after"].get("type_counts"),
                "item": north["after"].get("room_item_id"),
                "keys": north["after"].get("keys"),
            }
        )

        if north["changed_room"]:
            dest = north["after"]["screen"]
            reports["room_notes"][f"0x{dest:02x}"] = north["after"]
            reports["graph_edges"].append(
                {**_edge(ROOM_L3_DARKNUTS, "UP", north), "phase": "north_hop"}
            )
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_0x{dest:02x}_arrive.png")
            if save_checkpoints:
                path = save_state(env, GAME_DIR, GAME, f"L3Room_{dest:02X}")
                reports[f"saved_0x{dest:02x}"] = str(path)

            state_next = env.em.get_state()
            next_probes = []
            for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                env.em.set_state(state_next)
                obs, *_ = env.step(nes_idle_action())
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                pr = _try_exit(
                    env, assist, total, direction, tag=tag, stem=f"{dest:02x}_raw"
                )
                obs, *_ = env.step(nes_idle_action())
                save_rgb_png(obs, Path(pr["screenshot"]))
                next_probes.append(pr)
                edge = _edge(dest, direction, pr)
                edge["phase"] = f"0x{dest:02x}_raw"
                reports["graph_edges"].append(edge)
                if pr["changed_room"]:
                    reports["room_notes"][pr["after"]["sc"]] = pr["after"]
            reports[f"door_probes_0x{dest:02x}_raw"] = [
                {
                    "direction": p["direction"],
                    "changed_room": p["changed_room"],
                    "keys_spent": p["keys_spent"],
                    "to": (
                        f"0x{p['after']['screen']:02x}"
                        if p["changed_room"]
                        else None
                    ),
                    "types": p["after"].get("type_counts"),
                    "item": p["after"].get("room_item_id"),
                    "result": p["result"],
                }
                for p in next_probes
            ]

            # Clear Zols and try key pickup on 0x4b.
            env.em.set_state(state_next)
            obs, *_ = env.step(nes_idle_action())
            snap = read_snapshot(env.get_ram())
            combat_types = tuple(
                sorted(
                    {
                        o.type_id
                        for o in snap.objects
                        if 1 <= o.slot <= 10
                        and o.type_id not in (0, 0xFF)
                        and o.type_id not in _NON_COMBAT_TYPES
                        and o.hp > 0
                    }
                )
            )
            if combat_types:
                zol_clear = _fight_clear(
                    env,
                    assist,
                    total,
                    enemy_types=combat_types,
                    max_frames=10000,
                )
            else:
                zol_clear = {"ok": True, "skipped": True, "frames": 0}
            reports[f"clear_0x{dest:02x}"] = {
                "ok": zol_clear.get("ok"),
                "error": zol_clear.get("error"),
                "frames": zol_clear.get("frames"),
                "enemy_types": [f"0x{t:02x}" for t in combat_types],
                "final": zol_clear.get("final"),
            }
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_0x{dest:02x}_cleared.png")

            if zol_clear.get("ok"):
                keys0 = read_snapshot(env.get_ram()).keys
                for _ in range(500):
                    snap = read_snapshot(env.get_ram())
                    if snap.keys > keys0:
                        break
                    # roam mid room for fixed key drop
                    tx = 80 + (_ % 5) * 20
                    ty = 120 + ((_ // 5) % 4) * 16
                    if abs(snap.link_x - tx) > 4:
                        d = "RIGHT" if snap.link_x < tx else "LEFT"
                    elif abs(snap.link_y - ty) > 4:
                        d = "DOWN" if snap.link_y < ty else "UP"
                    else:
                        d = "UP"
                    env.step(nes_action(d))
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                after_pick = _room_fields(
                    read_snapshot(env.get_ram()), env.get_ram()
                )
                reports[f"key_pickup_0x{dest:02x}"] = {
                    "keys_before": keys0,
                    "keys_after": after_pick["keys"],
                    "picked": after_pick["keys"] > keys0,
                    "final": after_pick,
                }
                if after_pick["keys"] > keys0:
                    recon_notes.append(
                        f"0x{dest:02x} key pickup: {keys0}→{after_pick['keys']}"
                    )

                state_cleared_next = env.em.get_state()
                cleared_probes = []
                for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                    env.em.set_state(state_cleared_next)
                    obs, *_ = env.step(nes_idle_action())
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                    pr = _try_exit(
                        env,
                        assist,
                        total,
                        direction,
                        tag=tag,
                        stem=f"{dest:02x}_clr",
                    )
                    obs, *_ = env.step(nes_idle_action())
                    save_rgb_png(obs, Path(pr["screenshot"]))
                    cleared_probes.append(pr)
                    edge = _edge(dest, direction, pr)
                    edge["phase"] = f"0x{dest:02x}_cleared"
                    reports["graph_edges"].append(edge)
                    if pr["changed_room"]:
                        reports["room_notes"][pr["after"]["sc"]] = pr["after"]
                reports[f"door_probes_0x{dest:02x}_cleared"] = [
                    {
                        "direction": p["direction"],
                        "changed_room": p["changed_room"],
                        "keys_spent": p["keys_spent"],
                        "to": (
                            f"0x{p['after']['screen']:02x}"
                            if p["changed_room"]
                            else None
                        ),
                        "types": p["after"].get("type_counts"),
                        "item": p["after"].get("room_item_id"),
                        "result": p["result"],
                    }
                    for p in cleared_probes
                ]

        # --- Phase D: Compass west path 0x5b LEFT → 0x5a (source Raft route) ---
        # Walkthrough: LEFT Keese+Compass → key LEFT → Darknuts → DOWN → stairs → Raft
        env, obs = _reload(env, start_state, assist, total)
        _idle(env, assist, total, 30)
        west = _try_exit(env, assist, total, "LEFT", tag=tag, stem="5b_to_5a")
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, Path(west["screenshot"]))
        reports["path_log"].append(
            {
                "phase": "west_to_compass",
                "changed_room": west["changed_room"],
                "to": (
                    f"0x{west['after']['screen']:02x}"
                    if west["changed_room"]
                    else None
                ),
                "types": west["after"].get("type_counts"),
                "item": west["after"].get("room_item_id"),
                "item_name": west["after"].get("room_item_name"),
                "keys": west["after"].get("keys"),
                "bombs": west["after"].get("bombs"),
            }
        )

        if west["changed_room"]:
            west_room = west["after"]["screen"]
            reports["room_notes"][f"0x{west_room:02x}"] = west["after"]
            reports["graph_edges"].append(
                {**_edge(ROOM_L3_DARKNUTS, "LEFT", west), "phase": "compass_west"}
            )
            if west["after"].get("room_item_id") == 0x16:
                recon_notes.append(
                    f"0x{west_room:02x} RoomItemId compass (0x16) live"
                )
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_0x{west_room:02x}_arrive.png")
            if save_checkpoints:
                path = save_state(env, GAME_DIR, GAME, f"L3Room_{west_room:02X}")
                reports[f"saved_0x{west_room:02x}"] = str(path)

            # Clear combat (Keese; ignore blade traps 0x49 as non-kill target).
            state_west = env.em.get_state()
            snap = read_snapshot(env.get_ram())
            combat_types = tuple(
                sorted(
                    {
                        o.type_id
                        for o in snap.objects
                        if 1 <= o.slot <= 10
                        and o.type_id not in (0, 0xFF)
                        and o.type_id not in _NON_COMBAT_TYPES
                        and o.type_id != 0x49  # blade trap
                        and o.hp > 0
                    }
                )
            )
            if combat_types:
                west_clear = _fight_clear(
                    env,
                    assist,
                    total,
                    enemy_types=combat_types,
                    max_frames=10000,
                )
            else:
                west_clear = {"ok": True, "skipped": True, "frames": 0}
            reports[f"clear_0x{west_room:02x}"] = {
                "ok": west_clear.get("ok"),
                "error": west_clear.get("error"),
                "frames": west_clear.get("frames"),
                "enemy_types": [f"0x{t:02x}" for t in combat_types],
                "final": west_clear.get("final"),
            }
            # Touch compass (center walk).
            compass0 = int(read_u8(env.get_ram(), ADDR_COMPASS))
            for _ in range(400):
                snap = read_snapshot(env.get_ram())
                if int(read_u8(env.get_ram(), ADDR_COMPASS)) > compass0:
                    break
                tx, ty = 120, 141
                if abs(snap.link_x - tx) > 4:
                    d = "RIGHT" if snap.link_x < tx else "LEFT"
                elif abs(snap.link_y - ty) > 4:
                    d = "DOWN" if snap.link_y < ty else "UP"
                else:
                    d = ("LEFT", "UP", "RIGHT", "DOWN")[_ % 4]
                env.step(nes_action(d))
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
            compass1 = int(read_u8(env.get_ram(), ADDR_COMPASS))
            reports["compass_pickup"] = {
                "before": compass0,
                "after": compass1,
                "picked": compass1 > compass0,
                "room": f"0x{west_room:02x}",
            }
            if compass1 > compass0:
                recon_notes.append(f"COMPASS inventory set in 0x{west_room:02x}")
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_0x{west_room:02x}_cleared.png")

            state_west_clr = env.em.get_state()
            west_probes = []
            for direction in ("LEFT", "UP", "DOWN", "RIGHT"):
                env.em.set_state(state_west_clr)
                obs, *_ = env.step(nes_idle_action())
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                pr = _try_exit(
                    env,
                    assist,
                    total,
                    direction,
                    tag=tag,
                    stem=f"{west_room:02x}_clr",
                )
                obs, *_ = env.step(nes_idle_action())
                save_rgb_png(obs, Path(pr["screenshot"]))
                west_probes.append(pr)
                edge = _edge(west_room, direction, pr)
                edge["phase"] = f"0x{west_room:02x}_cleared"
                reports["graph_edges"].append(edge)
                if pr["changed_room"]:
                    reports["room_notes"][pr["after"]["sc"]] = pr["after"]
            reports[f"door_probes_0x{west_room:02x}_cleared"] = [
                {
                    "direction": p["direction"],
                    "changed_room": p["changed_room"],
                    "keys_spent": p["keys_spent"],
                    "to": (
                        f"0x{p['after']['screen']:02x}"
                        if p["changed_room"]
                        else None
                    ),
                    "types": p["after"].get("type_counts"),
                    "item": p["after"].get("room_item_id"),
                    "result": p["result"],
                }
                for p in west_probes
            ]

            # Multi-hop BFS-lite from compass room: prefer unvisited, LEFT/DOWN first.
            env.em.set_state(state_west_clr)
            obs, *_ = env.step(nes_idle_action())
            hops = []
            visited: set[int] = {ROOM_L3_DARKNUTS, west_room}
            prefer_order = ("LEFT", "DOWN", "UP", "RIGHT")
            for hop_i in range(max_hops):
                snap = read_snapshot(env.get_ram())
                room_before = snap.screen
                keys_before = snap.keys
                raft_before = int(read_u8(env.get_ram(), ADDR_RAFT))
                types_now = {
                    o.type_id
                    for o in snap.objects
                    if 1 <= o.slot <= 10
                    and o.type_id not in (0, 0xFF)
                    and o.hp > 0
                    and o.type_id not in _NON_COMBAT_TYPES
                    and o.type_id != 0x49
                }
                if types_now:
                    clr = _fight_clear(
                        env,
                        assist,
                        total,
                        enemy_types=tuple(sorted(types_now)),
                        max_frames=10000,
                    )
                else:
                    clr = {"ok": True, "skipped": True}

                # Item/key roam.
                k0 = read_snapshot(env.get_ram()).keys
                bombs0 = read_snapshot(env.get_ram()).bombs
                for _ in range(250):
                    snap = read_snapshot(env.get_ram())
                    if (
                        snap.keys > k0
                        or snap.bombs > bombs0
                        or snap.room_item_id in (0, 0xFF)
                    ):
                        break
                    tx = 72 + (_ % 6) * 16
                    ty = 109 + ((_ // 6) % 5) * 16
                    if abs(snap.link_x - tx) > 4:
                        d = "RIGHT" if snap.link_x < tx else "LEFT"
                    elif abs(snap.link_y - ty) > 4:
                        d = "DOWN" if snap.link_y < ty else "UP"
                    else:
                        break
                    env.step(nes_action(d))
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])

                # Prefer doors into unvisited rooms.
                candidates: list[tuple[int, str]] = []
                for di, direction in enumerate(prefer_order):
                    candidates.append((di, direction))

                moved = False
                # First pass: unvisited only; second: any exit.
                for require_new in (True, False):
                    if moved:
                        break
                    for _, direction in candidates:
                        st = env.em.get_state()
                        pr = _try_exit(
                            env,
                            assist,
                            total,
                            direction,
                            tag=tag,
                            stem=f"whop{hop_i}",
                        )
                        if (
                            pr["changed_room"]
                            and pr["after"]["screen"] != room_before
                        ):
                            dest_r = pr["after"]["screen"]
                            if require_new and dest_r in visited:
                                env.em.set_state(st)
                                obs, *_ = env.step(nes_idle_action())
                                continue
                            raft_after = pr["after"].get("raft")
                            hops.append(
                                {
                                    "hop": hop_i,
                                    "from": f"0x{room_before:02x}",
                                    "dir": direction,
                                    "to": f"0x{dest_r:02x}",
                                    "keys_before": keys_before,
                                    "keys_after": pr["after"]["keys"],
                                    "keys_spent": pr["keys_spent"],
                                    "types": pr["after"].get("type_counts"),
                                    "item": pr["after"].get("room_item_id"),
                                    "item_name": pr["after"].get("room_item_name"),
                                    "bombs": pr["after"].get("bombs"),
                                    "raft_before": raft_before,
                                    "raft_after": raft_after,
                                    "clear": clr.get("ok"),
                                }
                            )
                            reports["graph_edges"].append(
                                {
                                    **_edge(room_before, direction, pr),
                                    "phase": f"compass_hop{hop_i}",
                                }
                            )
                            reports["room_notes"][f"0x{dest_r:02x}"] = pr["after"]
                            visited.add(dest_r)
                            obs, *_ = env.step(nes_idle_action())
                            save_rgb_png(
                                obs,
                                RECORDINGS_DIR
                                / f"{tag}_whop{hop_i}_0x{dest_r:02x}.png",
                            )
                            if raft_after and not raft_before:
                                recon_notes.append(
                                    f"RAFT acquired at hop {hop_i} room 0x{dest_r:02x}"
                                )
                            moved = True
                            break
                        env.em.set_state(st)
                        obs, *_ = env.step(nes_idle_action())

                if not moved:
                    if try_bombs and poke_bombs:
                        _poke_bombs(env, poke_bombs)
                        recon_notes.append(
                            f"compass_hop{hop_i} stuck — bomb recon "
                            f"0x{room_before:02x}"
                        )
                        st_bomb = env.em.get_state()
                        for face, bx, by in BOMB_STANDS[:6]:
                            env.em.set_state(st_bomb)
                            if read_snapshot(env.get_ram()).bombs <= 0:
                                _poke_bombs(env, poke_bombs)
                            br = _try_bomb_stand(
                                env,
                                assist,
                                total,
                                face,
                                bx,
                                by,
                                tag=tag,
                                stem=f"whop{hop_i}",
                            )
                            if br["changed_room"]:
                                dest_r = br["after"]["screen"]
                                hops.append(
                                    {
                                        "hop": hop_i,
                                        "from": f"0x{room_before:02x}",
                                        "dir": f"BOMB_{face}",
                                        "to": f"0x{dest_r:02x}",
                                        "stand": br["stand"],
                                        "types": br["after"].get("type_counts"),
                                        "item": br["after"].get("room_item_id"),
                                    }
                                )
                                reports["graph_edges"].append(
                                    {
                                        "from": f"0x{room_before:02x}",
                                        "dir": f"BOMB_{face}",
                                        "to": f"0x{dest_r:02x}",
                                        "result": "bomb_open",
                                        "stand": br["stand"],
                                        "phase": f"compass_hop{hop_i}_bomb",
                                        "after_types": br["after"].get(
                                            "type_counts"
                                        ),
                                    }
                                )
                                reports["room_notes"][f"0x{dest_r:02x}"] = br[
                                    "after"
                                ]
                                visited.add(dest_r)
                                moved = True
                                break
                    if not moved:
                        hops.append(
                            {
                                "hop": hop_i,
                                "from": f"0x{room_before:02x}",
                                "stuck": True,
                                "clear": clr,
                                "visited": [f"0x{r:02x}" for r in sorted(visited)],
                                "final": _room_fields(
                                    read_snapshot(env.get_ram()),
                                    env.get_ram(),
                                ),
                            }
                        )
                        break
                if int(read_u8(env.get_ram(), ADDR_RAFT)):
                    recon_notes.append("raft inventory bit set — stop hops")
                    break
            reports["compass_path_hops"] = hops
            reports["compass_path_visited"] = [
                f"0x{r:02x}" for r in sorted(visited)
            ]

        # --- Phase E: bomb RIGHT from 0x5b (boss shortcut, recon poke OK) ---
        if try_bombs:
            env, obs = _reload(env, start_state, assist, total)
            _idle(env, assist, total, 20)
            if poke_bombs is not None:
                note = _poke_bombs(env, poke_bombs)
                recon_notes.append(f"RECON bomb poke for 0x5b RIGHT: {note}")
            elif read_snapshot(env.get_ram()).bombs <= 0:
                note = _poke_bombs(env, 8)
                recon_notes.append(
                    f"RECON bomb poke (default 8) for boss-shortcut: {note}"
                )
            # Prefer clear first so bomb walls more likely to register.
            if not skip_clear:
                _fight_clear(
                    env,
                    assist,
                    total,
                    enemy_types=(DARKNUT_OBJECT_TYPE,),
                    max_frames=12000,
                )
            st_b = env.em.get_state()
            bomb_hits = []
            for face, bx, by in BOMB_STANDS:
                env.em.set_state(st_b)
                if read_snapshot(env.get_ram()).bombs <= 0 and poke_bombs:
                    _poke_bombs(env, poke_bombs or 8)
                elif read_snapshot(env.get_ram()).bombs <= 0:
                    _poke_bombs(env, 8)
                br = _try_bomb_stand(
                    env, assist, total, face, bx, by, tag=tag, stem="5b_boss"
                )
                obs, *_ = env.step(nes_idle_action())
                save_rgb_png(obs, Path(br["screenshot"]))
                bomb_hits.append(
                    {
                        "face": br["face"],
                        "stand": br["stand"],
                        "result": br["result"],
                        "to": (
                            f"0x{br['after']['screen']:02x}"
                            if br["changed_room"]
                            else None
                        ),
                        "bombs_spent": br["bombs_spent"],
                        "types": br["after"].get("type_counts"),
                        "item": br["after"].get("room_item_id"),
                    }
                )
                if br["changed_room"]:
                    reports["graph_edges"].append(
                        {
                            "from": "0x5b",
                            "dir": f"BOMB_{face}",
                            "to": f"0x{br['after']['screen']:02x}",
                            "keys_spent": 0,
                            "result": "bomb_open",
                            "stand": br["stand"],
                            "after_types": br["after"].get("type_counts"),
                            "after_item": br["after"].get("room_item_id"),
                            "phase": "5b_bomb_boss_shortcut",
                        }
                    )
                    reports["room_notes"][
                        f"0x{br['after']['screen']:02x}"
                    ] = br["after"]
            reports["bomb_probes_5b"] = bomb_hits

        # Dedupe graph edges
        seen: set = set()
        deduped = []
        for e in reports["graph_edges"]:
            key = (
                e.get("from"),
                e.get("dir"),
                e.get("to"),
                e.get("result"),
                e.get("phase"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)
        reports["graph_edges"] = deduped

        # Summary live graph for docs (unique from→dir→to).
        live_summary = []
        seen_sum: set = set()
        for e in deduped:
            if e.get("to") and e.get("result") in ("room_change", "bomb_open"):
                line = (
                    f"{e['from']} --{e['dir']}--> {e['to']}"
                    + (
                        f" (keys_spent={e.get('keys_spent')})"
                        if e.get("keys_spent")
                        else ""
                    )
                    + (
                        f" types={e.get('after_types')}"
                        if e.get("after_types")
                        else ""
                    )
                )
                sk = (e["from"], e["dir"], e["to"])
                if sk in seen_sum:
                    continue
                seen_sum.add(sk)
                live_summary.append(line)
        reports["live_graph_summary"] = live_summary
        reports["ok"] = True
        reports["frames"] = total[0]
        reports["assist"] = assist.report() if assist else None
        reports["final"] = _room_fields(
            read_snapshot(env.get_ram()), env.get_ram()
        )

        out = RECORDINGS_DIR / f"{tag}_recon.json"
        write_json_report(out, reports)
        reports["report_path"] = str(out)
        return reports
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level3Darknuts")
    p.add_argument(
        "--infinite-life",
        action="store_true",
        default=True,
        help="Survival assist (default on for this recon probe)",
    )
    p.add_argument(
        "--no-infinite-life",
        action="store_true",
        help="Disable Survival assist (Clean attempt — may die to Darknuts)",
    )
    p.add_argument(
        "--skip-clear",
        action="store_true",
        help="Skip Darknut combat; door probes only",
    )
    p.add_argument(
        "--try-bombs",
        action="store_true",
        help="Probe bomb stands on 0x5b (and stuck hops)",
    )
    p.add_argument(
        "--poke-bombs",
        type=int,
        default=None,
        help="RECON-ONLY set bomb count (document in report; not Clean)",
    )
    p.add_argument("--max-hops", type=int, default=6)
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--tag", default="l3_past_5b")
    args = p.parse_args(argv)

    infinite = not args.no_infinite_life
    report = run_probe(
        start_state=args.from_state,
        infinite_life=infinite,
        skip_clear=args.skip_clear,
        try_bombs=args.try_bombs,
        poke_bombs=args.poke_bombs,
        max_hops=args.max_hops,
        save_checkpoints=args.save_state,
        tag=args.tag,
    )
    print(f"ok={report.get('ok')} track={report.get('track')}")
    if report.get("error"):
        print(f"error={report['error']}")
    print("entry:", report.get("entry", {}).get("sc"), report.get("entry", {}).get("type_counts"))
    if report.get("darknut_clear"):
        print(
            f"darknut_clear ok={report['darknut_clear'].get('ok')} "
            f"frames={report['darknut_clear'].get('frames')} "
            f"err={report['darknut_clear'].get('error')}"
        )
    print("edges:")
    for e in report.get("graph_edges") or []:
        print(
            f"  [{e.get('phase')}] {e.get('from')} -{e.get('dir')}-> {e.get('to')} "
            f"keys_spent={e.get('keys_spent')} result={e.get('result')} "
            f"types={e.get('after_types')} item={e.get('after_item')}"
        )
    if report.get("live_graph_summary"):
        print("live_graph:")
        for line in report["live_graph_summary"]:
            print(f"  {line}")
    if report.get("free_explore_hops"):
        print("hops:")
        for h in report["free_explore_hops"]:
            print(f"  {h}")
    if report.get("recon_notes"):
        print(f"notes={report['recon_notes']}")
    if report.get("trap_notes"):
        print(f"traps={report['trap_notes']}")
    print(f"wrote {report.get('report_path')}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
