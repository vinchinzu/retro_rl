"""Clear L5 0x55 5x Zol from Level5Cleared56, dump doors, probe every exit.

Walk LEFT 0x56 -> 0x55. Wait for play mode 5 (not settle mode 4).
GenericDungeonRoomController + AliveRule.TYPE_AND_HP + gel type_only
(0x57 pattern). No pokes, no candle, no bomb walls, no east67, not Clean.
Level5Cleared55 only if 5 Zols dead AND a real next dest exists.
Bombs<6: do not enter a Dodongo 0x31 room (need 6 eats).
"""
from __future__ import annotations

from pathlib import Path

from retro_harness.env import make_env, reset_obs, save_state, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
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
from zelda_i.dungeon_ids import GEL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, ZOL_OBJECT_TYPE, object_name
from zelda_i.dungeon_lab import _drive_exit
from zelda_i.dungeon_ops import exit_door, goto, idle
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

STATE = "Level5Cleared56"
ROOM_56 = 0x56
ROOM_55 = 0x55
DODONGO_TYPES = (0x31, 0x32)
ZOL_TYPES = (ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE)
GIBDO_TYPE = 0x30

_ZOL_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
    (96, 117),
    (144, 157),
    (80, 157),
    (160, 125),
)

_GIBDO_PATROL: tuple[tuple[int, int], ...] = (
    (64, 117),
    (112, 117),
    (160, 117),
    (192, 117),
    (192, 149),
    (160, 149),
    (112, 149),
    (64, 149),
    (64, 181),
    (112, 181),
    (160, 181),
    (192, 181),
)

EXIT_ROUTES: dict[str, DoorRoute] = {
    "UP": DoorRoute("UP", ((120, 141), (120, 93))),
    "DOWN": DoorRoute("DOWN", ((120, 141), (120, 205))),
    "LEFT": DoorRoute("LEFT", ((120, 141), (32, 141))),
    "RIGHT": DoorRoute("RIGHT", ((120, 141), (208, 141))),
}

DIR_PRIORITY = ("UP", "LEFT", "DOWN", "RIGHT")


def decode_doors(mask: int) -> dict:
    value = int(mask) & 0x0F
    return {
        "raw": value,
        "raw_hex": f"0x{value:02x}",
        "east": bool(value & DoorDir.RIGHT),
        "west": bool(value & DoorDir.LEFT),
        "south": bool(value & DoorDir.DOWN),
        "north": bool(value & DoorDir.UP),
        "open": sorted(d.name for d in dirs_from_mask(value)),
    }


def inv_block(ram) -> dict:
    snap = read_snapshot(ram)
    return {
        "selected_item_0x0656": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "bombs": int(snap.bombs),
        "keys": int(snap.keys),
    }


def dump_live(snap: ZeldaSnapshot, ram) -> dict:
    compact = compact_snapshot(snap)
    compact["doors"] = decode_doors(snap.cur_opened_doors)
    compact["doorway_mask"] = decode_doors(snap.open_doorway_mask)
    compact["room_hex"] = f"0x{snap.screen:02x}"
    compact["next_room_hex"] = f"0x{snap.next_screen:02x}"
    compact["inventory"] = inv_block(ram)
    compact["objects"] = [
        {
            "slot": obj.slot,
            "type_id": obj.type_id,
            "type_hex": f"0x{obj.type_id:02x}",
            "type_name": object_name(obj.type_id),
            "x": obj.x,
            "y": obj.y,
            "hp": obj.hp,
            "state": obj.state,
            "facing": obj.facing,
        }
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF)
    ]
    return compact


def live_zols(snap: ZeldaSnapshot) -> list:
    if snap.mode != PLAY_MODE:
        return []
    out = []
    for obj in snap.objects:
        if not (1 <= obj.slot <= 12) or obj.type_id not in ZOL_TYPES:
            continue
        if obj.type_id in (GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE) or obj.hp > 0:
            out.append(obj)
    return out


def live_types(snap: ZeldaSnapshot, types: tuple[int, ...], *, hp: bool) -> list:
    if snap.mode != PLAY_MODE:
        return []
    out = []
    for obj in snap.objects:
        if not (1 <= obj.slot <= 12) or obj.type_id not in types:
            continue
        if (not hp) or obj.hp > 0:
            out.append(obj)
    return out


def is_dodongo_room(dump: dict) -> bool:
    for obj in dump.get("objects") or []:
        tid = obj.get("type_id")
        if tid in DODONGO_TYPES and (obj.get("hp") or 0) > 0:
            return True
    objs = dump.get("objects")
    if isinstance(objs, dict):
        for key in objs:
            if key.lower() in ("0x31", "0x32"):
                return True
    return False


def classify_combat(dump: dict) -> str:
    objects = dump.get("objects") or []
    if isinstance(objects, dict):
        keys = {k.lower() for k in objects}
        if "0x31" in keys or "0x32" in keys:
            return "dodongo"
        if "0x13" in keys or "0x14" in keys or "0x15" in keys:
            return "zols"
        if "0x30" in keys:
            return "gibdos"
        combat = [k for k in keys if k not in ("0x40", "0x4e", "0x4f", "0x55", "0x49", "0x2b")]
        return "combat" if combat else "empty"
    types = []
    for obj in objects:
        tid = obj.get("type_id")
        hp = obj.get("hp") or 0
        if tid in (0, 0xFF, None):
            continue
        if tid in (0x40, 0x49, 0x4E, 0x4F, 0x55, 0x2B, 0x5A, 0x60):
            continue
        if hp > 0 or tid in ZOL_TYPES or tid == 0x1B:
            types.append(tid)
    if any(t in DODONGO_TYPES for t in types):
        return "dodongo"
    if any(t in ZOL_TYPES for t in types):
        return "zols"
    if GIBDO_TYPE in types:
        return "gibdos"
    if types:
        return "combat"
    return "empty"


def open_env(state: str = STATE):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def open_from_bytes(state_data: bytes):
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    env.em.set_state(state_data)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def step(env, assist, total, action):
    obs, *_ = env.step(action)
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    return obs



def walk_west_55(env, assist, total) -> dict:
    """y=141 channel then hold LEFT. Mid-room / y=133 sticks at the 0x56 jamb."""
    from retro_harness.nes import nes_action
    snap0 = read_snapshot(env.get_ram())
    room0 = snap0.screen
    before = (snap0.screen, snap0.mode, snap0.link_x, snap0.link_y)
    log = []
    changed = False
    for i in range(900):
        snap = read_snapshot(env.get_ram())
        if i % 40 == 0:
            log.append({"f": total[0], "sc": f"0x{snap.screen:02x}", "mode": snap.mode,
                        "xy": [snap.link_x, snap.link_y]})
        if snap.mode == PLAY_MODE and snap.screen == ROOM_55 and not snap.transitioning:
            changed = True
            log.append({"event": "left_56", "f": total[0], "mode": snap.mode,
                        "sc": f"0x{snap.screen:02x}", "xy": [snap.link_x, snap.link_y]})
            break
        if snap.mode != PLAY_MODE or snap.transitioning:
            step(env, assist, total, nes_action("LEFT"))
            continue
        # Correct Y while still in the room floor (x>40). At the jamb, only LEFT.
        if abs(snap.link_y - 141) > 2 and snap.link_x > 40:
            btn = "DOWN" if snap.link_y < 141 else "UP"
            step(env, assist, total, nes_action(btn))
        else:
            step(env, assist, total, nes_action("LEFT"))
    else:
        for _ in range(80):
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_55:
                changed = True
                break
            step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 20)
    snap = read_snapshot(env.get_ram())
    return {
        "before": before,
        "after": (snap.screen, snap.mode, snap.link_x, snap.link_y),
        "changed_room": changed or snap.screen == ROOM_55,
        "result": "room_change" if (changed or snap.screen == ROOM_55) else "blocked",
        "log": log,
    }


def wait_play(env, assist, total, room: int, *, max_f: int = 300) -> bool:
    """Idle until play mode 5 in *room* (not settle mode 4)."""
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if (
            snap.level == 5
            and snap.screen == room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            idle(env, assist, total, 16)
            snap = read_snapshot(env.get_ram())
            return (
                snap.screen == room
                and snap.mode == PLAY_MODE
                and not snap.transitioning
            )
        step(env, assist, total, nes_idle_action())
    return False


def zol_spec(room_id: int) -> DungeonRoomSpec:
    return DungeonRoomSpec(
        spec_id=f"level5_room{room_id:02x}_zols",
        source_room=room_id,
        room_id=room_id,
        entry=DoorRoute("LEFT", ((208, 141),)),
        enemy_types=ZOL_TYPES,
        expected_enemy_count=5,
        alive_rule=AliveRule.TYPE_AND_HP,
        type_only_enemy_types=(GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE),
        object_slot_max=12,
        combat=CombatTuning(
            patrol=_ZOL_PATROL,
            engage_distance=56,
            attack_phase=4,
            engage_attack_period=6,
            engage_attack_hold=3,
            patrol_attack_period=10,
            patrol_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        max_frames=16000,
        level=5,
    )


def gibdo_spec(room_id: int) -> DungeonRoomSpec:
    return DungeonRoomSpec(
        spec_id=f"level5_room{room_id:02x}_gibdos",
        source_room=room_id,
        room_id=room_id,
        entry=DoorRoute("DOWN", ((120, 93),)),
        enemy_types=(GIBDO_TYPE,),
        expected_enemy_count=5,
        alive_rule=AliveRule.TYPE_AND_HP,
        combat=CombatTuning(
            patrol=_GIBDO_PATROL,
            engage_distance=56,
            engage_attack_period=6,
            engage_attack_hold=3,
            patrol_attack_period=10,
            patrol_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=8),
        max_frames=14000,
        level=5,
    )


def fight_room(env, assist, total, spec: DungeonRoomSpec) -> dict:
    ctl = GenericDungeonRoomController(spec)
    obs = None
    start_live = None
    for _ in range(spec.max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == spec.room_id:
            live = spec.live_enemies(snap)
            if start_live is None:
                start_live = live
        action = ctl.step(snap)
        obs = step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    live = spec.live_enemies(snap) if snap.mode == PLAY_MODE else ()
    start_n = 0 if start_live is None else len(start_live)
    return {
        "obs": obs,
        "ok": bool(ctl.success) and snap.screen == spec.room_id and not live,
        "frames": ctl.frames,
        "start_n": start_n,
        "end_n": len(live),
        "kills": start_n - len(live),
        "controller": ctl.report(),
        "alive_rule": spec.alive_rule.value,
        "type_only": list(spec.type_only_enemy_types),
    }


def hunt_key(env, assist, total, keys0: int) -> dict:
    waypoints = (
        (120, 141),
        (96, 117),
        (144, 165),
        (80, 141),
        (160, 141),
        (120, 157),
        (120, 125),
        (64, 141),
        (176, 141),
    )
    for tx, ty in waypoints:
        goto(env, assist, total, tx, ty, tol=4, max_f=180)
        idle(env, assist, total, 8)
        snap = read_snapshot(env.get_ram())
        if snap.keys > keys0:
            return {"picked": True, "keys": snap.keys, "xy": [snap.link_x, snap.link_y]}
    snap = read_snapshot(env.get_ram())
    return {"picked": snap.keys > keys0, "keys": snap.keys, "xy": [snap.link_x, snap.link_y]}


def probe_all_exits(state_data: bytes, room_id: int, out_dir: Path) -> list[dict]:
    results = []
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        route = EXIT_ROUTES[direction]
        png = out_dir / f"l5_{room_id:02x}_exit_{direction.lower()}.png"
        raw = _drive_exit(
            state_data,
            spec_room=room_id,
            route=route,
            screenshot_path=png,
            max_frames=900,
        )
        dest = None
        sealed = not raw.get("success")
        if raw.get("success"):
            dest = raw.get("room_hex") or f"0x{raw.get('room', 0):02x}"
        results.append(
            {
                "direction": direction,
                "success": bool(raw.get("success")),
                "sealed": sealed,
                "dest_room": dest if not sealed else None,
                "dest_room_id": raw.get("room") if raw.get("success") else None,
                "frames": raw.get("frames"),
                "objects": raw.get("objects"),
                "room_item_id": raw.get("room_item_id"),
                "room_item_name": raw.get("room_item_name"),
                "x": raw.get("x"),
                "y": raw.get("y"),
                "mode": raw.get("mode"),
                "screenshot": raw.get("screenshot"),
            }
        )
    return results


def pick_next(probes: list[dict], *, bombs: int, came_from: str = "RIGHT") -> dict | None:
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    skip_back = opposite.get(came_from)
    real = [p for p in probes if p.get("success") and p.get("dest_room_id") not in (None, ROOM_55)]
    if not real:
        return None
    scored = []
    for p in real:
        room = p.get("dest_room_id")
        objs = p.get("objects") or {}
        dodo = False
        if isinstance(objs, dict):
            dodo = any(k.lower() in ("0x31", "0x32") for k in objs)
        if dodo and bombs < 6:
            p = dict(p)
            p["skip_reason"] = f"dodongo_bombs={bombs}<6"
            scored.append((9, DIR_PRIORITY.index(p["direction"]) if p["direction"] in DIR_PRIORITY else 9, p))
            continue
        back = 1 if p["direction"] == skip_back or room == ROOM_56 else 0
        pri = DIR_PRIORITY.index(p["direction"]) if p["direction"] in DIR_PRIORITY else 8
        scored.append((back, pri, p))
    scored.sort(key=lambda t: (t[0], t[1]))
    best = scored[0][2]
    if str(best.get("skip_reason") or "").startswith("dodongo"):
        # still a real dest for checkpoint, but do not enter
        best = dict(best)
        best["enter"] = False
        return best
    best = dict(best)
    best["enter"] = True
    return best


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        f"PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_55_clear.py  # {STATE}, infinite-life"
    ]
    env = None
    env, assist, obs = open_env()
    total = [1]
    checkpoints: list[str] = []
    try:
        idle(env, assist, total, 20)
        ram = env.get_ram()
        start_snap = read_snapshot(ram)
        start_dump = dump_live(start_snap, ram)
        start_room = start_snap.screen

        walked = False
        hop = None
        if start_snap.screen != ROOM_55:
            hop = walk_west_55(env, assist, total)
            walked = bool(hop.get("changed_room"))
            print("WEST_HOP", hop, flush=True)
        ready = wait_play(env, assist, total, ROOM_55, max_f=360)
        print(
            "READY",
            ready,
            "room",
            hex(read_snapshot(env.get_ram()).screen),
            "mode",
            read_snapshot(env.get_ram()).mode,
            flush=True,
        )
        ram = env.get_ram()
        arrive_snap = read_snapshot(ram)
        arrive_dump = dump_live(arrive_snap, ram)
        arrive_png = RECORDINGS_DIR / "l5_55_arrive.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, arrive_png)

        if not ready or arrive_snap.screen != ROOM_55 or arrive_snap.mode != PLAY_MODE:
            report = {
                "ok": False,
                "reason": "never_play_mode_5_in_0x55",
                "status_claim": None,
                "pokes": False,
                "commands": commands,
                "start": start_dump,
                "arrive": arrive_dump,
                "walked": walked,
                "hop": hop,
                "ready": ready,
                "checkpoint": None,
                "whistle_0x065C": arrive_dump.get("inventory", {}).get("whistle_0x065C"),
                "screenshot": str(arrive_png.resolve()),
            }
            write_json_report(RECORDINGS_DIR / "l5_55_clear.json", report)
            return report

        bombs_in = int(arrive_snap.bombs)
        keys_in = int(arrive_snap.keys)
        zols_arrive = live_zols(arrive_snap)

        spec = zol_spec(ROOM_55)
        fight = fight_room(env, assist, total, spec)
        if fight.get("obs") is not None:
            obs = fight["obs"]
        idle(env, assist, total, 30)
        extra = 0
        while extra < 90:
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_55:
                break
            obs = step(env, assist, total, nes_idle_action())
            extra += 1
        idle(env, assist, total, 20)

        ram = env.get_ram()
        mid_snap = read_snapshot(ram)
        mid_dump = dump_live(mid_snap, ram)
        dead = (
            mid_snap.screen == ROOM_55
            and mid_snap.mode == PLAY_MODE
            and not live_zols(mid_snap)
        )
        key_hunt = None
        if dead:
            key_hunt = hunt_key(env, assist, total, keys_in)
            idle(env, assist, total, 12)
            ram = env.get_ram()
            mid_snap = read_snapshot(ram)
            mid_dump = dump_live(mid_snap, ram)
            dead = (
                mid_snap.screen == ROOM_55
                and mid_snap.mode == PLAY_MODE
                and not live_zols(mid_snap)
            )

        bombs_out = int(mid_snap.bombs)
        png = RECORDINGS_DIR / "l5_55_clear.png"
        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
        save_rgb_png(obs, png)

        doors_end = decode_doors(mid_snap.cur_opened_doors)
        mask_end = decode_doors(mid_snap.open_doorway_mask)

        state_bytes = env.em.get_state()
        print(
            "MID_CLEAR dead",
            dead,
            "frames",
            fight["frames"],
            "kills",
            fight["kills"],
            "end_n",
            fight["end_n"],
            "doors",
            doors_end,
            "mask",
            mask_end,
            flush=True,
        )
        env.close()
        env = None

        probes = []
        if dead:
            probes = probe_all_exits(state_bytes, ROOM_55, RECORDINGS_DIR)
        real_dests = [
            p for p in probes if p.get("success") and p.get("dest_room_id") not in (None, ROOM_55)
        ]
        next_pick = pick_next(probes, bombs=bombs_out, came_from="RIGHT") if probes else None
        has_real_dest = bool(real_dests)
        print("PROBES", [(p.get("direction"), p.get("dest_room") or "sealed") for p in probes], flush=True)

        saved = None
        if dead and has_real_dest:
            path = write_state_bytes(
                state_path(GAME_DIR, GAME, "Level5Cleared55"),
                state_bytes,
            )
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state"
                ),
                request={
                    "segment": "Level5Cleared55",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                    "alive_rule": "hp",
                    "gel_type_only": True,
                },
                selected_trial={
                    "success": True,
                    "frames": fight["frames"],
                    "room": ROOM_55,
                    "live_zols": 0,
                    "doors_after": int(mid_snap.cur_opened_doors) & 0x0F,
                    "doorway_mask": int(mid_snap.open_doorway_mask) & 0x0F,
                    "next_dest": None if next_pick is None else next_pick.get("dest_room"),
                    "bombs": bombs_out,
                    "keys": mid_snap.keys,
                },
                natural_entry=False,
            )
            saved = "Level5Cleared55"
            checkpoints.append("Level5Cleared55")

        exits_report = {
            "from_room": "0x55",
            "doors": doors_end,
            "doorway_mask": mask_end,
            "probes": probes,
            "real_dests": [
                {"direction": p["direction"], "dest": p["dest_room"]} for p in real_dests
            ],
            "picked": None
            if next_pick is None
            else {
                "direction": next_pick.get("direction"),
                "dest": next_pick.get("dest_room"),
                "enter": next_pick.get("enter"),
                "skip_reason": next_pick.get("skip_reason"),
            },
            "status_claim": None,
            "pokes": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_55_exits.json", exits_report)

        next_room = None
        next_clear = None
        further = None
        if (
            dead
            and next_pick is not None
            and next_pick.get("enter")
            and next_pick.get("direction")
        ):
            env, assist, obs = open_from_bytes(state_bytes)
            hop = exit_door(env, assist, total, next_pick["direction"])
            dest_id = hop.get("after", {}).get("screen")
            wait_play(env, assist, total, dest_id if dest_id is not None else 0, max_f=240)
            idle(env, assist, total, 20)
            ram = env.get_ram()
            dest_snap = read_snapshot(ram)
            dest_dump = dump_live(dest_snap, ram)
            dest_png = RECORDINGS_DIR / f"l5_{dest_snap.screen:02x}_recon.png"
            obs, *_ = env.step(nes_idle_action())
            total[0] += 1
            assist.apply_env(env, frame=total[0])
            save_rgb_png(obs, dest_png)
            kind = classify_combat(dest_dump)
            next_room = {
                "ok": hop.get("changed_room"),
                "via": f"0x55 {next_pick['direction']}",
                "kind": kind,
                "dump": dest_dump,
                "screenshot": str(dest_png.resolve()),
                "status_claim": None,
                "pokes": False,
            }
            write_json_report(
                RECORDINGS_DIR / f"l5_{dest_snap.screen:02x}_recon.json",
                next_room,
            )

            if (
                hop.get("changed_room")
                and dest_snap.mode == PLAY_MODE
                and kind in ("zols", "gibdos", "combat")
                and kind != "dodongo"
            ):
                if kind == "zols":
                    nspec = zol_spec(dest_snap.screen)
                    nspec = DungeonRoomSpec(
                        spec_id=nspec.spec_id,
                        source_room=dest_snap.screen,
                        room_id=dest_snap.screen,
                        entry=DoorRoute(next_pick["direction"], ((120, 141),)),
                        enemy_types=nspec.enemy_types,
                        expected_enemy_count=5,
                        alive_rule=nspec.alive_rule,
                        type_only_enemy_types=nspec.type_only_enemy_types,
                        combat=nspec.combat,
                        reward=nspec.reward,
                        max_frames=16000,
                        level=5,
                    )
                elif kind == "gibdos":
                    nspec = gibdo_spec(dest_snap.screen)
                else:
                    types = tuple(
                        sorted(
                            {
                                o["type_id"]
                                for o in dest_dump.get("objects") or []
                                if o.get("type_id")
                                not in (0, 0x40, 0x49, 0x4E, 0x4F, 0x55, 0x2B)
                            }
                        )
                    )
                    nspec = DungeonRoomSpec(
                        spec_id=f"level5_room{dest_snap.screen:02x}_combat",
                        source_room=dest_snap.screen,
                        room_id=dest_snap.screen,
                        entry=DoorRoute(next_pick["direction"], ((120, 141),)),
                        enemy_types=types or (0x13,),
                        expected_enemy_count=max(1, dest_dump.get("room_obj_count") or 1),
                        alive_rule=AliveRule.TYPE_AND_HP,
                        combat=CombatTuning(
                            patrol=_ZOL_PATROL,
                            engage_distance=56,
                            attack_phase=4,
                            engage_attack_period=6,
                            engage_attack_hold=3,
                            patrol_attack_period=10,
                            patrol_attack_hold=3,
                        ),
                        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
                        max_frames=14000,
                        level=5,
                    )
                next_clear = fight_room(env, assist, total, nspec)
                idle(env, assist, total, 24)
                ram = env.get_ram()
                after_snap = read_snapshot(ram)
                after_dump = dump_live(after_snap, ram)
                next_clear = {
                    k: v for k, v in next_clear.items() if k != "obs"
                }
                next_clear["dump"] = after_dump
                next_clear["kind"] = kind

                # one more obvious room if not Dodongo
                doors = after_dump.get("doors") or {}
                mask = after_dump.get("doorway_mask") or {}
                came = {
                    "UP": "south",
                    "DOWN": "north",
                    "LEFT": "east",
                    "RIGHT": "west",
                }.get(next_pick["direction"])
                cands = []
                for name, dname in (
                    ("north", "UP"),
                    ("west", "LEFT"),
                    ("south", "DOWN"),
                    ("east", "RIGHT"),
                ):
                    if name == came:
                        continue
                    if doors.get(name) or mask.get(name):
                        cands.append(dname)
                if cands and after_snap.bombs < 6:
                    more_bytes = env.em.get_state()
                    env.close()
                    env = None
                    peek = _drive_exit(
                        more_bytes,
                        spec_room=after_snap.screen,
                        route=EXIT_ROUTES[cands[0]],
                        screenshot_path=RECORDINGS_DIR
                        / f"l5_{after_snap.screen:02x}_peek_{cands[0].lower()}.png",
                    )
                    peek_dodo = False
                    pobjs = peek.get("objects") or {}
                    if isinstance(pobjs, dict):
                        peek_dodo = any(k.lower() in ("0x31", "0x32") for k in pobjs)
                    if peek.get("success") and not peek_dodo:
                        env, assist, obs = open_from_bytes(more_bytes)
                        hop2 = exit_door(env, assist, total, cands[0])
                        idle(env, assist, total, 40)
                        wait_play(
                            env,
                            assist,
                            total,
                            read_snapshot(env.get_ram()).screen,
                            max_f=180,
                        )
                        ram = env.get_ram()
                        fur_snap = read_snapshot(ram)
                        fur_dump = dump_live(fur_snap, ram)
                        fur_png = RECORDINGS_DIR / f"l5_{fur_snap.screen:02x}_recon.png"
                        obs, *_ = env.step(nes_idle_action())
                        save_rgb_png(obs, fur_png)
                        further = {
                            "via": f"0x{after_snap.screen:02x} {cands[0]}",
                            "kind": classify_combat(fur_dump),
                            "dump": fur_dump,
                            "screenshot": str(fur_png.resolve()),
                            "peek": {
                                "success": peek.get("success"),
                                "room": peek.get("room_hex"),
                                "objects": peek.get("objects"),
                            },
                        }
                        write_json_report(
                            RECORDINGS_DIR / f"l5_{fur_snap.screen:02x}_recon.json",
                            further,
                        )
                    elif peek_dodo:
                        further = {
                            "stopped": True,
                            "reason": "dodongo_ahead_bombs<6",
                            "direction": cands[0],
                            "peek": {
                                "success": peek.get("success"),
                                "room": peek.get("room_hex"),
                                "objects": peek.get("objects"),
                            },
                        }

        if env is not None:
            ram = env.get_ram()
            end_snap = read_snapshot(ram)
            end_dump = dump_live(end_snap, ram)
            whistle = int(read_u8(ram, ADDR_WHISTLE))
        else:
            end_dump = mid_dump
            whistle = (mid_dump.get("inventory") or {}).get("whistle_0x065C", 0)

        report = {
            "ok": dead and has_real_dest,
            "status_claim": None,
            "from_state": STATE,
            "pokes": False,
            "commands": commands,
            "walked_left": walked,
            "start_room": f"0x{start_room:02x}",
            "arrive": {
                "room": arrive_dump.get("room_hex"),
                "mode": arrive_dump.get("mode"),
                "mode_name": arrive_dump.get("mode_name"),
                "xy": [arrive_dump.get("x"), arrive_dump.get("y")],
                "objects": arrive_dump.get("objects"),
                "doors": arrive_dump.get("doors"),
                "doorway_mask": arrive_dump.get("doorway_mask"),
                "bombs": bombs_in,
                "keys": keys_in,
                "zols": len(zols_arrive),
            },
            "clear": {
                "frames": fight["frames"],
                "kills": fight["kills"],
                "start_n": fight["start_n"],
                "end_n": fight["end_n"],
                "alive_rule": fight["alive_rule"],
                "type_only": fight["type_only"],
                "controller": fight["controller"],
                "bombs_in": bombs_in,
                "bombs_out": bombs_out,
                "dead": dead,
                "key_hunt": key_hunt,
            },
            "doors_end": doors_end,
            "doorway_mask_end": mask_end,
            "exits": probes,
            "next_pick": None
            if next_pick is None
            else {
                "direction": next_pick.get("direction"),
                "dest": next_pick.get("dest_room"),
                "enter": next_pick.get("enter"),
                "skip_reason": next_pick.get("skip_reason"),
            },
            "next_room": None
            if next_room is None
            else {
                "ok": next_room.get("ok"),
                "via": next_room.get("via"),
                "kind": next_room.get("kind"),
                "room": (next_room.get("dump") or {}).get("room_hex"),
                "objects": (next_room.get("dump") or {}).get("objects"),
                "doors": (next_room.get("dump") or {}).get("doors"),
                "item": (next_room.get("dump") or {}).get("room_item_name"),
            },
            "next_clear": next_clear,
            "further": None
            if further is None
            else {
                k: v for k, v in further.items() if k != "dump"
            }
            | (
                {
                    "room": further.get("dump", {}).get("room_hex"),
                    "objects": further.get("dump", {}).get("objects"),
                }
                if further.get("dump")
                else {}
            ),
            "checkpoint": saved,
            "checkpoint_reason": (
                "5 Zols dead in play mode 5 and a real next dest exists"
                if saved
                else (
                    "not saved: "
                    + (
                        "enemies still alive"
                        if not dead
                        else "no real next dest"
                    )
                )
            ),
            "checkpoints_written": checkpoints,
            "whistle_0x065C": whistle,
            "frames_total": total[0],
            "screenshot": str(png.resolve()),
            "dump": mid_dump,
            "end_dump": end_dump,
        }
        write_json_report(RECORDINGS_DIR / "l5_55_clear.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("WALKED", r.get("walked_left"), "START", r.get("start_room"))
    a = r.get("arrive") or {}
    print("ARRIVE", a.get("room"), "mode", a.get("mode"), a.get("mode_name"), "xy", a.get("xy"), "zols", a.get("zols"))
    print("ARRIVE_OBJECTS", a.get("objects"))
    c = r.get("clear") or {}
    print(
        "CLEAR frames",
        c.get("frames"),
        "kills",
        c.get("kills"),
        "start_n",
        c.get("start_n"),
        "end_n",
        c.get("end_n"),
        "dead",
        c.get("dead"),
        "bombs",
        c.get("bombs_in"),
        "->",
        c.get("bombs_out"),
    )
    print("CLEAR_CTRL", c.get("controller"))
    print("KEY", c.get("key_hunt"))
    print("DOORS_END", r.get("doors_end"))
    print("DOORWAY_END", r.get("doorway_mask_end"))
    print("EXITS")
    for p in r.get("exits") or []:
        print(
            " ",
            p.get("direction"),
            "dest" if p.get("success") else "sealed",
            p.get("dest_room"),
            p.get("objects"),
        )
    print("NEXT_PICK", r.get("next_pick"))
    print("NEXT_ROOM", r.get("next_room"))
    print("NEXT_CLEAR", None if r.get("next_clear") is None else {k: v for k, v in r["next_clear"].items() if k != "dump"})
    print("FURTHER", r.get("further"))
    print("CKPT", r.get("checkpoint"), r.get("checkpoint_reason"))
    print("CHECKPOINTS", r.get("checkpoints_written"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("FRAMES", r.get("frames_total"))
    print("status_claim", None)
