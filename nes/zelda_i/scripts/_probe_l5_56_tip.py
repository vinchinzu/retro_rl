"""Live L5 tip recon from Level5North56: dump 0x56, RIGHT 0x57 Zols, key-UP 0x46.

Controller-only. No key/door pokes. No bomb walls. No Whistle route.
Not Clean STATUS. Does not run east67.
"""
from __future__ import annotations

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
from zelda_i.dungeon_ids import GEL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, ZOL_OBJECT_TYPE
from zelda_i.dungeon_ops import exit_door
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

STATE_PRIMARY = "Level5North56"
STATE_FALLBACK = "Level5Entered56"
ROOM_56 = 0x56
ROOM_57 = 0x57
ROOM_46 = 0x46
CANDLE_NAMES = {0: "none", 1: "blue", 2: "red"}

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
    candle = read_u8(ram, ADDR_CANDLE)
    whistle = read_u8(ram, ADDR_WHISTLE)
    return {
        "candle_0x065B": {
            "raw": candle,
            "name": CANDLE_NAMES.get(candle, f"unknown_{candle}"),
        },
        "whistle_0x065C": int(whistle),
        "selected_item_0x0656": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "sword_0x0657": int(read_u8(ram, ADDR_SWORD)),
    }


def dump_live(snap, ram) -> dict:
    compact = compact_snapshot(snap)
    compact["doors"] = decode_doors(snap.cur_opened_doors)
    compact["doorway_mask"] = decode_doors(snap.open_doorway_mask)
    compact["room_hex"] = f"0x{snap.screen:02x}"
    compact["next_room_hex"] = f"0x{snap.next_screen:02x}"
    compact["inventory"] = inv_block(ram)
    return compact


def resolve_start_state() -> str:
    primary = GAME_DIR / "custom_integrations" / GAME / f"{STATE_PRIMARY}.state"
    if primary.exists():
        return STATE_PRIMARY
    return STATE_FALLBACK


def open_env(start_state: str, *, infinite_life: bool):
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    if assist is not None:
        assist.apply_env(env, frame=0)
    return env, assist, obs


def idle(env, assist, frames: int = 45, *, start: int = 0):
    obs = None
    for i in range(frames):
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=start + i + 1)
    return obs


def zol_spec(room_id: int, *, source_room: int, hp_positive: bool) -> DungeonRoomSpec:
    """Existing GenericDungeonRoomController Zol rule (L4 0x40 / dungeon_ops)."""
    if hp_positive:
        alive = AliveRule.TYPE_AND_HP
        type_only = (GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE)
    else:
        # Spawn HP stayed 0 (Keese-style); count type presence.
        alive = AliveRule.TYPE
        type_only = ()
    return DungeonRoomSpec(
        spec_id="level5_room57_zols_recon",
        source_room=source_room,
        room_id=room_id,
        entry=DoorRoute("RIGHT", ((120, 141), (208, 141))),
        enemy_types=(ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE),
        expected_enemy_count=5,
        alive_rule=alive,
        type_only_enemy_types=type_only,
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
        max_frames=12000,
        level=5,
    )


def classify_room(dump: dict) -> str:
    objects = dump.get("objects") or []
    combat = [
        obj
        for obj in objects
        if obj.get("type_id")
        not in (0, 0x40, 0x4E, 0x4F, 0x55, 0x56, 0x2B, 0x49, 0x5A, 0x60, 0x61, 0x62)
        and (
            (obj.get("hp") or 0) > 0
            or obj.get("type_id") in (0x13, 0x14, 0x15, 0x1B)
        )
    ]
    item_id = dump.get("room_item_id")
    item_name = dump.get("room_item_name")
    if combat and item_id not in (None, 0, 0x03):
        return f"combat+item ({len(combat)} live, room_item={item_name} 0x{item_id:02x})"
    if combat:
        return f"combat ({len(combat)} live objects)"
    if item_id not in (None, 0, 0x03):
        return f"item ({item_name} 0x{item_id:02x})"
    return "empty_or_no_spawn"


def obvious_next_door(dump: dict, *, came_from: str) -> str | None:
    """One already-open doorway that is not the entry we just used."""
    mask = dump.get("doorway_mask") or {}
    doors = dump.get("doors") or {}
    candidates = []
    for name in ("east", "west", "north", "south"):
        if name == came_from:
            continue
        if mask.get(name) or doors.get(name):
            candidates.append(
                {"east": "RIGHT", "west": "LEFT", "north": "UP", "south": "DOWN"}[name]
            )
    if len(candidates) == 1:
        return candidates[0]
    return None


def fight_zols(env, assist, *, start_frame: int) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    zols = [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id == ZOL_OBJECT_TYPE
    ]
    hp_positive = any(obj.hp > 0 for obj in zols)
    spec = zol_spec(snap.screen, source_room=ROOM_56, hp_positive=hp_positive)
    controller = GenericDungeonRoomController(spec)
    obs = None
    for frame in range(spec.max_frames):
        if assist is not None:
            assist.apply_env(env, frame=start_frame + frame)
        action = controller.step(read_snapshot(env.get_ram()))
        obs, *_ = env.step(action.action)
        if controller.success or controller.phase is DungeonPhase.FAILED:
            break
    ram = env.get_ram()
    snap = read_snapshot(ram)
    live = spec.live_enemies(snap)
    return {
        "ok": bool(controller.success) and snap.screen == ROOM_57 and not live,
        "hp_positive_at_start": hp_positive,
        "alive_rule": spec.alive_rule.value,
        "controller": controller.report(),
        "dump": dump_live(snap, ram),
        "live_zols_or_gels": len(live),
        "obs": obs,
    }


def maybe_save(env, *, name: str, start_state: str, trial: dict) -> dict:
    path = save_state(env, GAME_DIR, GAME, name)
    provenance = write_state_provenance(
        path,
        source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{start_state}.state",
        request={
            "segment": name,
            "predecessor_entry": True,
            "start_state": start_state,
            "key_poke": False,
            "door_poke": False,
        },
        selected_trial=trial,
        natural_entry=False,
    )
    return {"checkpoint": str(path), "provenance": str(provenance)}


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    start_state = resolve_start_state()
    written: list[str] = []
    commands = [
        f"uv run python nes/zelda_i/scripts/_probe_l5_56_tip.py --from-state {start_state} --infinite-life"
    ]

    # --- 1. dump 0x56 ---
    env, assist, obs = open_env(start_state, infinite_life=True)
    try:
        obs = idle(env, assist, 50)
        ram = env.get_ram()
        dump56 = dump_live(read_snapshot(ram), ram)
        png56 = RECORDINGS_DIR / "l5_56_recon.png"
        save_rgb_png(obs, png56)
        report56 = {
            "ok": dump56.get("room") == ROOM_56,
            "status_claim": None,
            "from_state": start_state,
            "pokes": False,
            "bomb_or_candle": False,
            "dump": dump56,
            "screenshot": str(png56.resolve()),
        }
        write_json_report(RECORDINGS_DIR / "l5_56_recon.json", report56)

        # --- 2. RIGHT 0x57, try Zol clear ---
        total = [50]
        right = exit_door(env, assist, total, "RIGHT")
        arrived57 = (
            right.get("changed_room")
            and right.get("after", {}).get("screen") == ROOM_57
        )
        obs = idle(env, assist, 50, start=total[0])
        total[0] += 50
        fight = None
        saved57 = None
        dump57 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        png57 = RECORDINGS_DIR / "l5_57_recon.png"
        save_rgb_png(obs, png57)
        if arrived57:
            fight = fight_zols(env, assist, start_frame=total[0])
            if fight.get("obs") is not None:
                save_rgb_png(fight["obs"], png57)
            dump57 = fight["dump"]
            if fight["ok"]:
                saved57 = maybe_save(
                    env,
                    name="Level5Cleared57",
                    start_state=start_state,
                    trial=fight["controller"],
                )
                written.append("Level5Cleared57")
        report57 = {
            "ok": bool(fight and fight["ok"]),
            "walked": arrived57,
            "cleared": bool(fight and fight["ok"]),
            "status_claim": None,
            "from_state": start_state,
            "pokes": False,
            "right_attempt": {
                "direction": "RIGHT",
                "result": right.get("result"),
                "changed_room": right.get("changed_room"),
                "dest": right.get("after", {}).get("sc"),
                "keys": right.get("after", {}).get("keys"),
            },
            "fight": None
            if fight is None
            else {k: v for k, v in fight.items() if k != "obs"},
            "dump": dump57,
            "checkpoint": None if saved57 is None else saved57,
            "screenshot": str(png57.resolve()),
        }
        write_json_report(RECORDINGS_DIR / "l5_57_recon.json", report57)
    finally:
        env.close()

    # --- 3. fresh 0x56, spend key UP to 0x46 ---
    env, assist, obs = open_env(start_state, infinite_life=True)
    try:
        obs = idle(env, assist, 40)
        total = [40]
        up = exit_door(env, assist, total, "UP")
        arrived46 = (
            up.get("changed_room") and up.get("after", {}).get("screen") == ROOM_46
        )
        obs = idle(env, assist, 70, start=total[0])
        total[0] += 70
        ram = env.get_ram()
        dump46 = dump_live(read_snapshot(ram), ram)
        png46 = RECORDINGS_DIR / "l5_46_recon.png"
        save_rgb_png(obs, png46)
        kind46 = classify_room(dump46)
        saved46 = None
        if arrived46:
            saved46 = maybe_save(
                env,
                name="Level5Entered46",
                start_state=start_state,
                trial={
                    "success": True,
                    "frames": total[0],
                    "keys": dump46.get("keys"),
                    "room": "0x46",
                },
            )
            written.append("Level5Entered46")

        # --- 4. one obvious live next door ---
        next_dir = obvious_next_door(dump46, came_from="south") if arrived46 else None
        next_room = None
        if next_dir is not None:
            nxt = exit_door(env, assist, total, next_dir)
            obs = idle(env, assist, 70, start=total[0])
            total[0] += 70
            ram = env.get_ram()
            dump_next = dump_live(read_snapshot(ram), ram)
            png_next = RECORDINGS_DIR / f"l5_{dump_next['room']:02x}_recon.png"
            save_rgb_png(obs, png_next)
            write_json_report(
                RECORDINGS_DIR / f"l5_{dump_next['room']:02x}_recon.json",
                {
                    "ok": nxt.get("changed_room"),
                    "status_claim": None,
                    "from_room": "0x46",
                    "direction": next_dir,
                    "pokes": False,
                    "dump": dump_next,
                    "screenshot": str(png_next.resolve()),
                },
            )
            next_room = {
                "direction": next_dir,
                "result": nxt.get("result"),
                "changed_room": nxt.get("changed_room"),
                "dump": dump_next,
                "kind": classify_room(dump_next),
                "screenshot": str(png_next.resolve()),
            }

        whistle = dump46.get("inventory", {}).get("whistle_0x065C")
        if next_room is not None:
            whistle = next_room["dump"].get("inventory", {}).get("whistle_0x065C")

        report46 = {
            "ok": arrived46,
            "status_claim": None,
            "from_state": start_state,
            "pokes": False,
            "up_attempt": {
                "direction": "UP",
                "result": up.get("result"),
                "changed_room": up.get("changed_room"),
                "dest": up.get("after", {}).get("sc"),
                "keys_before": up.get("before", {}).get("keys"),
                "keys_after": up.get("after", {}).get("keys"),
            },
            "kind": kind46,
            "dump": dump46,
            "checkpoint": None if saved46 is None else saved46,
            "next_room": next_room,
            "screenshot": str(png46.resolve()),
        }
        write_json_report(RECORDINGS_DIR / "l5_46_recon.json", report46)
    finally:
        env.close()

    summary = {
        "commands": commands,
        "start_state": start_state,
        "status_claim": None,
        "pokes": False,
        "room_56": report56,
        "room_57": report57,
        "room_46": report46,
        "checkpoints_written": written,
        "whistle_0x065C": whistle,
        "next_live_pin": (
            f"0x{next_room['dump']['room']:02x}"
            if next_room and next_room.get("changed_room")
            else ("0x46" if arrived46 else "0x56")
        ),
    }
    write_json_report(RECORDINGS_DIR / "l5_56_tip_summary.json", summary)
    return summary


if __name__ == "__main__":
    summary = main()
    d56 = summary["room_56"]["dump"]
    print("CMD", summary["commands"])
    print(
        "R56",
        d56.get("room_hex"),
        "xy",
        d56.get("x"),
        d56.get("y"),
        "keys",
        d56.get("keys"),
        "doors",
        d56.get("doors"),
        "doorway",
        d56.get("doorway_mask"),
        "item",
        d56.get("room_item_id"),
        d56.get("room_item_name"),
    )
    print("R56_OBJECTS", d56.get("objects"))
    print("R56_INV", d56.get("inventory"))
    r57 = summary["room_57"]
    print(
        "R57 walked",
        r57["walked"],
        "cleared",
        r57["cleared"],
        "frames",
        (r57.get("fight") or {}).get("controller", {}).get("frames"),
        "checkpoint",
        r57.get("checkpoint"),
    )
    if r57.get("fight"):
        print("R57_FIGHT", {k: v for k, v in r57["fight"].items() if k != "dump"})
    print("R57_DUMP", r57.get("dump"))
    r46 = summary["room_46"]
    print(
        "R46 arrived",
        r46["ok"],
        "kind",
        r46["kind"],
        "keys",
        r46["up_attempt"],
    )
    print("R46_DUMP", r46.get("dump"))
    print("R46_NEXT", r46.get("next_room"))
    print("CHECKPOINTS", summary["checkpoints_written"])
    print("WHISTLE", summary["whistle_0x065C"])
    print("NEXT_PIN", summary["next_live_pin"])
    print("status_claim", None)
