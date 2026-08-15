"""Retry 0x57 Zol clear (AliveRule.TYPE) and 0x46 item + east hop.

No pokes. No bomb walls. No Whistle route invented. Not Clean STATUS.
"""
from __future__ import annotations

from zelda_i.assist import UnlimitedHealthAssist
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
from zelda_i.dungeon_ops import exit_door, goto, idle, room_fields
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_MAP,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report

from zelda_i.door_graph.core import DoorDir, dirs_from_mask

CANDLE_NAMES = {0: "none", 1: "blue", 2: "red"}


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


def dump_live(snap, ram) -> dict:
    compact = compact_snapshot(snap)
    compact["doors"] = decode_doors(snap.cur_opened_doors)
    compact["doorway_mask"] = decode_doors(snap.open_doorway_mask)
    compact["room_hex"] = f"0x{snap.screen:02x}"
    compact["next_room_hex"] = f"0x{snap.next_screen:02x}"
    compact["candle"] = {
        "addr": "0x065B",
        "raw": int(read_u8(ram, ADDR_CANDLE)),
        "name": CANDLE_NAMES.get(int(read_u8(ram, ADDR_CANDLE)), "unknown"),
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
    }
    compact["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
    compact["item_id"] = snap.room_item_id
    compact["item_name"] = compact.get("room_item_name")
    return compact


ROOM_57 = 0x57
ZOL_TYPES = (ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE)

_PATROL = (
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


def open_env(state: str):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def wait_play(env, assist, total, room: int, max_f: int = 180) -> bool:
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.screen == room and snap.mode == PLAY_MODE and not snap.transitioning:
            idle(env, assist, total, 20)
            return True
        env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])
    return False


def type_present(snap, types) -> list:
    out = []
    for o in snap.objects:
        if 1 <= o.slot <= 12 and o.type_id in types:
            out.append(o)
    return out


def clear_57() -> dict:
    configure_headless()
    env, assist, obs = open_env("Level5North56")
    total = [1]
    try:
        idle(env, assist, total, 20)
        east = exit_door(env, assist, total, "RIGHT")
        ready = wait_play(env, assist, total, ROOM_57, 240)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        before = dump_live(snap, ram)
        if not ready or snap.screen != ROOM_57:
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / "l5_57_recon.png")
            report = {
                "walked": east.get("changed_room"),
                "cleared": False,
                "reason": "never_play_mode",
                "ready": ready,
                "east": {k: v for k, v in east.items() if k not in {"before", "at_door", "after"}},
                "dump": before,
                "checkpoint": None,
            }
            write_json_report(RECORDINGS_DIR / "l5_57_recon.json", report)
            return report

        spec = DungeonRoomSpec(
            spec_id="level5_room57_zols_type",
            source_room=ROOM_57,
            room_id=ROOM_57,
            entry=DoorRoute("RIGHT", ((208, 141),)),
            enemy_types=ZOL_TYPES,
            expected_enemy_count=5,
            alive_rule=AliveRule.TYPE,
            combat=CombatTuning(
                patrol=_PATROL,
                engage_distance=48,
                engage_attack_period=5,
                engage_attack_hold=3,
                patrol_attack_period=8,
                patrol_attack_hold=2,
            ),
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=8),
            max_frames=12000,
            level=5,
        )
        ctl = GenericDungeonRoomController(spec)
        obs = None
        for frame in range(spec.max_frames):
            snap = read_snapshot(env.get_ram())
            act = ctl.step(snap)
            obs, *_ = env.step(act.action)
            total[0] += 1
            assist.apply_env(env, frame=total[0])
            if ctl.success or ctl.phase is DungeonPhase.FAILED:
                break

        idle(env, assist, total, 30)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        dump = dump_live(snap, ram)
        live = type_present(snap, ZOL_TYPES)
        cleared = (
            snap.screen == ROOM_57
            and snap.mode == PLAY_MODE
            and not live
            and bool(ctl.success)
        )
        checkpoint = None
        if cleared:
            path = save_state(env, GAME_DIR, GAME, "Level5Cleared57")
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / "Level5North56.state"
                ),
                request={
                    "segment": "level5_clear57",
                    "predecessor_entry": True,
                    "start_state": "Level5North56",
                    "alive_rule": "type",
                },
                selected_trial={
                    "success": True,
                    "frames": ctl.frames,
                    "notes": list(ctl.notes),
                    "room": ROOM_57,
                    "keys": snap.keys,
                    "x": snap.link_x,
                    "y": snap.link_y,
                    "live_zols": 0,
                },
                natural_entry=False,
            )
            checkpoint = "Level5Cleared57"
        if obs is not None:
            save_rgb_png(obs, RECORDINGS_DIR / "l5_57_recon.png")
        report = {
            "walked": True,
            "cleared": cleared,
            "status_claim": None,
            "pokes": False,
            "alive_rule": "type",
            "controller": ctl.report() if hasattr(ctl, "report") else {
                "success": ctl.success,
                "frames": ctl.frames,
                "notes": list(ctl.notes),
                "max_live": ctl.max_live_enemies,
                "last_live": ctl.last_live_enemies,
                "phase": ctl.phase.name,
            },
            "before": before,
            "dump": dump,
            "live_after": [
                {"slot": o.slot, "type": o.type_id, "hp": o.hp, "x": o.x, "y": o.y}
                for o in live
            ],
            "checkpoint": checkpoint,
            "screenshot": str((RECORDINGS_DIR / "l5_57_recon.png").resolve()),
            "frames_total": total[0],
        }
        write_json_report(RECORDINGS_DIR / "l5_57_recon.json", report)
        return report
    finally:
        env.close()


def item_and_east() -> dict:
    configure_headless()
    env, assist, obs = open_env("Level5Entered46")
    total = [1]
    try:
        idle(env, assist, total, 20)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        start = dump_live(snap, ram)
        start_whistle = int(read_u8(ram, ADDR_WHISTLE))
        start_map = int(read_u8(ram, ADDR_MAP))
        start_item = snap.room_item_id

        # Walk to room center to collect the visible item (not a poke).
        goto(env, assist, total, 120, 141, tol=4, max_f=400)
        idle(env, assist, total, 40)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        after_item = dump_live(snap, ram)
        whistle = int(read_u8(ram, ADDR_WHISTLE))
        map_byte = int(read_u8(ram, ADDR_MAP))
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_46_item.png")

        # Natural east if doorway_mask has RIGHT (live mask 0x05). Not a bomb wall.
        east = None
        east_dump = None
        if after_item.get("doorway_mask", {}).get("east") or after_item.get("doors", {}).get("east"):
            east = exit_door(env, assist, total, "RIGHT")
            idle(env, assist, total, 50)
            wait_play(env, assist, total, read_snapshot(env.get_ram()).screen, 120)
            ram = env.get_ram()
            snap = read_snapshot(ram)
            east_dump = dump_live(snap, ram)
            obs, *_ = env.step(nes_idle_action())
            png = RECORDINGS_DIR / f"l5_{snap.screen:02x}_recon.png"
            save_rgb_png(obs, png)
            write_json_report(
                RECORDINGS_DIR / f"l5_{snap.screen:02x}_recon.json",
                {
                    "ok": east.get("changed_room"),
                    "via": "0x46 RIGHT",
                    "status_claim": None,
                    "pokes": False,
                    "dump": east_dump,
                    "screenshot": str(png.resolve()),
                },
            )

        report = {
            "start": start,
            "after_item": after_item,
            "item_pickup": {
                "start_item_id": start_item,
                "after_item_id": after_item.get("item_id"),
                "whistle_before": start_whistle,
                "whistle_after": whistle,
                "whistle_flipped": whistle != start_whistle,
                "map_before": start_map,
                "map_after": map_byte,
                "xy": [after_item.get("x"), after_item.get("y")],
            },
            "east": None if east is None else {
                "changed": east.get("changed_room"),
                "result": east.get("result"),
                "dump": east_dump,
            },
            "screenshot_item": str((RECORDINGS_DIR / "l5_46_item.png").resolve()),
            "status_claim": None,
            "pokes": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_46_item_east.json", report)
        return report
    finally:
        env.close()


def main() -> dict:
    a = clear_57()
    b = item_and_east()
    print("=== 0x57 retry ===")
    print("walked", a.get("walked"), "cleared", a.get("cleared"), "ckpt", a.get("checkpoint"))
    print("controller", a.get("controller"))
    d = a.get("dump") or {}
    print("room", d.get("room_hex"), "xy", d.get("x"), d.get("y"), "keys", d.get("keys"), "mode", d.get("mode"))
    print("live_after", a.get("live_after"))
    print("OBJECTS57", d.get("objects"))
    print("=== 0x46 item/east ===")
    print("pickup", b.get("item_pickup"))
    east = b.get("east") or {}
    ed = east.get("dump") or {}
    print("east", east.get("changed"), east.get("result"), ed.get("room_hex"), ed.get("objects"))
    print("whistle", (b.get("item_pickup") or {}).get("whistle_after"))
    return {"57": a, "46": b}


if __name__ == "__main__":
    main()
