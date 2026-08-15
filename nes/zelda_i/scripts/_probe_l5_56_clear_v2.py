"""Clear L5 0x56 3x 0x31 from Level5North56. Scripted first eat, then 2-eat cycle.

No pokes. No wall bombs. Not Clean STATUS.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon_ops import exit_door
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_SELECTED_ITEM, ADDR_WHISTLE, PLAY_MODE, ZeldaSnapshot, read_snapshot, read_u8

STATE = "Level5North56"
ROOM_56 = 0x56
TYPE_31 = 0x31
FACE_E, FACE_W, FACE_S, FACE_N = 0x01, 0x02, 0x04, 0x08
_FACE_BITS = {"LEFT": FACE_W, "RIGHT": FACE_E, "UP": FACE_N, "DOWN": FACE_S}
_RETREAT = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def decode_doors(mask: int) -> dict:
    value = int(mask) & 0x0F
    return {
        "raw": value, "raw_hex": f"0x{value:02x}",
        "east": bool(value & DoorDir.RIGHT), "west": bool(value & DoorDir.LEFT),
        "south": bool(value & DoorDir.DOWN), "north": bool(value & DoorDir.UP),
        "open": sorted(d.name for d in dirs_from_mask(value)),
    }


def dump_live(snap: ZeldaSnapshot, ram) -> dict:
    compact = compact_snapshot(snap)
    compact["doors"] = decode_doors(snap.cur_opened_doors)
    compact["doorway_mask"] = decode_doors(snap.open_doorway_mask)
    compact["room_hex"] = f"0x{snap.screen:02x}"
    compact["next_room_hex"] = f"0x{snap.next_screen:02x}"
    compact["inventory"] = {
        "selected_item_0x0656": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "bombs": int(snap.bombs), "keys": int(snap.keys),
    }
    compact["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
    return compact


def type31(snap):
    return [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == TYPE_31]


def ser31(snap):
    return [{"slot": o.slot, "hp": o.hp, "x": o.x, "y": o.y, "facing": o.facing, "state": o.state} for o in type31(snap)]


def clamp(x, y):
    return max(52, min(200, x)), max(113, min(185, y))


def walk_to(snap, tx, ty, tol=5):
    tx, ty = clamp(tx, ty)
    if abs(snap.link_x - tx) > tol:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT"), False
    if abs(snap.link_y - ty) > tol:
        return nes_action("DOWN" if snap.link_y < ty else "UP"), False
    return nes_idle_action(), True


def mouth(d):
    f = int(d.facing)
    if f & FACE_E:
        return d.x + 16, d.y, "LEFT"
    if f & FACE_W:
        return d.x - 16, d.y, "RIGHT"
    if f & FACE_S:
        return d.x, d.y + 16, "UP"
    return d.x, d.y - 16, "DOWN"


def open_env():
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def step(env, assist, total, action):
    obs, *_ = env.step(action)
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    return obs


def idle(env, assist, total, n=40):
    obs = None
    for _ in range(n):
        obs = step(env, assist, total, nes_idle_action())
    return obs


def drop_b(env, assist, total, face):
    step(env, assist, total, nes_action(face))
    step(env, assist, total, nes_idle_action())
    return step(env, assist, total, nes_action(face, "B"))


def fight(env, assist, total, *, max_frames=12000) -> dict:
    notes, events = [], []
    placed = 0
    cooldown = 0
    lock = None
    prev_n = -1
    eats = {}  # slot -> successful eat count (state 0->1 edges)
    prev_state = {}
    snap0 = read_snapshot(env.get_ram())
    bombs0 = last = snap0.bombs
    sel0 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
    obs = None
    f = 0
    # scripted: walk to spawn snout of slot 2 (south-facing at ~96,148)
    script = "goto2"
    for f in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return {"ok": False, "error": "death", "frames": f, "notes": notes, "events": events}
        if snap.screen != ROOM_56 and snap.mode == PLAY_MODE:
            obs = step(env, assist, total, nes_action("UP"))
            continue
        if snap.mode != PLAY_MODE:
            obs = step(env, assist, total, nes_idle_action())
            continue
        live = type31(snap)
        n = len(live)
        for o in live:
            ps = prev_state.get(o.slot, 0)
            if ps == 0 and o.state != 0:
                eats[o.slot] = eats.get(o.slot, 0) + 1
                notes.append(f"eat_slot{o.slot}_n{eats[o.slot]}_f{f}_b{snap.bombs}")
            prev_state[o.slot] = o.state
        if prev_n < 0:
            prev_n = n
        elif n < prev_n:
            notes.append(f"kill_to_{n}_f{f}_b{snap.bombs}")
            events.append({"event": "kill", "f": f, "n": n, "bombs": snap.bombs, "eats": dict(eats), "live": ser31(snap)})
            prev_n = n
            lock = None
            cooldown = 0
            script = "cycle"
        if snap.bombs < last:
            events.append({"event": "spent", "f": f, "bombs": f"{last}->{snap.bombs}", "xy": [snap.link_x, snap.link_y], "live": ser31(snap)})
            last = snap.bombs
        elif snap.bombs > last:
            last = snap.bombs
        if n == 0 and snap.room_all_dead >= 12:
            notes.append(f"all_dead_f{f}")
            break
        if n == 0:
            obs = step(env, assist, total, nes_idle_action())
            continue

        if script == "goto2":
            act, at = walk_to(snap, 96, 168, 4)
            if at:
                script = "drop2a"
                notes.append(f"at_spawn_snout_f{f}")
            obs = step(env, assist, total, act if not at else nes_action("UP"))
            continue
        if script == "drop2a" and snap.bombs > 0:
            obs = drop_b(env, assist, total, "UP")
            placed += 1
            cooldown = 80
            script = "cycle"
            notes.append(f"script_B1_f{f}_b{snap.bombs}")
            continue

        # prefer a half-fed living slot
        half = [o for o in live if eats.get(o.slot, 0) == 1]
        if half:
            lock = half[0].slot
        elif lock is None or all(o.slot != lock for o in live):
            lock = min(live, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y)).slot
            notes.append(f"lock{lock}_f{f}")
        t = next((o for o in live if o.slot == lock), live[0])

        if t.state != 0 or cooldown > 0:
            if cooldown > 0:
                cooldown -= 1
            ret = "DOWN" if snap.link_y < 165 else "UP"
            if snap.link_x < 52: ret = "RIGHT"
            elif snap.link_x > 200: ret = "LEFT"
            obs = step(env, assist, total, nes_action(ret))
            continue

        tx, ty, face = mouth(t)
        tx, ty = clamp(tx, ty)
        dist = abs(snap.link_x - t.x) + abs(snap.link_y - t.y)
        aligned = (
            (t.facing & (FACE_E | FACE_W) and abs(snap.link_y - t.y) <= 5)
            or (t.facing & (FACE_N | FACE_S) and abs(snap.link_x - t.x) <= 5)
        )
        in_front = (
            (t.facing & FACE_E and snap.link_x > t.x)
            or (t.facing & FACE_W and snap.link_x < t.x)
            or (t.facing & FACE_S and snap.link_y > t.y)
            or (t.facing & FACE_N and snap.link_y < t.y)
        )
        at = abs(snap.link_x - tx) <= 5 and abs(snap.link_y - ty) <= 5
        edge = t.y >= 185 or t.y <= 113 or t.x <= 52 or t.x >= 200
        if snap.bombs <= 0:
            if dist <= 20:
                obs = step(env, assist, total, nes_action(face, "A"))
            else:
                act, _ = walk_to(snap, t.x, t.y, 8)
                obs = step(env, assist, total, act)
            continue
        if at and in_front and aligned and 12 <= dist <= 22 and not edge:
            obs = drop_b(env, assist, total, face)
            placed += 1
            cooldown = 80
            notes.append(f"B_f{f}_d{dist}_s{t.slot}_tgt={t.x},{t.y}_eats={eats.get(t.slot,0)}_b{snap.bombs}")
            continue
        act, _ = walk_to(snap, tx, ty, 4)
        obs = step(env, assist, total, act)

    snap = read_snapshot(env.get_ram())
    return {
        "ok": len(type31(snap)) == 0,
        "frames": f + 1,
        "bombs_start": bombs0,
        "bombs_end": snap.bombs,
        "bombs_placed_est": placed,
        "selected_start": sel0,
        "selected_end": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
        "eats": eats,
        "live_after": ser31(snap),
        "notes": notes[-80:],
        "events": events[-50:],
        "obs": obs,
    }


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = ["PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_56_clear_v2.py"]
    env, assist, obs = open_env()
    total = [1]
    try:
        obs = idle(env, assist, total, 30)
        ram = env.get_ram()
        start = read_snapshot(ram)
        start_dump = dump_live(start, ram)
        start_doors = int(start.cur_opened_doors) & 0x0F
        start_mask = int(start.open_doorway_mask) & 0x0F
        sword_report = {"frames": 0, "success": False, "hp_min_seen": 240, "hp_still_240": True, "note": "prior 400-800f sword: 0 kills, HP=240"}

        bomb_report = fight(env, assist, total)
        if bomb_report.get("obs") is not None:
            obs = bomb_report["obs"]
        obs = idle(env, assist, total, 50)
        ram = env.get_ram()
        snap = read_snapshot(ram)
        dead = len(type31(snap)) == 0 and snap.screen == ROOM_56 and snap.mode == PLAY_MODE
        doors_now = int(snap.cur_opened_doors) & 0x0F
        mask_now = int(snap.open_doorway_mask) & 0x0F
        doors_changed = doors_now != start_doors or mask_now != start_mask
        dump56 = dump_live(snap, ram)
        png56 = RECORDINGS_DIR / "l5_56_clear.png"
        save_rgb_png(obs, png56)

        saved, reason = None, "not_saved"
        if dead and doors_changed:
            path = save_state(env, GAME_DIR, GAME, "Level5Cleared56")
            write_state_provenance(
                path,
                source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
                request={"segment": "Level5Cleared56", "start_state": STATE, "key_poke": False, "door_poke": False, "bomb_walls": False, "method": "bomb"},
                selected_trial={"success": True, "frames": total[0], "method": "bomb", "doors_before": start_doors, "doors_after": doors_now, "keys": snap.keys},
                natural_entry=False,
            )
            saved, reason = "Level5Cleared56", "three_dead_and_doors_changed"
        elif not dead:
            reason = f"type31_still_alive={len(type31(snap))}"
        else:
            reason = f"doors_unchanged_0x{doors_now:02x}"

        west_opened = bool(doors_now & DoorDir.LEFT) or bool(mask_now & DoorDir.LEFT)
        west = None
        if dead and west_opened:
            hop = exit_door(env, assist, total, "LEFT")
            idle(env, assist, total, 70)
            ram_w = env.get_ram()
            snap_w = read_snapshot(ram_w)
            obs_w, *_ = env.step(nes_idle_action())
            png_w = RECORDINGS_DIR / "l5_56_west.png"
            save_rgb_png(obs_w, png_w)
            dump_w = dump_live(snap_w, ram_w)
            west = {"walked": bool(hop.get("changed_room")), "result": hop.get("result"), "dest_room": dump_w.get("room_hex"), "dump": dump_w, "screenshot": str(png_w.resolve())}
            write_json_report(RECORDINGS_DIR / "l5_56_west.json", {"ok": bool(hop.get("changed_room")), "from_room": "0x56", "direction": "LEFT", "status_claim": None, "pokes": False, "dump": dump_w, "screenshot": str(png_w.resolve())})

        bomb_out = {k: v for k, v in bomb_report.items() if k != "obs"}
        report = {
            "ok": bool(dead and doors_changed), "status_claim": None, "from_state": STATE,
            "pokes": False, "bomb_walls": False, "commands": commands,
            "start": {"dump": start_dump, "live_type31": ser31(start), "doors": decode_doors(start_doors)},
            "sword_probe": sword_report, "bomb_combat": bomb_out, "method": "bomb",
            "three_dead": dead, "live_type31_after": ser31(snap), "frames_total": total[0],
            "post_clear_doors": decode_doors(doors_now), "post_clear_doorway_mask": decode_doors(mask_now),
            "doors_changed": doors_changed, "west_opened": west_opened, "west": west,
            "next_room": (west["dest_room"] if west and west.get("walked") else "west still sealed"),
            "checkpoint": saved, "checkpoint_reason": reason,
            "whistle_0x065C": dump56.get("whistle_0x065C"),
            "selected_item_0x0656": dump56.get("inventory", {}).get("selected_item_0x0656"),
            "dump": dump56, "screenshot": str(png56.resolve()),
        }
        write_json_report(RECORDINGS_DIR / "l5_56_clear.json", report)
        return report
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r["commands"])
    print("START_LIVE", r["start"]["live_type31"])
    print("SWORD", r["sword_probe"])
    bc = r.get("bomb_combat") or {}
    print("BOMB_OK", bc.get("ok"), "frames", bc.get("frames"), "bombs", bc.get("bombs_start"), "->", bc.get("bombs_end"), "placed", bc.get("bombs_placed_est"), "eats", bc.get("eats"))
    print("BOMB_NOTES", bc.get("notes"))
    print("BOMB_EVENTS", bc.get("events"))
    print("THREE_DEAD", r["three_dead"], "FRAMES", r["frames_total"])
    print("DOORS", r["post_clear_doors"], "MASK", r["post_clear_doorway_mask"])
    print("WEST_OPENED", r["west_opened"], "NEXT", r["next_room"])
    print("CHECKPOINT", r["checkpoint"], r["checkpoint_reason"])
    print("WHISTLE", r["whistle_0x065C"], "SELECTED", r["selected_item_0x0656"])
    print("LIVE_AFTER", r["live_type31_after"])
    print("status_claim", None)
