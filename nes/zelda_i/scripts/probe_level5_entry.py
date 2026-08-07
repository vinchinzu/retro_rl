"""Probe Level 5 (Lizard) overworld door via Lost Hills + entry room map.

Assisted first-pass (Survival infinite-life)::

    uv run python zelda_i/scripts/probe_level5_entry.py --infinite-life --save-state

From mid-east OW (default ``At4A`` / ``OW_4A``) walks hops into Lost Hills 0x1B,
frees the east pocket, climbs four UPs to door screen 0x0B, enters dungeon,
saves ``Level5Entrance``, and probes N/E/S/W for ~400f each.

From a Lost Hills / door checkpoint::

    uv run python zelda_i/scripts/probe_level5_entry.py --from-state OW_1B_LostHills \\
        --infinite-life --save-state --tag l5_from_hills

Do **not** promote as Clean STATUS. Contract: ``docs/ASSIST_CONTRACT.md``.
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
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import OBJECT_NAMES
from zelda_i.level5_overworld import (
    LEVEL5_ENTRY_ROOM,
    LEVEL5_LEVEL_ID,
    SEGMENT_MAX_FRAMES,
    SCREEN_LEVEL5_DOOR,
    SCREEN_LOST_HILLS,
    Level5NavPhase,
    OverworldToLevel5Controller,
    level5_entrance_success,
    level5_hops_from,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.sword_cave import SwordCaveController


def _snapshot_dict(snap, env=None) -> dict:
    objs = [
        {
            "slot": o.slot,
            "type": f"0x{o.type_id:02x}",
            "name": OBJECT_NAMES.get(o.type_id, f"unk_{o.type_id:02x}"),
            "x": o.x,
            "y": o.y,
            "hp": o.hp,
        }
        for o in snap.objects
        if o.slot >= 1 and o.type_id not in (0, 0xFF) and o.y > 0
    ][:16]
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "health": snap.health,
        "hearts": f"{snap.filled_hearts}/{snap.heart_containers}",
        "sword": snap.sword,
        "bombs": snap.bombs,
        "keys": snap.keys,
        "triforce": snap.triforce,
        "doors": snap.cur_opened_doors,
        "door_mask": snap.open_doorway_mask,
        "room_all_dead": snap.room_all_dead,
        "room_obj_count": snap.room_obj_count,
        "room_item_id": snap.room_item_id,
        "objects": objs,
    }


def _probe_dirs(env, obs, assist, tag: str, max_f: int = 450) -> dict:
    """From current dungeon room, try N/E/S/W briefly; restore not done — use saves."""
    rooms: dict = {}
    start_state_name = "Level5Entrance"
    for direction, name in (
        ("RIGHT", "east"),
        ("UP", "north"),
        ("LEFT", "west"),
        ("DOWN", "south"),
    ):
        # Reload entrance each direction for isolation.
        env.close()
        env = make_env(GAME, start_state_name, GAME_DIR, render_mode="rgb_array")
        if assist is not None:
            assist_local = UnlimitedHealthAssist(enabled=True)
        else:
            assist_local = None
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist_local is not None:
            assist_local.apply_env(env, frame=0)
        start_sc = read_snapshot(env.get_ram()).screen
        hit = None
        for f in range(max_f):
            snap = read_snapshot(env.get_ram())
            if snap.level != LEVEL5_LEVEL_ID:
                hit = {
                    "left_dungeon": True,
                    "level": snap.level,
                    "screen": snap.screen,
                    "x": snap.link_x,
                    "y": snap.link_y,
                }
                break
            if (
                snap.screen != start_sc
                and snap.mode == PLAY_MODE
                and not snap.transitioning
            ):
                for _ in range(80):
                    obs, *_ = env.step(nes_idle_action())
                    if assist_local is not None:
                        assist_local.apply_env(env, frame=0)
                hit = _snapshot_dict(read_snapshot(env.get_ram()))
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_room_{snap.screen:02x}.png")
                break
            if direction == "RIGHT":
                act = (
                    nes_action("UP")
                    if snap.link_y > 170
                    else nes_action("RIGHT", "A")
                    if f % 10 < 3
                    else nes_action("RIGHT")
                )
            elif direction == "LEFT":
                act = (
                    nes_action("UP")
                    if snap.link_y > 170
                    else nes_action("LEFT", "A")
                    if f % 10 < 3
                    else nes_action("LEFT")
                )
            elif direction == "UP":
                act = (
                    nes_action("RIGHT")
                    if snap.link_x < 114
                    else nes_action("LEFT")
                    if snap.link_x > 126
                    else nes_action("UP")
                )
            else:
                act = nes_action("DOWN")
            obs, *_ = env.step(act)
            if assist_local is not None:
                assist_local.apply_env(env, frame=f)
        if hit is None:
            hit = {"failed": True, **_snapshot_dict(read_snapshot(env.get_ram()))}
        rooms[name] = hit
    return rooms, env, obs


def run_probe(
    *,
    start_state: str,
    max_frames: int,
    save_checkpoint: bool,
    tag: str,
    infinite_life: bool,
    map_rooms: bool,
    get_sword: bool,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

        if get_sword and read_snapshot(env.get_ram()).sword < 1:
            sword = SwordCaveController()
            for f in range(SWORD_MAX):
                act = sword.step(read_snapshot(env.get_ram()))
                obs, *_ = env.step(act.action)
                if assist is not None:
                    assist.apply_env(env, frame=f)
                if sword.success or sword.phase.name == "FAILED":
                    break

        entry = _snapshot_dict(read_snapshot(env.get_ram()))
        snap = read_snapshot(env.get_ram())
        hops = level5_hops_from(snap.screen)
        # If already on hills/door/dungeon, empty hops is fine.
        if snap.screen == SCREEN_LOST_HILLS:
            hops = ()
        if snap.level == LEVEL5_LEVEL_ID:
            nav = OverworldToLevel5Controller(hops=(), require_dungeon=True)
            nav.success = level5_entrance_success(env.get_ram())
            nav.phase = (
                Level5NavPhase.DONE if nav.success else Level5NavPhase.DUNGEON_SETTLE
            )
        else:
            nav = OverworldToLevel5Controller(
                hops=hops,
                require_dungeon=True,
            )
            if snap.screen == SCREEN_LOST_HILLS:
                nav.phase = Level5NavPhase.FREE_POCKET
            elif snap.screen == SCREEN_LEVEL5_DOOR:
                nav.phase = Level5NavPhase.DOOR

        trail: list[dict] = []
        last_screen = snap.screen
        frames = 0
        while frames < max_frames and not nav.success:
            snap = read_snapshot(env.get_ram())
            if snap.screen != last_screen or snap.level != entry.get("level", 0):
                trail.append({"f": frames, **_snapshot_dict(snap)})
                last_screen = snap.screen
                save_rgb_png(
                    obs, RECORDINGS_DIR / f"{tag}_sc{snap.level}_{snap.screen:02x}.png"
                )
            if snap.mode == 17:
                break
            if nav.phase.name == "FAILED":
                break
            act = nav.step(snap)
            obs, *_ = env.step(act.action)
            frames += 1
            if assist is not None:
                assist.apply_env(env, frame=frames)
            if nav.success or nav.phase.name == "FAILED":
                break

        # Settle room-ready if entered mid-scroll.
        if read_snapshot(env.get_ram()).level == LEVEL5_LEVEL_ID:
            for i in range(300):
                snap = read_snapshot(env.get_ram())
                if snap.mode == PLAY_MODE and snap.screen == LEVEL5_ENTRY_ROOM:
                    nav.success = True
                    break
                obs, *_ = env.step(nes_idle_action())
                if assist is not None:
                    assist.apply_env(env, frame=frames + i)

        snap = read_snapshot(env.get_ram())
        final = _snapshot_dict(snap)
        ok = bool(level5_entrance_success(env.get_ram()) or nav.success)
        checkpoint = None
        if ok and save_checkpoint:
            checkpoint = str(save_state(env, GAME_DIR, GAME, "Level5Entrance"))
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_entrance.png")
            # OW checkpoints along the way if still useful
            if snap.level == 0 and snap.screen == SCREEN_LEVEL5_DOOR:
                save_state(env, GAME_DIR, GAME, "OW_0B_L5Door")

        rooms = None
        if ok and map_rooms and snap.level == LEVEL5_LEVEL_ID:
            if save_checkpoint or True:
                # Ensure Level5Entrance exists for dir probes.
                save_state(env, GAME_DIR, GAME, "Level5Entrance")
            rooms, env, obs = _probe_dirs(env, obs, assist, tag=tag)
            snap = read_snapshot(env.get_ram())
            final = _snapshot_dict(snap)

        png = RECORDINGS_DIR / f"{tag}_final.png"
        save_rgb_png(obs, png)
        return {
            "ok": ok,
            "track": track,
            "infinite_life": infinite_life,
            "entry": entry,
            "trail": trail,
            "nav": nav.report(),
            "hops": [{"t": f"0x{h.target:02x}", "d": h.direction} for h in hops],
            "assist": assist.report() if assist else None,
            "final": final,
            "rooms": rooms,
            "door_screen_ow": f"0x{SCREEN_LEVEL5_DOOR:02X}",
            "lost_hills": f"0x{SCREEN_LOST_HILLS:02X}",
            "entry_room": f"0x{LEVEL5_ENTRY_ROOM:02X}",
            "screenshot": str(png),
            "checkpoint": checkpoint,
            "frames": frames,
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-state",
        default="At4A",
        help="stable-retro state (default At4A mid-east; or Level1, OW_1B_LostHills, …)",
    )
    p.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--tag", default="l5_entry")
    p.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (ASSIST_CONTRACT). Not Clean.",
    )
    p.add_argument(
        "--map-rooms",
        action="store_true",
        default=True,
        help="After entry, probe N/E/S/W (default on)",
    )
    p.add_argument("--no-map-rooms", action="store_true")
    p.add_argument(
        "--sword",
        action="store_true",
        help="Run SwordCaveController if sword==0 (for Level1 start)",
    )
    args = p.parse_args(argv)

    rep = run_probe(
        start_state=args.from_state,
        max_frames=args.max_frames,
        save_checkpoint=args.save_state,
        tag=args.tag,
        infinite_life=args.infinite_life,
        map_rooms=not args.no_map_rooms,
        get_sword=args.sword,
    )
    out = RECORDINGS_DIR / f"{args.tag}_recon.json"
    write_json_report(out, rep)
    fin = rep["final"]
    print(
        f"ok={rep['ok']} track={rep.get('track')} "
        f"sc={fin['screen']:#04x} lvl={fin['level']} mode={fin['mode']} "
        f"xy=({fin['x']},{fin['y']}) hills_ups={rep['nav'].get('hills_ups')} "
        f"ckpt={rep.get('checkpoint')}"
    )
    if rep.get("rooms"):
        for name, r in rep["rooms"].items():
            if r.get("left_dungeon"):
                print(f"  room {name}: left dungeon L{r.get('level')} sc={r.get('screen'):#04x}")
            elif r.get("failed"):
                print(f"  room {name}: fail sc={r.get('screen'):#04x}")
            else:
                objs = [o.get("type") for o in r.get("objects") or []]
                print(f"  room {name}: sc={r.get('screen'):#04x} objs={objs}")
    print(f"report={out}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
