"""Power-on Survival spine through 0x7a, instrumented after the clear signal.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_l6_room7a_livespine.py

Real power-on run (same machinery as ``run_survival_spine.py --through
level6-east-key``) so combat/health/RNG match production exactly — the
``Level6Entrance.state`` dev fixture has a malformed health byte (3
containers, low-nibble 0xF) that makes Link one-shot-fragile and unusable
for combat recon. This script adds an ``on_frame`` hook that, once the room
0x7a spec reports zero live enemies, logs every 16 frames: frame, link xy,
room_item_id, keys, room_all_dead, live-enemy count, and dumps a PNG + ascii
tile map at first-clear and again at the end of the run.
"""

from __future__ import annotations

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon.tilemap import ascii_room, has_room_tile_map, tile_at_screen
from zelda_i.level6.dungeon import ROOM_7A_SPEC
from zelda_i.level6.overworld import LEVEL6, LEVEL6_EAST_KEY_ROOM
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine, spine_final_fields

TAG = "l6_7a_livespine"

_state = {
    "clear_frame": None,
    "clear_dumped": False,
    "last_log": -1,
    "fine_scan_done": False,
}


def _fine_scan(ram, cx: int, cy: int) -> None:
    print(f"  fine tile scan around ({cx},{cy}) [col-major x, rows are y]")
    for y in range(cy - 24, cy + 40, 4):
        row = []
        for x in range(cx - 24, cx + 40, 8):
            try:
                row.append(f"{tile_at_screen(ram, x, y):3x}")
            except (IndexError, ValueError):
                row.append("  ?")
        print(f"  y={y:4d}: " + " ".join(row))


def on_frame(env, obs, action, frame):
    snap = read_snapshot(env.get_ram())
    if snap.level != LEVEL6 or snap.screen != LEVEL6_EAST_KEY_ROOM:
        return
    live = ROOM_7A_SPEC.live_enemies(snap)
    if live:
        return
    if _state["clear_frame"] is None:
        _state["clear_frame"] = frame
        print(f"f={frame} CLEAR DETECTED xy=({snap.link_x},{snap.link_y}) keys={snap.keys}")
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        png = RECORDINGS_DIR / f"{TAG}_clear_f{frame}.png"
        save_rgb_png(obs, png)
        print(f"  png={png}")
        ram = env.get_ram()
        if has_room_tile_map(ram):
            print(ascii_room(ram))
        objs = [
            (o.slot, hex(o.type_id), o.x, o.y, o.hp)
            for o in snap.objects
            if o.type_id != 0
        ]
        print(f"  objs={objs}")
    if (
        not _state["fine_scan_done"]
        and _state["clear_frame"] is not None
        and frame - _state["clear_frame"] >= 300
    ):
        _state["fine_scan_done"] = True
        _fine_scan(env.get_ram(), int(snap.link_x), int(snap.link_y))
    if frame - _state["last_log"] >= 16:
        _state["last_log"] = frame
        objs = [
            (o.slot, hex(o.type_id), o.x, o.y, o.hp)
            for o in snap.objects
            if o.type_id != 0
        ]
        print(
            f"f={frame} xy=({snap.link_x},{snap.link_y}) "
            f"room_item_id=0x{snap.room_item_id:02x} keys={snap.keys} "
            f"room_all_dead={snap.room_all_dead} live={len(live)} objs={objs}"
        )


def main() -> None:
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    try:
        obs, _ = reset_obs(env)
        env = AuditedEnv(env, capabilities=AuditCapabilities.all("zelda_i.probe_l6_7a"))
        run = run_survival_spine(
            env,
            obs,
            assist=assist,
            through="level6-east-key",
            on_frame=on_frame,
        )
        final_ram = env.get_ram()
        snap = read_snapshot(final_ram)
        png = RECORDINGS_DIR / f"{TAG}_final.png"
        save_rgb_png(run.obs, png)
        print(f"final png={png}")
        if has_room_tile_map(final_ram):
            print(ascii_room(final_ram))
        print(
            f"ok={run.success} failed_stage={run.failed_stage} "
            f"final={spine_final_fields(snap, final_ram)}"
        )
        for stage in run.stages:
            if stage.name == "level6_east_key_0x7a":
                report = stage.controller.report()
                print(f"stage report: {report}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
