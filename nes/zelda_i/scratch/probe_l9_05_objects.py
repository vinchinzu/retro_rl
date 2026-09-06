"""Dump live RAM objects on OW screen 0x05 to find the real Spectacle Rock
bombable-wall object (ObjType 0x63/0x67 = UpdateRockWall in the aldonunez
zelda1-disassembly UpdateObject_JumpTable), instead of guessing coordinates
against the two large decorative mountain piles (which are terrain, not
objects -- see rr-sz8.5 notes / probe_l9_05_entry_sweep.py).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_05_objects.py
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
from zelda_i.level9.overworld import Level9PostL8OverworldController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "Level8OWLeaveLive", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)

    ow = Level9PostL8OverworldController(handoff=MEASURED_POST_L8_HANDOFF)
    ow.bind_env(env)
    frame = 0
    for _ in range(ow.max_frames):
        snap = read_snapshot(env.get_ram())
        act = ow.step(snap)
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if ow.failed or ow.success:
            break
    print(f"overworld walk: failed={ow.failed} success={ow.success} frame={frame} "
          f"screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y})")
    if ow.failed:
        return 1

    # Settle a bit and dump every live object slot.
    for _ in range(30):
        snap = read_snapshot(env.get_ram())
        obs, *_ = env.step((False,) * 8)
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    print(f"settled xy=({snap.link_x},{snap.link_y}) screen=0x{snap.screen:02x}")
    print(f"{len(snap.objects)} object slots:")
    for o in snap.objects:
        tag = ""
        if o.type_id in (0x63, 0x67):
            tag = "  <-- ROCK WALL (bombable, UpdateRockWall)"
        elif o.type_id in (0x62, 0x65, 0x66):
            tag = "  <-- rock/gravestone (push, needs bracelet)"
        elif o.type_id == 0x64:
            tag = "  <-- tree (burnable)"
        print(f"  slot={o.slot} type=0x{o.type_id:02x} x={o.x} y={o.y} "
              f"facing={o.facing} hp={o.hp} state=0x{o.state:02x}{tag}")

    png = RECORDINGS_DIR / "probe_l9_05_objects_settle.png"
    save_rgb_png(obs, png)
    print(f"screenshot: {png}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
