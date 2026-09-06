"""Verify the real Spectacle Rock bombable-wall object (ObjType 0x63 @ (80,160),
found live via probe_l9_05_objects.py) actually opens when bombed from the
correct stand position, before wiring a fix into Level9SpectacleRockBombController.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_05_rockwall_bomb.py
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
from zelda_i.level9.overworld import Level9PostL8OverworldController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot


def find_rockwall(snap):
    for o in snap.objects:
        if o.type_id in (0x63, 0x67):
            return o
    return None


def axis_step(snap, axis, target, tol=3):
    value = getattr(snap, f"link_{axis}")
    delta = target - value
    if abs(delta) <= tol:
        return None
    btn = "RIGHT" if delta > 0 else "LEFT"
    if axis == "y":
        btn = "DOWN" if delta > 0 else "UP"
    return nes_action(btn)


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
    assert ow.success, ow.blocked_reason

    rw = find_rockwall(read_snapshot(env.get_ram()))
    print(f"rockwall object: x={rw.x} y={rw.y} hp={rw.hp}")

    # Walk to south of the rock wall, x aligned to its column, y in the sand.
    target_x, target_y = rw.x + 8, 181
    for i in range(2000):
        snap = read_snapshot(env.get_ram())
        act = axis_step(snap, "x", target_x)
        if act is None:
            act = axis_step(snap, "y", target_y)
        if act is None:
            break
        obs, *_ = env.step(act)
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    print(f"pre-bomb stand: xy=({snap.link_x},{snap.link_y}) bombs={snap.bombs} "
          f"selected={snap.selected_item if hasattr(snap,'selected_item') else '?'}")
    png_before = RECORDINGS_DIR / "probe_l9_05_rockwall_before.png"
    save_rgb_png(obs, png_before)

    # Face UP, drop a bomb, wait for the blast.
    obs, *_ = env.step(nes_action("UP"))
    frame += 1
    assist.apply_env(env, frame=frame)
    obs, *_ = env.step(nes_action("B"))
    frame += 1
    assist.apply_env(env, frame=frame)
    bombs_before = snap.bombs
    for i in range(200):
        obs, *_ = env.step(nes_idle_action())
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    rw_after = find_rockwall(snap)
    print(f"post-bomb: bombs {bombs_before}->{snap.bombs} "
          f"rockwall={'gone' if rw_after is None else f'hp={rw_after.hp} x={rw_after.x} y={rw_after.y}'}")
    png_after = RECORDINGS_DIR / "probe_l9_05_rockwall_after.png"
    save_rgb_png(obs, png_after)
    print(f"screenshots: {png_before} {png_after}")

    # Try walking up through the (possible) opening.
    for i in range(200):
        snap = read_snapshot(env.get_ram())
        if snap.level != 0:
            print(f"ENTERED! level={snap.level} screen=0x{snap.screen:02x} mode={snap.mode}")
            break
        act = axis_step(snap, "x", target_x)
        if act is None:
            act = nes_action("UP")
        obs, *_ = env.step(act)
        frame += 1
        assist.apply_env(env, frame=frame)
    else:
        snap = read_snapshot(env.get_ram())
        print(f"did not enter; final xy=({snap.link_x},{snap.link_y}) level={snap.level}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
