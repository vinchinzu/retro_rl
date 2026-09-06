"""After confirming the real rock-wall bomb at (80,160) opens a hole (see
probe_l9_05_rockwall_bomb.py screenshots), sweep entry columns to find the
exact walkable x that crosses level==9.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_05_rockwall_enter.py
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
from zelda_i.level9.overworld import Level9PostL8OverworldController
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

CLEAN_STATE = "L9RockWallBlastClean"


def axis_step(snap, axis, target, tol=3):
    value = getattr(snap, f"link_{axis}")
    delta = target - value
    if abs(delta) <= tol:
        return None
    btn = "RIGHT" if delta > 0 else "LEFT"
    if axis == "y":
        btn = "DOWN" if delta > 0 else "UP"
    return nes_action(btn)


def make_clean_state() -> None:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "Level8OWLeaveLive", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    ow = Level9PostL8OverworldController(handoff=MEASURED_POST_L8_HANDOFF)
    ow.bind_env(env)
    frame = 0
    for _ in range(ow.max_frames):
        snap = read_snapshot(env.get_ram())
        act = ow.step(snap)
        env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if ow.failed or ow.success:
            break
    assert ow.success

    for i in range(2000):
        snap = read_snapshot(env.get_ram())
        act = axis_step(snap, "x", 88)
        if act is None:
            act = axis_step(snap, "y", 181)
        if act is None:
            break
        env.step(act)
        frame += 1
        assist.apply_env(env, frame=frame)

    env.step(nes_action("UP"))
    frame += 1
    assist.apply_env(env, frame=frame)
    env.step(nes_action("B"))
    frame += 1
    assist.apply_env(env, frame=frame)
    for i in range(240):
        env.step(nes_idle_action())
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    print(f"clean state settle: xy=({snap.link_x},{snap.link_y}) bombs={snap.bombs} "
          f"screen=0x{snap.screen:02x} level={snap.level}")
    save_state(env, GAME_DIR, GAME, CLEAN_STATE)
    env.close()


def try_column(x: int) -> tuple[bool, int, int]:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, CLEAN_STATE, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    frame = 0
    for i in range(2000):
        snap = read_snapshot(env.get_ram())
        if snap.level != 0:
            env.close()
            return True, snap.link_x, snap.link_y
        act = axis_step(snap, "x", x)
        if act is None:
            act = nes_action("UP")
        env.step(act)
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    env.close()
    return False, snap.link_x, snap.link_y


def main() -> int:
    make_clean_state()
    for x in range(56, 121, 4):
        entered, lx, ly = try_column(x)
        print(f"x={x}: entered={entered} final=({lx},{ly})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
