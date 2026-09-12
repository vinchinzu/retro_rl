from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.stepladder import make_stepladder_controller

env = make_env(GAME, "Level4Stepladder", GAME_DIR, render_mode="rgb_array")
env.reset()
snap0 = read_snapshot(env.get_ram())
print(f"Room60 start: screen=0x{snap0.screen:02x} mode={snap0.mode} xy=({snap0.link_x},{snap0.link_y}) hp={snap0.health}")

ctl = make_stepladder_controller(clear_first=False)
last_hp = snap0.health
for f in range(ctl.max_frames):
    snap = read_snapshot(env.get_ram())
    if snap.health < last_hp:
        print(f"  Hit at f={f}: {last_hp} -> {snap.health} (-{last_hp - snap.health}) link=({snap.link_x},{snap.link_y})")
        last_hp = snap.health
    act = ctl.step(snap)
    if ctl.success:
        print(f"Room60 success at f={f}! final_hp={snap.health} damage={snap0.health - snap.health}")
        break
    if snap.mode == 17:
        print(f"Dead at f={f}!")
        break
    env.step(act.action)
env.close()
