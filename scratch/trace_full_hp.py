from retro_harness.env import make_env
from retro_harness.segment_runner import configure_headless
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.level4.spine import run_level4_entrance_tf
from zelda_i.ram import read_snapshot

configure_headless()
env = make_env(GAME, "Level4Entrance", GAME_DIR, render_mode="rgb_array")
env.reset()

last_hp = [111]

def on_frame(env, obs, action, frame):
    snap = read_snapshot(env.get_ram())
    if snap.health < last_hp[0]:
        print(f"Damage at total_f={frame}: hp {last_hp[0]} -> {snap.health} (-{last_hp[0]-snap.health}) xy=({snap.link_x},{snap.link_y}) room=0x{snap.screen:02x}")
        last_hp[0] = snap.health

res = run_level4_entrance_tf(env, through="level4-gleeok13", on_frame=on_frame)
leftover = res.get("leftover", {})
print(f"Final arrival hp={leftover.get('health')}")
env.close()
