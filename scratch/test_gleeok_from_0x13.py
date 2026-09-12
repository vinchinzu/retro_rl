import sys
from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.level4.boss_combat import make_gleeok_fight_controller
from zelda_i.ram import read_snapshot

def main():
    env = make_env(GAME, "Level4Entrance", GAME_DIR, render_mode="rgb_array")
    env.reset()
    with open("scratch/l4_entered_0x13.state", "rb") as f:
        state = f.read()
    env.em.set_state(state)
    snap = read_snapshot(env.get_ram())
    print(f"Loaded state: level={snap.level} screen=0x{snap.screen:02x} mode={snap.mode} xy=({snap.link_x},{snap.link_y}) hp={snap.health}")
    
    ctl = make_gleeok_fight_controller(tag="test_gleeok", continuous_mode=True)
    total = [0]
    res = ctl.run(env, None, total)
    print("Result:", res)
    env.close()

if __name__ == "__main__":
    main()
