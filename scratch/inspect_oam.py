from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR

env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
env.reset()
for _ in range(150):
    env.step([0]*9)

ram = env.get_ram()
oam = ram[0x0200:0x0300]
print("Active sprites in OAM at frame 150:")
for i in range(0, 256, 4):
    y, tile, attr, x = oam[i:i+4]
    if y < 240:
        print(f"  sprite {i//4:2d}: xy=({x:3d},{y:3d}) tile=0x{tile:02x} attr=0x{attr:02x}")
env.close()
