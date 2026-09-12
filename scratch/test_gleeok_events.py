from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.level4.boss_combat import Level4GleeokFightController

env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
env.reset()
ctl = Level4GleeokFightController(tag="test", continuous_mode=True, stand_dy=22, max_frames=3000)
total = [0]
res = ctl.run(env, None, total)
env.close()
print("ok:", res.get("ok"))
print("tf08:", res.get("tf08"))
print("frames:", res.get("frames"))
print("notes:", res.get("notes"))
for entry in res.get("log", []):
    print("log:", entry)
