from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.boss_combat import Level4GleeokFightController

for test_hp in (111, 106, 104, 102, 100):
    env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.data.set_value("health", test_hp)
    ctl = Level4GleeokFightController(tag="test", continuous_mode=True, stand_dy=24, max_frames=3000)
    total = [0]
    res = ctl.run(env, None, total)
    env.close()
    ok = res.get("ok", False)
    frames = res.get("frames", 0)
    err = res.get("error", "")
    print(f"dy=24 start_hp={test_hp:3d} -> ok={ok} frames={frames:4d} err={err}")
