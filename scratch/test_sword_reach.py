from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.level4.boss_combat import Level4GleeokFightController

for dy in range(18, 42, 2):
    env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
    env.reset()
    ctl = Level4GleeokFightController(tag="test", continuous_mode=True, stand_dy=dy, max_frames=2000)
    total = [0]
    res = ctl.run(env, None, total)
    env.close()
    dmg = res.get("dmg_events", 0)
    ok = res.get("ok", False)
    frames = res.get("frames", 0)
    err = res.get("error", "")
    print(f"stand_dy={dy:2d} (y={111+dy:3d}) -> ok={ok} dmg_events={dmg:3d} frames={frames:4d} err={err}")
